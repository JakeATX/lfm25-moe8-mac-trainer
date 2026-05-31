#!/usr/bin/env python3
"""Layerwise LM-loss update for int8-expert MoE checkpoints.

This is the first true next-token-loss path for the experimental runtime:

1. Load the mixed checkpoint with all experts stored as int8.
2. Pick one MoE layer.
3. Temporarily replace that layer's quantized SwitchGLU with a dense BF16
   SwitchGLU initialized from the int8 weights.
4. Freeze the rest of the model.
5. Run one next-token LM loss step, updating only that dense block and
   optionally its BF16 router.
6. Requantize the updated dense expert weights back into int8 SwitchLinear
   modules immediately.

This avoids `mx.gather_qmm` gradients wrt quantized expert weights and avoids a
full-model BF16 expert master copy.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchGLU
from mlx_lm.tuner.datasets import CacheDataset, create_dataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint, iterate_batches


def dequant_switchglu(qswitch, dtype=mx.bfloat16) -> SwitchGLU:
    dense = SwitchGLU(
        qswitch.gate_proj.input_dims,
        qswitch.gate_proj.output_dims,
        qswitch.gate_proj.num_experts,
        bias=False,
    )
    for name in ("gate_proj", "up_proj", "down_proj"):
        qproj = getattr(qswitch, name)
        weight = mx.dequantize(
            qproj.weight,
            qproj.scales,
            qproj.biases,
            group_size=qproj.group_size,
            bits=qproj.bits,
            mode=qproj.mode,
            dtype=dtype,
        )
        getattr(dense, name).weight = weight
    mx.eval(dense.parameters())
    return dense


def quantize_switchglu(dense: SwitchGLU, group_size: int, bits: int, mode: str) -> SwitchGLU:
    quant = SwitchGLU(
        dense.gate_proj.input_dims,
        dense.gate_proj.output_dims,
        dense.gate_proj.num_experts,
        bias=False,
    )
    quant.gate_proj = dense.gate_proj.to_quantized(group_size=group_size, bits=bits, mode=mode)
    quant.up_proj = dense.up_proj.to_quantized(group_size=group_size, bits=bits, mode=mode)
    quant.down_proj = dense.down_proj.to_quantized(group_size=group_size, bits=bits, mode=mode)
    quant.activation = dense.activation
    mx.eval(quant.parameters())
    return quant


def packed_code_change(before, after) -> dict:
    report = {}
    for name in ("gate_proj", "up_proj", "down_proj"):
        b = getattr(before, name).weight
        a = getattr(after, name).weight
        changed = int(mx.sum(b != a).item())
        total = int(b.size)
        report[name] = {
            "changed_codes": changed,
            "total_codes": total,
            "changed_fraction": changed / total,
            "saturation_rate": float(mx.mean(mx.logical_or(a == 0, a == 255)).item()),
        }
    return report


def load_first_batch(data_dir: Path, tokenizer, max_seq_length: int, prefer_long: bool):
    cfg = SimpleNamespace(mask_prompt=False)
    if prefer_long:
        rows = []
        with (data_dir / "train.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                if int(row.get("token_count", 0)) <= max_seq_length:
                    rows.append(row)
        if not rows:
            raise SystemExit(f"No train examples <= {max_seq_length} tokens")
        rows.sort(key=lambda r: int(r.get("token_count", 0)), reverse=True)
        train = create_dataset([rows[0]], tokenizer, cfg)
        valid = train
    else:
        train, valid, _ = load_local_dataset(data_dir, tokenizer, cfg)
    train = CacheDataset(train)
    valid = CacheDataset(valid)
    batch = next(
        iterate_batches(
            train,
            batch_size=1,
            max_seq_length=max_seq_length,
            loop=True,
            seed=42,
        )
    )
    return train, valid, batch


def run_one_step(args):
    model, tokenizer = load(args.model, lazy=True)
    model.train()
    train, valid, batch = load_first_batch(
        args.data, tokenizer, args.max_seq_length, args.prefer_long
    )

    layer = model.model.layers[args.layer]
    if not hasattr(layer.feed_forward, "switch_mlp"):
        raise SystemExit(f"Layer {args.layer} is not an MoE layer.")
    original_quant = layer.feed_forward.switch_mlp
    if not all(
        isinstance(getattr(original_quant, name), QuantizedSwitchLinear)
        for name in ("gate_proj", "up_proj", "down_proj")
    ):
        raise SystemExit(f"Layer {args.layer} switch_mlp is not fully quantized.")

    before_quant = original_quant
    dense = dequant_switchglu(original_quant, dtype=mx.bfloat16)

    model.freeze()
    dense.unfreeze()
    if args.train_router:
        layer.feed_forward.gate.unfreeze()
    layer.feed_forward.switch_mlp = dense

    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    trainable = dict(tree_flatten(model.trainable_parameters()))
    opt = optim.SGD(learning_rate=args.lr)

    loss_value_and_grad = nn.value_and_grad(model, default_loss)

    tic = time.perf_counter()
    loss_before, toks = default_loss(model, *batch)
    (loss_after_step, toks2), grad = loss_value_and_grad(model, *batch)
    opt.update(model, grad)
    mx.eval(model.state, opt.state)
    elapsed = time.perf_counter() - tic

    updated_dense = layer.feed_forward.switch_mlp
    requant = quantize_switchglu(
        updated_dense,
        group_size=args.group_size,
        bits=args.bits,
        mode=args.mode,
    )
    changes = packed_code_change(before_quant, requant)
    layer.feed_forward.switch_mlp = requant
    model.freeze()

    loss_after_requant, toks3 = default_loss(model, *batch)
    mx.eval(loss_before, loss_after_step, loss_after_requant, toks, toks2, toks3)

    report = {
        "ok": any(v["changed_codes"] > 0 for v in changes.values()),
        "model": args.model,
        "data": str(args.data),
        "layer": args.layer,
        "max_seq_length": args.max_seq_length,
        "prefer_long": args.prefer_long,
        "train_router": args.train_router,
        "trainable_param_count": sum(int(v.size) for v in trainable.values()),
        "trainable_tensors": list(trainable.keys()),
        "loss_before": float(loss_before.item()),
        "loss_step_graph": float(loss_after_step.item()),
        "loss_after_requant": float(loss_after_requant.item()),
        "tokens": int(toks.item()),
        "elapsed_s": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "code_changes": changes,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--layer", type=int, default=23)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--prefer-long", action="store_true")
    args = parser.parse_args()
    run_one_step(args)


if __name__ == "__main__":
    main()
