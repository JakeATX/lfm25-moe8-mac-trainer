#!/usr/bin/env python3
"""Grouped MoE block-coordinate trainer for int8-expert MLX checkpoints.

This generalizes the layerwise runtime:

- Experts are stored as int8 outside the active group.
- A group of MoE layers is temporarily dequantized to BF16.
- The group is updated together against true next-token LM loss.
- The group is requantized immediately before the next group.

This is a semi-full-gradient approach: gradients are simultaneous within each
group, but groups are processed sequentially to stay under memory limits.
"""

from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.tuner.datasets import CacheDataset, create_dataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint, iterate_batches
from mlx_lm.utils import save_config, save_model

from layerwise_lm_update import (
    dequant_switchglu,
    packed_code_change,
    quantize_switchglu,
)


def detect_unified_memory_gb() -> float | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / 1e9
    except Exception:
        return None


def moe_layers(model) -> list[int]:
    return [
        i
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.feed_forward, "switch_mlp")
    ]


def parse_layers(model, spec: str) -> list[int]:
    available = moe_layers(model)
    if spec == "all":
        return available
    out = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    invalid = [x for x in out if x not in available]
    if invalid:
        raise SystemExit(f"Requested non-MoE layers: {invalid}; available MoE layers: {available}")
    return sorted(dict.fromkeys(out))


def make_groups(layers: list[int], group_size: int, order: str) -> list[list[int]]:
    layers = list(layers)
    if order == "reverse":
        layers = list(reversed(layers))
    elif order != "forward":
        raise SystemExit(f"Unknown order: {order}")
    return [layers[i : i + group_size] for i in range(0, len(layers), group_size)]


def load_batch(data_dir: Path, tokenizer, max_seq_length: int, prefer_long: bool):
    cfg = SimpleNamespace(mask_prompt=False)
    if prefer_long:
        rows = []
        with (data_dir / "train.jsonl").open() as f:
            for line in f:
                row = json.loads(line)
                if int(row.get("token_count", 0)) <= max_seq_length:
                    rows.append(row)
        rows.sort(key=lambda r: int(r.get("token_count", 0)), reverse=True)
        if not rows:
            raise SystemExit(f"No train examples <= {max_seq_length} tokens")
        train = create_dataset([rows[0]], tokenizer, cfg)
    else:
        train, _, _ = load_local_dataset(data_dir, tokenizer, cfg)
    train = CacheDataset(train)
    return next(
        iterate_batches(
            train,
            batch_size=1,
            max_seq_length=max_seq_length,
            loop=True,
            seed=42,
        )
    )


def save_checkpoint(path: Path, source_model: str, model, tokenizer, config):
    if path.exists():
        raise SystemExit(f"Refusing to overwrite checkpoint path: {path}")
    path.mkdir(parents=True)
    save_model(path, model, donate_model=False)
    save_config(dict(config), config_path=path / "config.json")
    tokenizer.save_pretrained(path)
    src = Path(source_model)
    if src.exists():
        for pattern in ("*.py", "generation_config.json", "chat_template.jinja", "README.md"):
            for file in glob.glob(str(src / pattern)):
                shutil.copy(file, path)


def activate_group(model, group: list[int], train_router: bool, dtype=mx.bfloat16):
    originals = {}
    for layer_idx in group:
        layer = model.model.layers[layer_idx]
        qswitch = layer.feed_forward.switch_mlp
        if not all(
            isinstance(getattr(qswitch, name), QuantizedSwitchLinear)
            for name in ("gate_proj", "up_proj", "down_proj")
        ):
            raise RuntimeError(f"Layer {layer_idx} is not fully quantized.")
        originals[layer_idx] = qswitch
        layer.feed_forward.switch_mlp = dequant_switchglu(qswitch, dtype=dtype)
        layer.feed_forward.switch_mlp.unfreeze()
        if train_router:
            layer.feed_forward.gate.unfreeze()
    mx.eval(model.parameters())
    return originals


def requantize_group(model, originals, group_size: int, bits: int, mode: str):
    changes = {}
    for layer_idx, original in originals.items():
        layer = model.model.layers[layer_idx]
        requant = quantize_switchglu(
            layer.feed_forward.switch_mlp,
            group_size=group_size,
            bits=bits,
            mode=mode,
        )
        changes[str(layer_idx)] = packed_code_change(original, requant)
        layer.feed_forward.switch_mlp = requant
    model.freeze()
    mx.clear_cache()
    return changes


def update_group(
    model,
    group: list[int],
    batch,
    lr: float,
    train_router: bool,
    q_group_size: int,
    bits: int,
    mode: str,
):
    model.freeze()
    model.train()
    originals = activate_group(model, group, train_router=train_router)
    trainable = dict(tree_flatten(model.trainable_parameters()))
    opt = optim.SGD(learning_rate=lr)
    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    tic = time.perf_counter()
    (loss_value, toks), grad = loss_value_and_grad(model, *batch)
    opt.update(model, grad)
    mx.eval(model.state, opt.state)
    elapsed = time.perf_counter() - tic
    changes = requantize_group(model, originals, q_group_size, bits, mode)
    return {
        "group": group,
        "loss": float(loss_value.item()),
        "tokens": int(toks.item()),
        "elapsed_s": elapsed,
        "trainable_param_count": sum(int(v.size) for v in trainable.values()),
        "trainable_tensor_count": len(trainable),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "code_changes": changes,
    }


def run_once(args):
    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    model.train()
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])
    selected_layers = parse_layers(model, args.layers)
    groups = make_groups(selected_layers, args.group_size, args.order)
    batch = load_batch(args.data, tokenizer, args.max_seq_length, args.prefer_long)
    loss_before, toks = default_loss(model, *batch)
    mx.eval(loss_before, toks)
    reports = []
    start = time.perf_counter()
    for group in groups:
        print(f"Updating group {group}", flush=True)
        reports.append(
            update_group(
                model,
                group,
                batch,
                args.lr,
                args.train_router,
                args.quant_group_size,
                args.bits,
                args.mode,
            )
        )
        if mx.get_peak_memory() / 1e9 > args.hard_memory_limit_gb:
            raise SystemExit(
                f"Hard memory limit exceeded: {mx.get_peak_memory()/1e9:.3f} GB > {args.hard_memory_limit_gb:.3f} GB"
            )
    loss_after, toks_after = default_loss(model, *batch)
    mx.eval(loss_after, toks_after)
    elapsed = time.perf_counter() - start
    if args.save_model:
        save_checkpoint(args.save_model, args.model, model, tokenizer, config)
    report = {
        "ok": True,
        "mode": "run",
        "model": args.model,
        "data": str(args.data),
        "save_model": str(args.save_model) if args.save_model else None,
        "unified_memory_gb": detect_unified_memory_gb(),
        "target_memory_limit_gb": args.target_memory_limit_gb,
        "hard_memory_limit_gb": args.hard_memory_limit_gb,
        "layers": selected_layers,
        "groups": groups,
        "group_size": args.group_size,
        "max_seq_length": args.max_seq_length,
        "prefer_long": args.prefer_long,
        "train_router": args.train_router,
        "loss_before": float(loss_before.item()),
        "loss_after": float(loss_after.item()),
        "tokens": int(toks.item()),
        "elapsed_s": elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "group_reports": reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--layers", default="all")
    parser.add_argument("--group-size", type=int, default=1)
    parser.add_argument("--order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--max-seq-length", type=int, default=10000)
    parser.add_argument("--prefer-long", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--quant-group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--target-memory-limit-gb", type=float, default=55.0)
    parser.add_argument("--hard-memory-limit-gb", type=float, default=60.0)
    args = parser.parse_args()
    run_once(args)


if __name__ == "__main__":
    main()
