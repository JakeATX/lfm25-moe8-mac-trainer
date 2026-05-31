#!/usr/bin/env python3
"""All-MoE-layer pilot for the layerwise int8 expert training runtime."""

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
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
from mlx_lm.utils import save

from layerwise_lm_update import (
    dequant_switchglu,
    load_first_batch,
    packed_code_change,
    quantize_switchglu,
)


def moe_layers(model):
    out = []
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.feed_forward, "switch_mlp"):
            out.append(i)
    return out


def update_one_layer(model, layer_idx, batch, lr, train_router, group_size, bits, mode):
    layer = model.model.layers[layer_idx]
    original_quant = layer.feed_forward.switch_mlp
    if not all(
        isinstance(getattr(original_quant, name), QuantizedSwitchLinear)
        for name in ("gate_proj", "up_proj", "down_proj")
    ):
        raise RuntimeError(f"Layer {layer_idx} is not fully quantized.")

    dense = dequant_switchglu(original_quant, dtype=mx.bfloat16)
    model.freeze()
    dense.unfreeze()
    if train_router:
        layer.feed_forward.gate.unfreeze()
    layer.feed_forward.switch_mlp = dense
    opt = optim.SGD(learning_rate=lr)
    trainable = dict(tree_flatten(model.trainable_parameters()))

    loss_value_and_grad = nn.value_and_grad(model, default_loss)
    tic = time.perf_counter()
    (loss_value, toks), grad = loss_value_and_grad(model, *batch)
    opt.update(model, grad)
    mx.eval(model.state, opt.state)
    elapsed = time.perf_counter() - tic

    requant = quantize_switchglu(
        layer.feed_forward.switch_mlp,
        group_size=group_size,
        bits=bits,
        mode=mode,
    )
    changes = packed_code_change(original_quant, requant)
    layer.feed_forward.switch_mlp = requant
    model.freeze()
    mx.clear_cache()
    return {
        "layer": layer_idx,
        "loss": float(loss_value.item()),
        "tokens": int(toks.item()),
        "elapsed_s": elapsed,
        "trainable_param_count": sum(int(v.size) for v in trainable.values()),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "code_changes": changes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--save-model", type=Path)
    parser.add_argument("--max-seq-length", type=int, default=10000)
    parser.add_argument("--prefer-long", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--layers", default="all")
    args = parser.parse_args()

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    model.train()
    _, _, batch = load_first_batch(
        args.data, tokenizer, args.max_seq_length, args.prefer_long
    )
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    if args.layers == "all":
        layers = moe_layers(model)
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]

    loss_before, toks = default_loss(model, *batch)
    mx.eval(loss_before, toks)
    layer_reports = []
    start = time.perf_counter()
    for layer_idx in layers:
        model.train()
        print(f"Updating layer {layer_idx}...", flush=True)
        layer_reports.append(
            update_one_layer(
                model,
                layer_idx,
                batch,
                args.lr,
                args.train_router,
                args.group_size,
                args.bits,
                args.mode,
            )
        )
    loss_after, toks_after = default_loss(model, *batch)
    mx.eval(loss_after, toks_after)
    total_elapsed = time.perf_counter() - start

    if args.save_model:
        if args.save_model.exists():
            raise SystemExit(f"Refusing to overwrite save-model path: {args.save_model}")
        save(args.save_model, args.model, model, tokenizer, config)

    report = {
        "ok": True,
        "model": args.model,
        "data": str(args.data),
        "save_model": str(args.save_model) if args.save_model else None,
        "max_seq_length": args.max_seq_length,
        "prefer_long": args.prefer_long,
        "train_router": args.train_router,
        "layers": layers,
        "loss_before": float(loss_before.item()),
        "loss_after": float(loss_after.item()),
        "tokens": int(toks.item()),
        "total_elapsed_s": total_elapsed,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "layer_reports": layer_reports,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
