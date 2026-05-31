#!/usr/bin/env python3
"""Resumable multi-step layerwise trainer for int8-expert MoE checkpoints."""

from __future__ import annotations

import argparse
import json
import time
import glob
import shutil
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, grad_checkpoint, iterate_batches
from mlx_lm.utils import save_config, save_model

from layerwise_lm_pilot import moe_layers, update_one_layer


def load_train_dataset(data_dir: Path, tokenizer):
    cfg = SimpleNamespace(mask_prompt=False)
    train, valid, test = load_local_dataset(data_dir, tokenizer, cfg)
    return CacheDataset(train), CacheDataset(valid), CacheDataset(test)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--max-seq-length", type=int, default=10000)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    progress_path = args.run_dir / "progress.jsonl"
    config_path = args.run_dir / "run_config.json"
    config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    model.train()
    train, valid, test = load_train_dataset(args.data, tokenizer)
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    layers = moe_layers(model) if args.layers == "all" else [
        int(x) for x in args.layers.split(",") if x.strip()
    ]
    batches = iterate_batches(
        train,
        batch_size=1,
        max_seq_length=args.max_seq_length,
        loop=True,
        seed=args.seed,
    )

    for step in range(1, args.steps + 1):
        batch = next(batches)
        step_start = time.perf_counter()
        loss_before, toks = default_loss(model, *batch)
        mx.eval(loss_before, toks)

        layer_reports = []
        for layer_idx in layers:
            model.train()
            print(f"Step {step}/{args.steps}: updating layer {layer_idx}", flush=True)
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
        step_elapsed = time.perf_counter() - step_start
        row = {
            "step": step,
            "loss_before": float(loss_before.item()),
            "loss_after": float(loss_after.item()),
            "tokens": int(toks.item()),
            "elapsed_s": step_elapsed,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "layers": layers,
            "layer_reports": layer_reports,
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps({k: row[k] for k in ["step", "loss_before", "loss_after", "tokens", "elapsed_s", "peak_memory_gb"]}, indent=2), flush=True)

        if args.save_every > 0 and step % args.save_every == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:05d}",
                args.model,
                model,
                tokenizer,
                config,
            )

    save_checkpoint(ckpt_dir / "final", args.model, model, tokenizer, config)


if __name__ == "__main__":
    main()
