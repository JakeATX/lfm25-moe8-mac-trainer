#!/usr/bin/env python3
"""Resumable epoch runner for grouped int8-expert MoE fine-tuning."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.datasets import CacheDataset, load_local_dataset
from mlx_lm.tuner.trainer import default_loss, iterate_batches, grad_checkpoint

from grouped_lm_train import (
    detect_unified_memory_gb,
    moe_layers,
    parse_layers,
    save_checkpoint,
    update_group,
)


def make_overlapping_groups(layers: list[int], group_size: int, stride: int, order: str):
    layers = list(layers)
    if order == "reverse":
        layers = list(reversed(layers))
    if stride <= 0:
        raise ValueError("stride must be positive")
    groups = []
    i = 0
    while i < len(layers):
        group = layers[i : i + group_size]
        if not group:
            break
        groups.append(group)
        if i + group_size >= len(layers):
            break
        i += stride
    tail = layers[-group_size:]
    if groups and groups[-1][-1] != layers[-1] and groups[-1] != tail and len(tail) > 0:
        groups.append(tail)
    return groups


def load_train(data_dir: Path, tokenizer, mask_prompt: bool):
    cfg = SimpleNamespace(mask_prompt=mask_prompt)
    train, valid, test = load_local_dataset(data_dir, tokenizer, cfg)
    return CacheDataset(train), CacheDataset(valid), CacheDataset(test)


def read_progress(path: Path):
    if not path.exists():
        return 0
    last = 0
    with path.open() as f:
        for line in f:
            if line.strip():
                last = json.loads(line)["global_step"]
    return last


def checkpoint_step(path: Path):
    match = re.fullmatch(r"step_(\d{5})(?:_final)?", path.name)
    return int(match.group(1)) if match else None


def latest_checkpoint(ckpt_dir: Path):
    candidates = []
    for path in ckpt_dir.glob("step_*"):
        step = checkpoint_step(path)
        if step is not None:
            candidates.append((step, path))
    return max(candidates, default=(0, None), key=lambda item: item[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-seq-length", type=int, default=10000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--order", choices=["forward", "reverse"], default="forward")
    parser.add_argument("--layers", default="all")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--quant-group-size", type=int, default=64)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    parser.add_argument("--save-every-steps", type=int, default=25)
    parser.add_argument("--keep-last-checkpoints", type=int, default=2)
    parser.add_argument("--target-memory-limit-gb", type=float, default=55.0)
    parser.add_argument("--hard-memory-limit-gb", type=float, default=60.0)
    parser.add_argument("--limit-steps", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-prompt", action="store_true")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    progress_path = args.run_dir / "progress.jsonl"
    config_path = args.run_dir / "run_config.json"
    if not args.resume:
        if progress_path.exists():
            raise SystemExit(f"Progress exists; pass --resume or use a new run dir: {progress_path}")
        config_path.write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    latest_step, latest_path = latest_checkpoint(ckpt_dir)
    progress_step = read_progress(progress_path) if args.resume else 0
    if args.resume and latest_path is not None:
        model_path = str(latest_path)
        completed = latest_step
        if progress_step > latest_step:
            print(
                f"Resume progress is ahead of checkpoint; resuming from checkpoint step "
                f"{latest_step} instead of progress step {progress_step}.",
                flush=True,
            )
    elif args.resume:
        model_path = args.model
        completed = 0
        if progress_step:
            print(
                f"Resume requested but no checkpoint exists; ignoring progress step {progress_step} "
                "and restarting from base model.",
                flush=True,
            )
    else:
        model_path = args.model
        completed = 0

    model, tokenizer, config = load(model_path, lazy=True, return_config=True)
    model.train()
    if args.grad_checkpoint:
        grad_checkpoint(model.layers[0])

    selected_layers = parse_layers(model, args.layers)
    groups = make_overlapping_groups(selected_layers, args.group_size, args.stride, args.order)
    train, valid, test = load_train(args.data, tokenizer, args.mask_prompt)
    batches_per_epoch = len(train)
    total_steps = args.epochs * batches_per_epoch
    if args.limit_steps:
        total_steps = min(total_steps, completed + args.limit_steps)

    batch_iter = iterate_batches(
        train,
        batch_size=1,
        max_seq_length=args.max_seq_length,
        loop=True,
        seed=args.seed,
    )
    for _ in range(completed):
        next(batch_iter)

    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "model_path": model_path,
                "completed": completed,
                "total_steps": total_steps,
                "batches_per_epoch": batches_per_epoch,
                "groups": groups,
                "unified_memory_gb": detect_unified_memory_gb(),
            },
            indent=2,
        ),
        flush=True,
    )

    for global_step in range(completed + 1, total_steps + 1):
        batch = next(batch_iter)
        epoch = (global_step - 1) // batches_per_epoch + 1
        step_in_epoch = (global_step - 1) % batches_per_epoch + 1
        step_start = time.perf_counter()
        loss_before, toks = default_loss(model, *batch)
        mx.eval(loss_before, toks)
        group_reports = []
        for group in groups:
            print(
                f"global_step={global_step} epoch={epoch} batch={step_in_epoch}/{batches_per_epoch} group={group}",
                flush=True,
            )
            group_reports.append(
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
            peak = mx.get_peak_memory() / 1e9
            if peak > args.hard_memory_limit_gb:
                raise SystemExit(
                    f"Hard memory limit exceeded: {peak:.3f} GB > {args.hard_memory_limit_gb:.3f} GB"
                )
        loss_after, toks_after = default_loss(model, *batch)
        mx.eval(loss_after, toks_after)
        elapsed = time.perf_counter() - step_start
        row = {
            "global_step": global_step,
            "epoch": epoch,
            "step_in_epoch": step_in_epoch,
            "loss_before": float(loss_before.item()),
            "loss_after": float(loss_after.item()),
            "tokens": int(toks.item()),
            "elapsed_s": elapsed,
            "peak_memory_gb": mx.get_peak_memory() / 1e9,
            "groups": groups,
            "group_reports": group_reports,
        }
        with progress_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            json.dumps(
                {
                    "global_step": row["global_step"],
                    "epoch": row["epoch"],
                    "step_in_epoch": row["step_in_epoch"],
                    "loss_before": row["loss_before"],
                    "loss_after": row["loss_after"],
                    "tokens": row["tokens"],
                    "elapsed_s": row["elapsed_s"],
                    "peak_memory_gb": row["peak_memory_gb"],
                },
                indent=2,
            ),
            flush=True,
        )
        if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
            save_checkpoint(ckpt_dir / f"step_{global_step:05d}", model_path, model, tokenizer, config)
            checkpoints = sorted(ckpt_dir.glob("step_[0-9][0-9][0-9][0-9][0-9]"))
            if args.keep_last_checkpoints > 0:
                for old in checkpoints[: -args.keep_last_checkpoints]:
                    import shutil

                    shutil.rmtree(old)

    save_checkpoint(ckpt_dir / f"step_{total_steps:05d}_final", model_path, model, tokenizer, config)


if __name__ == "__main__":
    main()
