#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--resume-adapter-file")
    parser.add_argument("--out-config", type=Path)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--fine-tune-type", choices=["lora", "dora"], default="lora")
    parser.add_argument("--mask-prompt", action="store_true")
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--val-batches", type=int, default=2)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--steps-per-report", type=int, default=10)
    parser.add_argument("--steps-per-eval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = vars(args).copy()
    if args.out_config:
        args.out_config.parent.mkdir(parents=True, exist_ok=True)
        args.out_config.write_text(json.dumps(config, indent=2, default=str), encoding="utf-8")

    cmd = [
        sys.executable,
        "-m",
        "mlx_lm.lora",
        "--model",
        args.model,
        "--train",
        "--data",
        args.data,
        "--fine-tune-type",
        args.fine_tune_type,
        "--adapter-path",
        args.adapter_path,
        "--batch-size",
        str(args.batch_size),
        "--iters",
        str(args.iters),
        "--val-batches",
        str(args.val_batches),
        "--max-seq-length",
        str(args.max_seq_length),
        "--num-layers",
        str(args.num_layers),
        "--learning-rate",
        str(args.learning_rate),
        "--steps-per-report",
        str(args.steps_per_report),
        "--steps-per-eval",
        str(args.steps_per_eval),
        "--grad-accumulation-steps",
        "1",
        "--save-every",
        str(args.save_every),
        "--grad-checkpoint",
        "--seed",
        str(args.seed),
    ]
    if args.mask_prompt:
        cmd.append("--mask-prompt")
    if args.resume_adapter_file:
        cmd.extend(["--resume-adapter-file", args.resume_adapter_file])
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w", encoding="utf-8") as f:
            f.write("$ " + " ".join(cmd) + "\n\n")
            f.flush()
            proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
