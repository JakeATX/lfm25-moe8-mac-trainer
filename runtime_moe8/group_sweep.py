#!/usr/bin/env python3
"""Sweep grouped MoE layer sizes in isolated subprocesses."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import os
from pathlib import Path


def detect_unified_memory_gb() -> float | None:
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        return int(out) / 1e9
    except Exception:
        return None


def auto_group_sizes() -> str:
    mem = detect_unified_memory_gb()
    if mem is None:
        return "1,2,4"
    if mem >= 96:
        return "1,2,4,8,11,16,22"
    if mem >= 64:
        return "1,2,4,8,11"
    if mem >= 48:
        return "1,2,4,8"
    if mem >= 32:
        return "1,2,4"
    return "1,2"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--group-sizes", default="auto")
    parser.add_argument("--max-seq-length", type=int, default=10000)
    parser.add_argument("--prefer-long", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-7)
    parser.add_argument("--target-memory-limit-gb", type=float, default=55.0)
    parser.add_argument("--hard-memory-limit-gb", type=float, default=60.0)
    parser.add_argument("--timeout-s", type=int, default=900)
    parser.add_argument("--train-router", action="store_true")
    parser.add_argument("--grad-checkpoint", action="store_true")
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    group_sizes = auto_group_sizes() if args.group_sizes == "auto" else args.group_sizes
    for size in [int(x) for x in group_sizes.split(",") if x.strip()]:
        report = args.work_dir / f"group_{size}.json"
        log = args.work_dir / f"group_{size}.log"
        cmd = [
            sys.executable,
            str(Path(__file__).with_name("grouped_lm_train.py")),
            "--model",
            args.model,
            "--data",
            args.data,
            "--out",
            str(report),
            "--group-size",
            str(size),
            "--max-seq-length",
            str(args.max_seq_length),
            "--lr",
            str(args.lr),
            "--target-memory-limit-gb",
            str(args.target_memory_limit_gb),
            "--hard-memory-limit-gb",
            str(args.hard_memory_limit_gb),
        ]
        if args.prefer_long:
            cmd.append("--prefer-long")
        if args.train_router:
            cmd.append("--train-router")
        if args.grad_checkpoint:
            cmd.append("--grad-checkpoint")

        print(f"Running group size {size}: {' '.join(cmd)}", flush=True)
        start = time.perf_counter()
        with log.open("w") as lf:
            proc = subprocess.run(
                cmd,
                stdout=lf,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_s,
                env={**os.environ, "PYTHONPATH": str(Path(__file__).parent)},
            )
        elapsed = time.perf_counter() - start
        entry = {
            "group_size": size,
            "returncode": proc.returncode,
            "elapsed_s": elapsed,
            "report": str(report),
            "log": str(log),
            "ok": False,
            "peak_memory_gb": None,
            "loss_before": None,
            "loss_after": None,
        }
        if report.exists():
            try:
                data = json.load(report.open())
                entry.update(
                    {
                        "ok": bool(data.get("ok")) and proc.returncode == 0,
                        "peak_memory_gb": data.get("peak_memory_gb"),
                        "loss_before": data.get("loss_before"),
                        "loss_after": data.get("loss_after"),
                        "tokens": data.get("tokens"),
                    }
                )
            except Exception as exc:
                entry["parse_error"] = repr(exc)
        results.append(entry)
        if proc.returncode != 0:
            print(f"Group size {size} failed; stopping sweep.", flush=True)
            break
        if entry["peak_memory_gb"] and entry["peak_memory_gb"] > args.target_memory_limit_gb:
            print(f"Group size {size} exceeded target memory; stopping sweep.", flush=True)
            break

    successful = [
        r
        for r in results
        if r.get("ok")
        and r.get("peak_memory_gb") is not None
        and r["peak_memory_gb"] <= args.target_memory_limit_gb
    ]
    summary = {
        "results": results,
        "recommended_group_size": max([r["group_size"] for r in successful], default=None),
        "target_memory_limit_gb": args.target_memory_limit_gb,
        "hard_memory_limit_gb": args.hard_memory_limit_gb,
        "unified_memory_gb": detect_unified_memory_gb(),
        "group_sizes_requested": group_sizes,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
