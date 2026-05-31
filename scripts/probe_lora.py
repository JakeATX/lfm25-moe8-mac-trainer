#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], log_path: Path, timeout: int | None = None) -> dict:
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    return {"cmd": cmd, "returncode": proc.returncode, "log": str(log_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-MLX-bf16")
    parser.add_argument("--data", default="lfm25_hermes_ft/datasets/hermes_probe")
    parser.add_argument("--full-data", default="lfm25_hermes_ft/datasets/hermes_filtered_text")
    parser.add_argument("--adapter-path", default="lfm25_hermes_ft/checkpoints/probe_adapter")
    parser.add_argument("--logs", default="lfm25_hermes_ft/logs")
    args = parser.parse_args()

    data_dir = Path(args.data)
    data_dir.mkdir(parents=True, exist_ok=True)
    full_data = Path(args.full_data)
    train_rows = (full_data / "train.jsonl").read_text(encoding="utf-8").splitlines()
    valid_rows = (full_data / "valid.jsonl").read_text(encoding="utf-8").splitlines()
    if not train_rows:
        raise SystemExit("no train rows for probe")
    (data_dir / "train.jsonl").write_text(train_rows[0] + "\n", encoding="utf-8")
    (data_dir / "valid.jsonl").write_text((valid_rows[0] if valid_rows else train_rows[0]) + "\n", encoding="utf-8")
    (data_dir / "test.jsonl").write_text((valid_rows[0] if valid_rows else train_rows[0]) + "\n", encoding="utf-8")

    logs = Path(args.logs)
    logs.mkdir(parents=True, exist_ok=True)
    adapter_path = Path(args.adapter_path)
    adapter_path.mkdir(parents=True, exist_ok=True)

    results = []
    py = sys.executable
    results.append(run([
        py, "-m", "mlx_lm.generate",
        "--model", args.model,
        "--prompt", "Say hello in five words.",
        "--max-tokens", "16",
    ], logs / "probe_generate.log", timeout=600))
    if results[-1]["returncode"] != 0:
        print(json.dumps({"ok": False, "stage": "generate", "results": results}, indent=2))
        raise SystemExit(1)

    results.append(run([
        py, "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", str(data_dir),
        "--fine-tune-type", "lora",
        "--adapter-path", str(adapter_path),
        "--batch-size", "1",
        "--iters", "1",
        "--val-batches", "1",
        "--max-seq-length", "16000",
        "--num-layers", "16",
        "--learning-rate", "1e-5",
        "--steps-per-report", "1",
        "--steps-per-eval", "1",
        "--grad-checkpoint",
        "--seed", "42",
    ], logs / "probe_lora_step.log", timeout=1200))
    if results[-1]["returncode"] != 0:
        print(json.dumps({"ok": False, "stage": "lora_step", "results": results}, indent=2))
        raise SystemExit(1)

    results.append(run([
        py, "-m", "mlx_lm.generate",
        "--model", args.model,
        "--adapter-path", str(adapter_path),
        "--prompt", "Say hello in five words.",
        "--max-tokens", "16",
    ], logs / "probe_adapter_reload.log", timeout=600))
    ok = results[-1]["returncode"] == 0
    print(json.dumps({"ok": ok, "stage": "done" if ok else "adapter_reload", "results": results}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
