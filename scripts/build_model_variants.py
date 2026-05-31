#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], log: Path) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        f.write("$ " + " ".join(cmd) + "\n\n")
        f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True).returncode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-root", required=True, type=Path)
    parser.add_argument("--logs", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--modes", default="8,6")
    args = parser.parse_args()

    args.out_root.mkdir(parents=True, exist_ok=True)
    summary = {"source": args.source, "variants": []}
    for bits in [int(x) for x in args.modes.split(",") if x.strip()]:
        out = args.out_root / f"mlx-{bits}bit"
        if out.exists():
            shutil.rmtree(out)
        cmd = [
            sys.executable,
            "-m",
            "mlx_lm.convert",
            "--hf-path",
            args.source,
            "--mlx-path",
            str(out),
            "--quantize",
            "--q-bits",
            str(bits),
            "--q-group-size",
            str(args.group_size),
        ]
        rc = run(cmd, args.logs / f"convert_{bits}bit.log")
        summary["variants"].append({"bits": bits, "path": str(out), "returncode": rc, "ok": rc == 0})
    (args.out_root / "variant_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if all(v["ok"] for v in summary["variants"]) else 1)


if __name__ == "__main__":
    main()
