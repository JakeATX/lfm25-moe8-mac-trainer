#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None


def unified_memory_gb() -> float | None:
    raw = run(["sysctl", "-n", "hw.memsize"])
    return int(raw) / 1e9 if raw and raw.isdigit() else None


def recommended_limits(mem_gb: float | None) -> dict:
    if not mem_gb:
        return {"target_memory_limit_gb": 0, "hard_memory_limit_gb": 0, "group_sizes": [1, 2, 4]}
    hard = max(8.0, mem_gb * 0.94)
    target = max(6.0, mem_gb * 0.86)
    if mem_gb >= 96:
        sizes = [1, 2, 4, 8, 11, 16, 22]
    elif mem_gb >= 64:
        sizes = [1, 2, 4, 8, 11]
    elif mem_gb >= 48:
        sizes = [1, 2, 4, 8]
    elif mem_gb >= 32:
        sizes = [1, 2, 4]
    else:
        sizes = [1, 2]
    return {
        "target_memory_limit_gb": round(target, 2),
        "hard_memory_limit_gb": round(hard, 2),
        "group_sizes": sizes,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    mem = unified_memory_gb()
    profile = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "mac_model": run(["sysctl", "-n", "hw.model"]),
        "cpu_brand": run(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "unified_memory_gb": mem,
        "recommendation": recommended_limits(mem),
    }
    text = json.dumps(profile, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
