#!/usr/bin/env python3
"""Create an MLX checkpoint with only MoE expert SwitchLinear layers quantized.

This is Gate 1 for the experimental runtime. It intentionally quantizes only
the expert MLP projections and leaves routers / attention / conv / embeddings
in the source dtype.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
from mlx_lm.utils import save
from mlx.utils import tree_unflatten


EXPERT_MARKERS = (
    "feed_forward.switch_mlp.gate_proj",
    "feed_forward.switch_mlp.up_proj",
    "feed_forward.switch_mlp.down_proj",
)


def is_expert_switch(path: str, module: nn.Module) -> bool:
    return isinstance(module, SwitchLinear) and any(m in path for m in EXPERT_MARKERS)


def quantize_expert_switches(model: nn.Module, group_size: int, bits: int, mode: str):
    converted = []
    replacements = []

    def collect(path: str, module: nn.Module):
        if is_expert_switch(path, module):
            converted.append(
                {
                    "path": path,
                    "shape": list(module.weight.shape),
                    "params": int(module.weight.size),
                }
            )
            replacements.append(
                (
                    path,
                    module.to_quantized(group_size=group_size, bits=bits, mode=mode),
                )
            )

    model.apply_to_modules(collect)
    model.update_modules(tree_unflatten(replacements))
    return converted


def count_modules(model: nn.Module):
    counts = {"quantized_expert_switches": 0, "bf16_switches": 0, "other": 0}
    for path, module in model.named_modules():
        if any(m in path for m in EXPERT_MARKERS):
            if isinstance(module, QuantizedSwitchLinear):
                counts["quantized_expert_switches"] += 1
            elif isinstance(module, SwitchLinear):
                counts["bf16_switches"] += 1
        else:
            counts["other"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-MLX-bf16")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--mode", default="affine")
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()

    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer, config = load(args.model, lazy=True, return_config=True)
    before = count_modules(model)
    converted = quantize_expert_switches(model, args.group_size, args.bits, args.mode)
    after = count_modules(model)

    config = dict(config)
    config["moe8_runtime"] = {
        "expert_quantization": {
            "bits": args.bits,
            "group_size": args.group_size,
            "mode": args.mode,
            "paths": [c["path"] for c in converted],
        },
        "non_expert_dtype": config.get("dtype") or config.get("torch_dtype"),
    }
    config["quantization"] = {
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": args.mode,
    }
    config["quantization"].update({
        c["path"]: {
            "bits": args.bits,
            "group_size": args.group_size,
            "mode": args.mode,
        }
        for c in converted
    })
    config["quantization_config"] = config["quantization"]

    save(args.out, args.model, model, tokenizer, config)
    mx.eval(model.parameters())

    manifest = {
        "model": args.model,
        "output": str(args.out),
        "bits": args.bits,
        "group_size": args.group_size,
        "mode": args.mode,
        "before": before,
        "after": after,
        "converted_count": len(converted),
        "converted_params": sum(c["params"] for c in converted),
        "converted": converted,
    }
    manifest_path = args.manifest or args.out / "moe8_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Keep a convenient top-level copy for later gates when a separate manifest
    # path was requested.
    if manifest_path.resolve() != (args.out / "moe8_manifest.json").resolve():
        shutil.copy2(manifest_path, args.out / "moe8_manifest.json")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
