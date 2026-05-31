#!/usr/bin/env python3
"""Feasibility probes for trainable 8-bit MoE expert weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate, load
from mlx_lm.models.switch_layers import QuantizedSwitchLinear


@mx.custom_function
def ste_dequant_int8(q: mx.array, scale: mx.array) -> mx.array:
    return q.astype(mx.float32) * scale


@ste_dequant_int8.vjp
def ste_dequant_int8_vjp(primals, cotangents, outputs):
    q, scale = primals
    grad_q = cotangents * scale
    grad_scale = (cotangents * q.astype(mx.float32)).sum()
    return grad_q, grad_scale


class TinySTEModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.q = mx.array([[1, -2], [3, -4]], dtype=mx.int8)
        self.scale = mx.array(0.25, dtype=mx.float32)

    def __call__(self, x):
        return (x @ ste_dequant_int8(self.q, self.scale)).sum()


def generation_smoke(model_path: str, out: Path) -> dict:
    prompts = [
        "Say hello in five words.",
        "What is 18.5 multiplied by 42?",
        "Write one valid XML tag named tool_call.",
        "Name the capital of Texas.",
        "Reply with exactly three comma-separated colors.",
    ]
    model, tokenizer = load(model_path)
    results = []
    for prompt in prompts:
        text = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=32,
            verbose=False,
        )
        results.append({"prompt": prompt, "output": text})
    report = {"model": model_path, "results": results}
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out), "cases": len(results)}, indent=2))
    return report


def stock_quant_grad_probe(model_path: str, out: Path) -> dict:
    model, _ = load(model_path, lazy=True)
    target_path = None
    target_module = None
    for path, module in model.named_modules():
        if isinstance(module, QuantizedSwitchLinear):
            target_path = path
            target_module = module
            break
    if target_module is None:
        raise SystemExit("No QuantizedSwitchLinear modules found.")

    class Probe(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
            self.module.unfreeze()

        def __call__(self):
            x = mx.ones((1, 1, 1, target_module.input_dims), dtype=mx.bfloat16)
            idx = mx.array([[[0]]])
            return self.module(x, idx).astype(mx.float32).sum()

    probe = Probe(target_module)
    report = {"target_path": target_path, "ok": False, "error": None}
    try:
        value, grad = nn.value_and_grad(probe, lambda m: m())(probe)
        report["ok"] = True
        report["value"] = float(value.item())
        report["grad_keys"] = list(grad.keys())
        report["grad_dtypes"] = {k: str(v.dtype) for k, v in grad.items()}
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def ste_toy_probe(out: Path) -> dict:
    model = TinySTEModel()
    x = mx.ones((1, 2), dtype=mx.float32)
    value, grad = nn.value_and_grad(model, lambda m, inp: m(inp))(model, x)
    report = {
        "ok": True,
        "value": float(value.item()),
        "grad_dtypes": {k: str(v.dtype) for k, v in grad.items()},
        "grad_shapes": {k: list(v.shape) for k, v in grad.items()},
        "note": "This proves MLX custom VJP can return float STE gradients for int8 arrays, but not that full LFM2 experts fit.",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def memory_estimate(out: Path) -> dict:
    expert_params = 7_751_073_792
    non_expert_params = 716_783_040
    report = {
        "expert_params": expert_params,
        "non_expert_params": non_expert_params,
        "stored_weights_gb": {
            "experts_int8": expert_params / 1e9,
            "non_experts_bf16": non_expert_params * 2 / 1e9,
            "scale_metadata_estimate": [0.25, 0.50],
            "total_estimate": [9.43, 9.68],
        },
        "training_gradient_pressure_gb": {
            "expert_grad_float32_if_naive_ste": expert_params * 4 / 1e9,
            "expert_grad_bf16_if_supported": expert_params * 2 / 1e9,
            "expert_grad_int8_sign_if_custom_optimizer": expert_params / 1e9,
        },
        "blocker": "A full-model value_and_grad path must store expert gradients. Naive STE float gradients alone are ~31 GB for experts.",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    smoke = sub.add_parser("generate-smoke")
    smoke.add_argument("--model", required=True)
    smoke.add_argument("--out", required=True, type=Path)

    grad = sub.add_parser("stock-quant-grad")
    grad.add_argument("--model", required=True)
    grad.add_argument("--out", required=True, type=Path)

    toy = sub.add_parser("ste-toy")
    toy.add_argument("--out", required=True, type=Path)

    mem = sub.add_parser("memory-estimate")
    mem.add_argument("--out", required=True, type=Path)

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "generate-smoke":
        generation_smoke(args.model, args.out)
    elif args.cmd == "stock-quant-grad":
        stock_quant_grad_probe(args.model, args.out)
    elif args.cmd == "ste-toy":
        ste_toy_probe(args.out)
    elif args.cmd == "memory-estimate":
        memory_estimate(args.out)


if __name__ == "__main__":
    main()
