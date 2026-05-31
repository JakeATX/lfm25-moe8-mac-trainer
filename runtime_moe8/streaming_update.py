#!/usr/bin/env python3
"""Streaming-ish trainable int8 expert projection experiments.

This module deliberately avoids mx.gather_qmm for the trainable path. It
dequantizes and updates one selected expert matrix at a time, then requantizes
that matrix immediately. This is not yet a full LFM2 training loop; it is the
primitive needed before the full loop is credible.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.switch_layers import QuantizedSwitchLinear
from mlx_lm.models.activations import swiglu


@dataclass
class UpdateStats:
    expert: int
    changed_codes: int
    total_codes: int
    saturation_rate: float
    loss_before: float
    loss_after: float


def _pack_single_expert(weight: mx.array, group_size: int, bits: int, mode: str):
    q, scales, *biases = mx.quantize(weight, group_size=group_size, bits=bits, mode=mode)
    return q, scales, biases[0] if biases else None


def _dequant_single_expert(module: QuantizedSwitchLinear, expert: int, dtype=mx.float32):
    q = module.weight[expert : expert + 1]
    scales = module.scales[expert : expert + 1]
    biases = None if module.biases is None else module.biases[expert : expert + 1]
    return mx.dequantize(
        q,
        scales,
        biases,
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
        dtype=dtype,
    )[0]


def _replace_expert(module: QuantizedSwitchLinear, expert: int, new_weight: mx.array):
    q, scales, biases = _pack_single_expert(
        new_weight[None, :, :],
        group_size=module.group_size,
        bits=module.bits,
        mode=module.mode,
    )
    before = module.weight[expert]
    changed = int(mx.sum(before != q[0]).item())
    total = int(before.size)
    module.weight = mx.concatenate(
        [module.weight[:expert], q, module.weight[expert + 1 :]], axis=0
    )
    module.scales = mx.concatenate(
        [module.scales[:expert], scales, module.scales[expert + 1 :]], axis=0
    )
    if biases is not None:
        module.biases = mx.concatenate(
            [module.biases[:expert], biases, module.biases[expert + 1 :]], axis=0
        )
    sat = float(mx.mean(mx.logical_or(q[0] == 0, q[0] == 255)).item())
    mx.eval(module.weight, module.scales)
    return changed, total, sat


def expert_sgd_update(
    module: QuantizedSwitchLinear,
    expert: int,
    x: mx.array,
    target: mx.array,
    lr: float,
) -> UpdateStats:
    """Update one expert matrix for a simple projection MSE objective."""

    w0 = _dequant_single_expert(module, expert, dtype=mx.float32)

    def loss_fn(w):
        pred = x @ w.T
        return mx.mean(mx.square(pred - target))

    loss_before, grad = mx.value_and_grad(loss_fn)(w0)
    w1 = w0 - lr * grad
    changed, total, sat = _replace_expert(module, expert, w1)
    w_after = _dequant_single_expert(module, expert, dtype=mx.float32)
    loss_after = loss_fn(w_after)
    mx.eval(loss_before, loss_after)
    return UpdateStats(
        expert=expert,
        changed_codes=changed,
        total_codes=total,
        saturation_rate=sat,
        loss_before=float(loss_before.item()),
        loss_after=float(loss_after.item()),
    )


def find_first_quantized_expert(model):
    for path, module in model.named_modules():
        if isinstance(module, QuantizedSwitchLinear):
            return path, module
    raise RuntimeError("No QuantizedSwitchLinear found")


def toy_projection_probe(out: Path, lr: float = 1e-3):
    module = QuantizedSwitchLinear(64, 32, 4, bias=False, group_size=64, bits=8)
    expert = 2
    x = mx.random.normal((8, 64)).astype(mx.float32)
    true_w = mx.random.normal((32, 64)).astype(mx.float32) * 0.02
    target = x @ true_w.T
    stats = expert_sgd_update(module, expert, x, target, lr)
    report = {
        "ok": stats.changed_codes > 0,
        "stats": stats.__dict__,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def checkpoint_projection_probe(model_path: str, out: Path, lr: float = 1e-6):
    model, _ = load(model_path, lazy=True)
    path, module = find_first_quantized_expert(model)
    expert = 0
    x = mx.random.normal((4, module.input_dims)).astype(mx.float32) * 0.01
    w = _dequant_single_expert(module, expert, dtype=mx.float32)
    target = (x @ w.T) + 0.001
    stats = expert_sgd_update(module, expert, x, target, lr)
    report = {
        "ok": stats.changed_codes > 0,
        "module_path": path,
        "stats": stats.__dict__,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "note": "This updates one quantized expert projection outside mx.gather_qmm and requantizes immediately.",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def switchglu_three_projection_probe(model_path: str, out: Path, lr: float = 1e-6):
    """Update gate/up/down for one expert in a real LFM2 SwitchGLU block.

    This uses normal dense matmul on one selected expert at a time and computes
    gradients for only that expert's three matrices. It is the smallest useful
    routed-MoE training primitive before integrating with a full model loss.
    """
    model, _ = load(model_path, lazy=True)
    block_path = None
    switch = None
    for path, module in model.named_modules():
        if path.endswith("feed_forward.switch_mlp"):
            if all(
                isinstance(getattr(module, name), QuantizedSwitchLinear)
                for name in ("gate_proj", "up_proj", "down_proj")
            ):
                block_path = path
                switch = module
                break
    if switch is None:
        raise SystemExit("No fully quantized SwitchGLU block found.")

    expert = 0
    hidden = switch.gate_proj.input_dims
    inner = switch.gate_proj.output_dims
    x = mx.random.normal((16, hidden)).astype(mx.float32) * 0.01
    target = mx.random.normal((16, hidden)).astype(mx.float32) * 0.01

    wg0 = _dequant_single_expert(switch.gate_proj, expert, dtype=mx.float32)
    wu0 = _dequant_single_expert(switch.up_proj, expert, dtype=mx.float32)
    wd0 = _dequant_single_expert(switch.down_proj, expert, dtype=mx.float32)

    def loss_fn(wg, wu, wd):
        gate = x @ wg.T
        up = x @ wu.T
        out = swiglu(gate, up) @ wd.T
        return mx.mean(mx.square(out - target))

    loss_before, grads = mx.value_and_grad(loss_fn, argnums=[0, 1, 2])(wg0, wu0, wd0)
    gg, gu, gd = grads
    wg1 = wg0 - lr * gg
    wu1 = wu0 - lr * gu
    wd1 = wd0 - lr * gd

    changed = {}
    for name, module, new_w in (
        ("gate_proj", switch.gate_proj, wg1),
        ("up_proj", switch.up_proj, wu1),
        ("down_proj", switch.down_proj, wd1),
    ):
        c, total, sat = _replace_expert(module, expert, new_w)
        changed[name] = {"changed_codes": c, "total_codes": total, "saturation_rate": sat}

    wg2 = _dequant_single_expert(switch.gate_proj, expert, dtype=mx.float32)
    wu2 = _dequant_single_expert(switch.up_proj, expert, dtype=mx.float32)
    wd2 = _dequant_single_expert(switch.down_proj, expert, dtype=mx.float32)
    loss_after = loss_fn(wg2, wu2, wd2)
    mx.eval(loss_before, loss_after)
    report = {
        "ok": any(v["changed_codes"] > 0 for v in changed.values()),
        "block_path": block_path,
        "expert": expert,
        "input_shape": list(x.shape),
        "loss_before": float(loss_before.item()),
        "loss_after": float(loss_after.item()),
        "changed": changed,
        "grad_shapes": {
            "gate_proj": list(gg.shape),
            "up_proj": list(gu.shape),
            "down_proj": list(gd.shape),
        },
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "note": "Updates one expert's gate/up/down matrices without mx.gather_qmm.",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def routed_micro_train_probe(
    model_path: str,
    out: Path,
    lr: float = 1e-6,
    steps: int = 20,
    tokens: int = 32,
):
    """Route-aware micro training for one real MoE block.

    This does not train against language-model loss yet. It validates the
    memory/update mechanics we need: use the block router to choose experts,
    then update only selected experts with immediate requantization.
    """
    model, _ = load(model_path, lazy=True)
    block = None
    block_path = None
    for path, module in model.named_modules():
        if path.endswith("feed_forward") and hasattr(module, "switch_mlp"):
            if all(
                isinstance(getattr(module.switch_mlp, name), QuantizedSwitchLinear)
                for name in ("gate_proj", "up_proj", "down_proj")
            ):
                block = module
                block_path = path
                break
    if block is None:
        raise SystemExit("No quantized MoE feed_forward block found.")

    hidden = block.switch_mlp.gate_proj.input_dims
    losses = []
    changed_total = 0
    touched = {}

    for step in range(1, steps + 1):
        x = mx.random.normal((tokens, hidden)).astype(mx.bfloat16) * 0.01
        gates = mx.softmax(block.gate(x).astype(mx.float32), axis=-1)
        inds = mx.argpartition(gates, kth=-block.top_k, axis=-1)[..., -block.top_k :]
        mx.eval(inds)
        selected = sorted(set(int(v) for v in mx.flatten(inds).tolist()))
        # A synthetic local target keeps the probe independent of full LM loss
        # while still exercising routed expert updates.
        target = mx.random.normal((tokens, hidden)).astype(mx.float32) * 0.01

        step_loss = 0.0
        for expert in selected:
            mask = mx.any(inds == expert, axis=-1)
            positions = [i for i, active in enumerate(mask.tolist()) if active]
            if not positions:
                continue
            x_e = x[positions].astype(mx.float32)
            target_e = target[positions]

            wg0 = _dequant_single_expert(block.switch_mlp.gate_proj, expert, dtype=mx.float32)
            wu0 = _dequant_single_expert(block.switch_mlp.up_proj, expert, dtype=mx.float32)
            wd0 = _dequant_single_expert(block.switch_mlp.down_proj, expert, dtype=mx.float32)

            def loss_fn(wg, wu, wd):
                gate = x_e @ wg.T
                up = x_e @ wu.T
                out = swiglu(gate, up) @ wd.T
                return mx.mean(mx.square(out - target_e))

            loss_before, grads = mx.value_and_grad(loss_fn, argnums=[0, 1, 2])(
                wg0, wu0, wd0
            )
            gg, gu, gd = grads
            for name, module, new_w in (
                ("gate_proj", block.switch_mlp.gate_proj, wg0 - lr * gg),
                ("up_proj", block.switch_mlp.up_proj, wu0 - lr * gu),
                ("down_proj", block.switch_mlp.down_proj, wd0 - lr * gd),
            ):
                changed, _, _ = _replace_expert(module, expert, new_w)
                changed_total += changed
                touched[f"{expert}:{name}"] = touched.get(f"{expert}:{name}", 0) + changed
            step_loss += float(loss_before.item())

        losses.append(step_loss / max(1, len(selected)))

    report = {
        "ok": changed_total > 0,
        "block_path": block_path,
        "steps": steps,
        "tokens_per_step": tokens,
        "losses": losses,
        "changed_total": changed_total,
        "touched_projection_count": len(touched),
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "note": "Route-aware micro trainer updates only selected experts, but uses synthetic local loss rather than LM loss.",
    }
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    toy = sub.add_parser("toy-projection")
    toy.add_argument("--out", required=True, type=Path)
    toy.add_argument("--lr", type=float, default=1e-3)

    ckpt = sub.add_parser("checkpoint-projection")
    ckpt.add_argument("--model", required=True)
    ckpt.add_argument("--out", required=True, type=Path)
    ckpt.add_argument("--lr", type=float, default=1e-6)

    glu = sub.add_parser("switchglu-three-projection")
    glu.add_argument("--model", required=True)
    glu.add_argument("--out", required=True, type=Path)
    glu.add_argument("--lr", type=float, default=1e-6)

    routed = sub.add_parser("routed-micro-train")
    routed.add_argument("--model", required=True)
    routed.add_argument("--out", required=True, type=Path)
    routed.add_argument("--lr", type=float, default=1e-6)
    routed.add_argument("--steps", type=int, default=20)
    routed.add_argument("--tokens", type=int, default=32)

    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.cmd == "toy-projection":
        toy_projection_probe(args.out, args.lr)
    elif args.cmd == "checkpoint-projection":
        checkpoint_projection_probe(args.model, args.out, args.lr)
    elif args.cmd == "switchglu-three-projection":
        switchglu_three_projection_probe(args.model, args.out, args.lr)
    elif args.cmd == "routed-micro-train":
        routed_micro_train_probe(args.model, args.out, args.lr, args.steps, args.tokens)


if __name__ == "__main__":
    main()
