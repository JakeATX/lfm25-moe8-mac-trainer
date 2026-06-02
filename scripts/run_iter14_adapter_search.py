#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def run(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"\n[exit {proc.returncode}]\n\n")
        log.flush()
        return proc.returncode


def load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))["summary"]


def copy_adapter_snapshot(adapter_path: Path, snapshots_dir: Path, step: int) -> Path:
    snap = snapshots_dir / f"step_{step:04d}"
    snap.mkdir(parents=True, exist_ok=True)
    for name in ["adapters.safetensors", "adapter_config.json", "mlx_lora_config.yaml"]:
        src = adapter_path / name
        if src.exists():
            shutil.copy2(src, snap / name)
    for src in adapter_path.glob("*_adapters.safetensors"):
        shutil.copy2(src, snap / src.name)
    return snap


def decision(
    step: int,
    focused: dict[str, Any],
    broad: dict[str, Any],
    baseline_focused: dict[str, Any],
    baseline_broad: dict[str, Any],
    baseline_margins: dict[str, Any],
    margins: dict[str, Any],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    iter14 = focused["iter14"]
    acceptance = focused["iter14_acceptance"]
    if focused["failure_labels"].get("runtime_error") or broad["failure_labels"].get("runtime_error"):
        return "eval_runtime_error_stop", [
            f"focused_runtime_errors={focused['failure_labels'].get('runtime_error', 0)}",
            f"broad_runtime_errors={broad['failure_labels'].get('runtime_error', 0)}",
        ]
    if focused["text_tool_leaks"] > 0:
        reasons.append("hard_stop:text_tool_leaks")
    if focused["invented_tool_names"] > 0:
        reasons.append("hard_stop:invented_tool_names")
    if focused["no_tool_false_positive_rate"] > 0.07:
        reasons.append("hard_stop:no_tool_false_positive_gt_0_07")
    broad_normal = broad["category_metrics"].get("normal_chat", {}).get("rate", 0)
    base_normal = baseline_broad["category_metrics"].get("normal_chat", {}).get("rate", 0)
    if broad_normal < base_normal:
        reasons.append(f"hard_stop:normal_chat_regression_{broad_normal}_lt_{base_normal}")
    base_browser = baseline_focused["iter14"]["browser_pass_rate"]
    if step >= 150 and iter14["browser_pass_rate"] <= base_browser:
        reasons.append(f"hard_stop:no_browser_improvement_by_150_{iter14['browser_pass_rate']}_lte_{base_browser}")
    if reasons:
        return "reject_stop", reasons

    margin_moved = margins["positive_margin_rate"] > baseline_margins["positive_margin_rate"] or (
        margins["mean_margin_best_wrong_minus_correct"] is not None
        and baseline_margins["mean_margin_best_wrong_minus_correct"] is not None
        and margins["mean_margin_best_wrong_minus_correct"] > baseline_margins["mean_margin_best_wrong_minus_correct"]
    )
    focused_gate = all(acceptance.values())
    if focused_gate and iter14["browser_pass_rate"] >= 0.90 and broad_normal >= base_normal:
        return "focused_gate_pass", ["focused_iter14_acceptance_passed"]
    if step >= 300 and margin_moved:
        return "continue_candidate", ["margins_moved_but_acceptance_not_met"]
    if step >= 300:
        return "reject_no_movement", ["step_300_without_gate_or_margin_movement"]
    return "continue", ["continue_training"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Iter14 LoRA in checkpointed chunks with focused evals.")
    parser.add_argument("--model", default="/Users/jkooker/Documents/Codex/2026-05-28/get-this-new-model-up-and/release_work/model_runtime_step01746_pythonic")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--tools-json", type=Path, default=Path("artifacts/tool_surfaces/live_hermes_cli_tools.json"))
    parser.add_argument("--data", type=Path, default=Path("artifacts/repair_datasets/iter14_browser_x_files_contrast_router"))
    parser.add_argument("--adapter-path", type=Path, default=Path("artifacts/adapters/iter14_browser_x_files_contrast_r8"))
    parser.add_argument("--out-prefix", default="iter14_browser_x_files_contrast_r8")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--scale", type=float, default=16.0)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--chunk-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--continue-to", type=int, default=600)
    parser.add_argument("--seed", type=int, default=1414)
    parser.add_argument("--broad-limit", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.tools_json = args.tools_json.resolve()
    args.data = args.data.resolve()
    args.adapter_path = args.adapter_path.resolve()

    if args.adapter_path.exists() and not args.force:
        raise SystemExit(f"Adapter path already exists: {args.adapter_path}. Use --force to continue/overwrite intentionally.")
    args.adapter_path.mkdir(parents=True, exist_ok=True)

    results_dir = Path("artifacts/evals")
    live_results_dir = Path("artifacts/live_hermes_eval/results")
    live_reports_dir = Path("artifacts/live_hermes_eval/reports")
    logs_dir = Path("artifacts/logs")
    snapshots_dir = args.adapter_path / "snapshots"
    state_path = Path("artifacts") / f"{args.out_prefix}_state.json"
    train_log = logs_dir / f"{args.out_prefix}_train.log"
    eval_log = logs_dir / f"{args.out_prefix}_eval.log"

    state: dict[str, Any] = {
        "model": args.model,
        "endpoint": args.endpoint,
        "tools_json": str(args.tools_json),
        "data": str(args.data),
        "adapter_path": str(args.adapter_path),
        "rank": args.rank,
        "scale": args.scale,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkpoints": [],
    }

    base_focused_jsonl = live_results_dir / f"{args.out_prefix}_base_focused.jsonl"
    base_focused_summary = base_focused_jsonl.with_suffix(".summary.json")
    base_broad_jsonl = live_results_dir / f"{args.out_prefix}_base_broad_limit{args.broad_limit}.jsonl"
    base_broad_summary = base_broad_jsonl.with_suffix(".summary.json")
    base_margins_json = results_dir / f"{args.out_prefix}_base_margins.json"

    commands = [
        [
            sys.executable,
            "scripts/eval_iter14_contrast_router.py",
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
            "--tools-json",
            str(args.tools_json),
            "--out-jsonl",
            str(base_focused_jsonl),
            "--out-report",
            str(live_reports_dir / f"{args.out_prefix}_base_focused.md"),
            "--name",
            f"{args.out_prefix}_base_focused",
            "--allow-fail",
        ],
        [
            sys.executable,
            "scripts/live_hermes_eval.py",
            "--endpoint",
            args.endpoint,
            "--model",
            args.model,
            "--tools-json",
            str(args.tools_json),
            "--out-jsonl",
            str(base_broad_jsonl),
            "--out-report",
            str(live_reports_dir / f"{args.out_prefix}_base_broad_limit{args.broad_limit}.md"),
            "--name",
            f"{args.out_prefix}_base_broad_limit{args.broad_limit}",
            "--limit",
            str(args.broad_limit),
            "--allow-fail",
        ],
        [
            sys.executable,
            "scripts/diagnose_iter14_contrast_margins.py",
            "--model",
            args.model,
            "--tools-json",
            str(args.tools_json),
            "--out",
            str(base_margins_json),
        ],
    ]
    for cmd in commands:
        rc = run(cmd, eval_log)
        if rc != 0:
            raise SystemExit(f"Baseline command failed: {' '.join(cmd)}")

    baseline_focused = load_summary(base_focused_summary)
    baseline_broad = load_summary(base_broad_summary)
    baseline_margins = json.loads(base_margins_json.read_text(encoding="utf-8"))["summary"]
    state["baseline"] = {"focused": baseline_focused, "broad": baseline_broad, "margins": baseline_margins}
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    step = 0
    resume_file: Path | None = None
    accepted_candidate = False
    while step < args.max_steps:
        target_step = step + args.chunk_steps
        train_cmd = [
            sys.executable,
            "scripts/run_lora_repair.py",
            "--model",
            args.model,
            "--data",
            str(args.data),
            "--adapter-path",
            str(args.adapter_path),
            "--mask-prompt",
            "--iters",
            str(args.chunk_steps),
            "--num-layers",
            "-1",
            "--batch-size",
            "1",
            "--max-seq-length",
            str(args.max_seq_length),
            "--learning-rate",
            str(args.lr),
            "--lora-rank",
            str(args.rank),
            "--lora-scale",
            str(args.scale),
            "--val-batches",
            "2",
            "--save-every",
            str(args.chunk_steps),
            "--steps-per-report",
            "10",
            "--steps-per-eval",
            str(args.chunk_steps),
            "--seed",
            str(args.seed + step),
        ]
        if resume_file:
            train_cmd.extend(["--resume-adapter-file", str(resume_file)])
        rc = run(train_cmd, train_log)
        if rc != 0:
            state["decision"] = "training_failed"
            state["failed_step_target"] = target_step
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            raise SystemExit(rc)
        if not (args.adapter_path / "adapters.safetensors").exists():
            raise SystemExit(f"Training finished but no adapter produced at {args.adapter_path / 'adapters.safetensors'}")
        step = target_step
        resume_file = args.adapter_path / "adapters.safetensors"
        snap = copy_adapter_snapshot(args.adapter_path, snapshots_dir, step)

        focused_jsonl = live_results_dir / f"{args.out_prefix}_step{step:04d}_focused.jsonl"
        broad_jsonl = live_results_dir / f"{args.out_prefix}_step{step:04d}_broad_limit{args.broad_limit}.jsonl"
        margins_json = results_dir / f"{args.out_prefix}_step{step:04d}_margins.json"
        eval_cmds = [
            [
                sys.executable,
                "scripts/eval_iter14_contrast_router.py",
                "--endpoint",
                args.endpoint,
                "--model",
                args.model,
                "--tools-json",
                str(args.tools_json),
                "--out-jsonl",
                str(focused_jsonl),
                "--out-report",
                str(live_reports_dir / f"{args.out_prefix}_step{step:04d}_focused.md"),
                "--adapter-path",
                str(args.adapter_path),
                "--name",
                f"{args.out_prefix}_step{step:04d}_focused",
                "--allow-fail",
            ],
            [
                sys.executable,
                "scripts/live_hermes_eval.py",
                "--endpoint",
                args.endpoint,
                "--model",
                args.model,
                "--tools-json",
                str(args.tools_json),
                "--out-jsonl",
                str(broad_jsonl),
                "--out-report",
                str(live_reports_dir / f"{args.out_prefix}_step{step:04d}_broad_limit{args.broad_limit}.md"),
                "--adapter-path",
                str(args.adapter_path),
                "--name",
                f"{args.out_prefix}_step{step:04d}_broad_limit{args.broad_limit}",
                "--limit",
                str(args.broad_limit),
                "--allow-fail",
            ],
            [
                sys.executable,
                "scripts/diagnose_iter14_contrast_margins.py",
                "--model",
                args.model,
                "--tools-json",
                str(args.tools_json),
                "--out",
                str(margins_json),
                "--adapter-path",
                str(args.adapter_path),
            ],
        ]
        for cmd in eval_cmds:
            rc = run(cmd, eval_log)
            if rc != 0:
                raise SystemExit(f"Checkpoint eval failed: {' '.join(cmd)}")
        focused = load_summary(focused_jsonl.with_suffix(".summary.json"))
        broad = load_summary(broad_jsonl.with_suffix(".summary.json"))
        margins = json.loads(margins_json.read_text(encoding="utf-8"))["summary"]
        verdict, reasons = decision(step, focused, broad, baseline_focused, baseline_broad, baseline_margins, margins)
        checkpoint = {
            "step": step,
            "snapshot": str(snap),
            "focused_jsonl": str(focused_jsonl),
            "broad_jsonl": str(broad_jsonl),
            "margins_json": str(margins_json),
            "verdict": verdict,
            "reasons": reasons,
            "focused_summary": focused,
            "broad_summary": broad,
            "margin_summary": margins,
        }
        state["checkpoints"].append(checkpoint)
        state["latest_step"] = step
        state["latest_verdict"] = verdict
        state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        print(json.dumps({"step": step, "verdict": verdict, "reasons": reasons}, indent=2))
        if verdict == "focused_gate_pass":
            accepted_candidate = True
            break
        if verdict.startswith("reject"):
            break

    if accepted_candidate and step < args.continue_to:
        state["decision"] = "focused_gate_pass_adapter_candidate"
    elif state.get("latest_verdict") == "continue_candidate":
        state["decision"] = "margins_moved_acceptance_not_met_continue_manually_or_run_alt"
    else:
        state["decision"] = state.get("latest_verdict", "completed_without_acceptance")
    state["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"decision": state["decision"], "state": str(state_path)}, indent=2))


if __name__ == "__main__":
    main()
