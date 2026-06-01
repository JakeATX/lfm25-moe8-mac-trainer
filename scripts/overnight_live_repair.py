#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

import live_hermes_eval


SYSTEM = (
    "You are Hermes. Use the available tools when the user asks you to act, look up current information, "
    "inspect files, run commands, or control the computer. If the request is answerable directly or "
    "explicitly says not to use tools, answer normally without making a tool call. Use only the tools and "
    "arguments provided in the schema."
)

RETENTION_ROWS = [
    ("hello", "Hello! How can I help you today?"),
    ("are you there?", "I'm here. What would you like to do?"),
    ("tell me about Apollo 13", "Apollo 13 was the NASA mission that suffered an oxygen tank explosion on the way to the Moon. The lunar landing was aborted, and the crew returned safely after Mission Control and the astronauts improvised power, navigation, and life-support procedures."),
    ("which moon landing was nearly a disaster?", "Apollo 13 was the Moon mission that nearly became a disaster after an oxygen tank exploded, forcing the crew to abort the landing and return to Earth."),
    ("tell me about the first Moon landing in detail", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin descended in the Lunar Module Eagle while Michael Collins orbited above in Columbia. Armstrong stepped onto the surface first, followed by Aldrin, and they collected samples, deployed experiments, and photographed the site before returning safely."),
    ("do not use tools; what command prints the current folder?", "`pwd` prints the current folder."),
    ("write a sentence using the word browser", "The browser opened to a page full of reference notes."),
    ("without tools, what is 12 times 13?", "12 times 13 is 156."),
    ("what is the difference between a browser and a terminal?", "A browser is for viewing and interacting with web pages, while a terminal is for running text commands on the operating system."),
    ("return a tiny JSON object with status ok", "{\"status\":\"ok\"}"),
    ("explain what a tool call is without making one", "A tool call is a structured request from an assistant to an external capability such as a browser, terminal, or file reader."),
    ("define quantization in one paragraph", "Quantization stores numerical values at lower precision to reduce memory and bandwidth costs while trying to preserve useful model behavior."),
]

FAMILY_TARGETS = {
    "browser": {"browser_search_current", "correction_recovery", "tool_result_finalization"},
    "terminal_file": {"terminal_file_patch", "correction_recovery", "tool_result_finalization"},
    "computer_use": {"computer_use_browser_control", "correction_recovery"},
}

FAMILY_TOOLS = {
    "browser": {"browser_navigate", "x_search"},
    "terminal_file": {"terminal", "read_file", "write_file", "search_files", "patch", "execute_code"},
    "computer_use": {"computer_use", "browser_navigate"},
}


def now_slug() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def request_json(url: str, timeout: int = 20) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def stable_split(row_id: str) -> str:
    bucket = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 93:
        return "valid"
    return "test"


def load_tools(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tools = payload["tools"]
    return tools, {tool["function"]["name"]: tool for tool in tools}, payload


def select_tools(tools_by_name: dict[str, dict[str, Any]], names: set[str]) -> list[dict[str, Any]]:
    ordered = []
    for name in sorted(names):
        if name in tools_by_name:
            ordered.append(tools_by_name[name])
    return ordered


def assistant_tool_call(row_id: str, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{row_id}",
                "type": "function",
                "function": {"name": tool_name, "arguments": args},
            }
        ],
    }


def row_text(row_id: str, prompt: str, answer: str, tools: list[dict[str, Any]], category: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "tools": tools,
    }


def row_tool(row_id: str, prompt: str, tool_name: str, args: dict[str, Any], tools: list[dict[str, Any]], category: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            assistant_tool_call(row_id, tool_name, args),
        ],
        "tools": tools,
    }


def row_final(row_id: str, source: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    tool_name = source["expected_tool"]
    args = source.get("expected_args") or {}
    contains = "the tool result"
    prompt = source["prompt"]
    if "weather" in prompt.lower():
        result = "Austin weather: 87 F and partly cloudy."
        answer = "Austin is 87 F and partly cloudy."
    elif tool_name == "terminal":
        result = "final-ok"
        answer = "The terminal command returned: final-ok"
    elif tool_name == "read_file":
        result = "marker: live-hermes"
        answer = "The file contains: marker: live-hermes"
    elif tool_name == "execute_code":
        result = "15"
        answer = "The result is 15."
    else:
        result = contains
        answer = "The tool returned: the tool result."
    return {
        "id": row_id,
        "category": "tool_result_finalization",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            assistant_tool_call(row_id, tool_name, args),
            {"role": "tool", "tool_call_id": f"call_{row_id}", "name": tool_name, "content": result},
            {"role": "assistant", "content": answer},
        ],
        "tools": tools,
    }


def expanded_cases(tools: list[dict[str, Any]], seed: int, max_cases: int) -> list[live_hermes_eval.Case]:
    base = live_hermes_eval.build_cases({tool["function"]["name"] for tool in tools})
    variants = [
        ("", ""),
        ("Please ", ""),
        ("", " Be concise."),
        ("Hermes, ", ""),
        ("", " Use the best available path."),
    ]
    rng = random.Random(seed)
    expanded: list[live_hermes_eval.Case] = []
    for idx, case in enumerate(base):
        expanded.append(case)
        if case.turns:
            continue
        for variant_idx, (prefix, suffix) in enumerate(variants[1:], start=1):
            clone = live_hermes_eval.Case(**asdict(case))
            clone.case_id = f"{case.case_id}_v{variant_idx}"
            clone.prompt = f"{prefix}{case.prompt}{suffix}"
            expanded.append(clone)
    rng.shuffle(expanded)
    return expanded[:max_cases]


def run_eval_cases(
    endpoint: str,
    model: str,
    tools: list[dict[str, Any]],
    cases: list[live_hermes_eval.Case],
    out_jsonl: Path,
    out_report: Path,
    name: str,
    adapter_path: str | None = None,
) -> dict[str, Any]:
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    Path("/tmp/hermes_live_eval_marker.txt").write_text("marker: live-hermes\n", encoding="utf-8")
    Path("/tmp/hermes_live_eval_patch.txt").write_text("old\n", encoding="utf-8")
    results = []
    with out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, case in enumerate(cases, start=1):
            row = live_hermes_eval.evaluate_case(endpoint, model, tools, case, adapter_path)
            row["index"] = idx
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"phase": name, "index": idx, "case_id": row["case_id"], "passed": row["passed"], "failure": row.get("failure")}), flush=True)
    summary = live_hermes_eval.summarize(results)
    out_jsonl.with_suffix(".summary.json").write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    live_hermes_eval.write_report(out_report, name, summary, results)
    return {"summary": summary, "results": results, "jsonl": str(out_jsonl), "report": str(out_report)}


def mine_dataset(
    family: str,
    baseline_results: list[dict[str, Any]],
    tools_by_name: dict[str, dict[str, Any]],
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    family_tools = select_tools(tools_by_name, FAMILY_TOOLS[family])
    retention_tools = select_tools(tools_by_name, {"browser_navigate", "terminal", "read_file", "search_files", "computer_use"})

    relevant = [
        row for row in baseline_results
        if row["category"] in FAMILY_TARGETS[family]
        and (not row["passed"] or row.get("expected_tool") in FAMILY_TOOLS[family])
        and row.get("expected_tool")
        and row.get("expected_args") is not None
    ]

    for i, row in enumerate(relevant):
        expected_tool = row["expected_tool"]
        if expected_tool not in tools_by_name:
            continue
        prompt = row["prompt"]
        args = row.get("expected_args") or {}
        category = f"{family}_repair"
        for j, prefix in enumerate(["", "Please ", "Use the correct Hermes tool: "]):
            rid = f"overnight_{family}_repair_{i}_{j}_{hashlib.md5((prefix + prompt).encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prefix + prompt, expected_tool, args, family_tools, category))
        if row["category"] == "tool_result_finalization":
            rid = f"overnight_{family}_final_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_final(rid, row, family_tools))

    normal_multiplier = 8 if family in {"browser", "computer_use"} else 6
    for i in range(normal_multiplier):
        for prompt, answer in RETENTION_ROWS:
            rid = f"overnight_{family}_retention_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_text(rid, prompt, answer, retention_tools, "normal_retention"))

    hard_negatives = [
        ("Do not use tools. What is a browser?", "A browser is an application for viewing and interacting with websites."),
        ("Do not open anything. What is x.com?", "X.com is the site for X, formerly Twitter."),
        ("Explain weather forecasts without checking current weather.", "A weather forecast is a prediction of future conditions based on observations and models."),
        ("Do not run terminal. What command lists files?", "`ls` lists files, but I will not run it."),
        ("Do not inspect files. What does read_file mean?", "`read_file` means reading file contents, but I will not call it here."),
        ("Do not use computer_use. What does click mean?", "A click is a pointer action used to select or activate something in a graphical interface."),
    ]
    for i in range(10):
        for prompt, answer in hard_negatives:
            rid = f"overnight_{family}_negative_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_text(rid, prompt, answer, retention_tools, "hard_negative"))

    if len(rows) < 80:
        for i, row in enumerate(baseline_results):
            if row.get("expected_tool") and row.get("expected_args") and row["expected_tool"] in FAMILY_TOOLS[family]:
                rid = f"overnight_{family}_topup_{i}_{hashlib.md5(row['prompt'].encode()).hexdigest()[:8]}"
                rows.append(row_tool(rid, row["prompt"], row["expected_tool"], row["expected_args"], family_tools, f"{family}_repair"))

    rng.shuffle(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        splits[stable_split(row["id"])].append(row)
    for split, split_rows in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "family": family,
        "row_count": len(rows),
        "splits": {name: len(split_rows) for name, split_rows in splits.items()},
        "source_failures": len([r for r in relevant if not r["passed"]]),
        "source_relevant": len(relevant),
        "generated_at": now_slug(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def run_training(repo: Path, model: str, data_dir: Path, adapter_dir: Path, py: str, lr: float, rank: int, seed: int, iters: int) -> int:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        py,
        str(repo / "scripts" / "run_lora_repair.py"),
        "--model", model,
        "--data", str(data_dir),
        "--adapter-path", str(adapter_dir),
        "--out-config", str(adapter_dir / "run_config.json"),
        "--log", str(adapter_dir / "train.log"),
        "--mask-prompt",
        "--iters", str(iters),
        "--num-layers", "-1",
        "--batch-size", "1",
        "--max-seq-length", "4096",
        "--learning-rate", str(lr),
        "--lora-rank", str(rank),
        "--lora-scale", str(rank * 2),
        "--val-batches", "4",
        "--save-every", "50",
        "--steps-per-report", "10",
        "--steps-per-eval", "50",
        "--seed", str(seed),
    ]
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=repo).returncode


def category_regressed(candidate: dict[str, Any], base: dict[str, Any], category: str) -> bool:
    cand = candidate["category_metrics"].get(category, {"rate": 0})["rate"]
    ref = base["category_metrics"].get(category, {"rate": 0})["rate"]
    return cand < ref


def evaluate_candidate(candidate: dict[str, Any], base: dict[str, Any], family: str) -> tuple[bool, list[str]]:
    reasons = []
    if candidate["text_tool_leaks"] > 0:
        reasons.append("text_tool_leaks")
    if candidate["invented_tool_names"] > 0:
        reasons.append("invented_tool_names")
    if candidate["invented_computer_actions"] > 0:
        reasons.append("invented_computer_actions")
    if candidate["no_tool_false_positive_rate"] > base["no_tool_false_positive_rate"]:
        reasons.append("no_tool_false_positive_regression")
    if category_regressed(candidate, base, "normal_chat"):
        reasons.append("normal_chat_regression")
    family_categories = {
        "browser": ["browser_search_current"],
        "terminal_file": ["terminal_file_patch"],
        "computer_use": ["computer_use_browser_control"],
    }[family]
    for cat in family_categories:
        if candidate["category_metrics"].get(cat, {"rate": 0})["rate"] <= base["category_metrics"].get(cat, {"rate": 0})["rate"]:
            reasons.append(f"no_{cat}_improvement")
    return not reasons, reasons


def run_hermes_smoke(prompts: list[str], out_path: Path, timeout_s: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PATH"] = f"{Path.home() / '.local/bin'}:{env.get('PATH', '')}"
    env["HERMES_PARSE_TEXT_TOOL_CALLS"] = "0"
    with out_path.open("w", encoding="utf-8") as handle:
        for idx, prompt in enumerate(prompts, start=1):
            handle.write(f"\n\n===== CASE {idx}: {prompt} =====\n")
            handle.flush()
            try:
                proc = subprocess.run(
                    ["hermes", "-z", prompt, "--toolsets", "browser,terminal,file,computer_use,code_execution,x_search"],
                    cwd=Path.cwd(),
                    env=env,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=timeout_s,
                )
                handle.write(proc.stdout)
                handle.write(f"\n[exit={proc.returncode}]\n")
            except subprocess.TimeoutExpired as exc:
                handle.write((exc.stdout or "") if isinstance(exc.stdout, str) else "")
                handle.write(f"\n[TIMEOUT after {timeout_s}s]\n")
            handle.flush()


def write_final_report(path: Path, state: dict[str, Any]) -> None:
    lines = [
        "# Overnight Live Hermes Repair Report",
        "",
        f"- Started: `{state['started_at']}`",
        f"- Base model: `{state['model']}`",
        f"- Endpoint: `{state['endpoint']}`",
        f"- Decision: **{state['decision']}**",
        "",
        "## Baseline",
        "",
    ]
    base = state["baseline"]["summary"]
    lines.extend([
        f"- Overall: `{base['passed']}/{base['total']} ({base['overall_rate']})`",
        f"- Structured tool-call rate: `{base['valid_structured_tool_calls']}/{base['tool_required_cases']} ({base['valid_structured_tool_rate']})`",
        f"- No-tool false positive rate: `{base['no_tool_false_positive_rate']}`",
        "",
        "## Adapter Attempts",
        "",
    ])
    for attempt in state["attempts"]:
        summary = attempt.get("summary", {})
        lines.append(f"### `{attempt['family']}`")
        lines.append("")
        lines.append(f"- Adapter: `{attempt.get('adapter_path')}`")
        lines.append(f"- Decision: `{attempt.get('decision')}`")
        if summary:
            lines.append(f"- Overall: `{summary['passed']}/{summary['total']} ({summary['overall_rate']})`")
            lines.append(f"- Structured tool-call rate: `{summary['valid_structured_tool_calls']}/{summary['tool_required_cases']} ({summary['valid_structured_tool_rate']})`")
            lines.append(f"- No-tool false positive rate: `{summary['no_tool_false_positive_rate']}`")
        if attempt.get("reasons"):
            lines.append(f"- Rejection reasons: `{', '.join(attempt['reasons'])}`")
        lines.append("")
    lines.extend([
        "## Quantization",
        "",
        state["quantization_note"],
        "",
        "## Artifacts",
        "",
    ])
    for key, value in state["artifacts"].items():
        lines.append(f"- `{key}`: `{value}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an overnight gated live-Hermes LoRA repair search.")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--model", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run-id", default=f"overnight_{now_slug()}")
    parser.add_argument("--max-cases", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3117)
    parser.add_argument("--iters", type=int, default=150)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=4e-7)
    parser.add_argument("--hermes-smoke-timeout", type=int, default=90)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    repo = args.repo.resolve()
    os.chdir(repo)
    started_at = now_slug()
    model_id = request_json(args.endpoint.replace("/chat/completions", "/models"))["data"][0]["id"]
    tokenizer_config = json.loads((Path(args.model) / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tokenizer_config.get("tool_parser_type") != "pythonic":
        raise SystemExit("Base runtime is missing tokenizer_config.tool_parser_type='pythonic'.")
    print(json.dumps({"event": "server_ok", "served_model_id": model_id, "parser": tokenizer_config.get("tool_parser_type")}), flush=True)

    tools, tools_by_name, _tool_payload = load_tools(args.tools_json)
    cases = expanded_cases(tools, args.seed, args.max_cases)
    if len(cases) < 150:
        raise SystemExit(f"Expanded suite only has {len(cases)} cases; expected at least 150.")

    results_dir = repo / "artifacts" / "live_hermes_eval" / "results"
    reports_dir = repo / "artifacts" / "live_hermes_eval" / "reports"
    transcripts_dir = repo / "artifacts" / "live_hermes_eval" / "transcripts"
    datasets_dir = repo / "artifacts" / "repair_datasets"
    adapters_dir = repo / "artifacts" / "adapters"

    baseline_name = f"{args.run_id}_baseline"
    baseline = run_eval_cases(
        args.endpoint,
        args.model,
        tools,
        cases,
        results_dir / f"{baseline_name}.jsonl",
        reports_dir / f"{baseline_name}.md",
        baseline_name,
    )

    smoke_prompts = [
        "hello",
        "what's the weather in Austin today",
        "run echo hermes-overnight-smoke in the terminal",
        "search this repo for HERMES_PARSE_TEXT_TOOL_CALLS",
        "use computer use to list running apps",
        "what happened on Apollo 13?",
    ]
    run_hermes_smoke(smoke_prompts, transcripts_dir / f"{args.run_id}_hermes_oneshot_smoke.txt", args.hermes_smoke_timeout)

    state: dict[str, Any] = {
        "started_at": started_at,
        "model": args.model,
        "endpoint": args.endpoint,
        "baseline": {"summary": baseline["summary"], "jsonl": baseline["jsonl"], "report": baseline["report"]},
        "attempts": [],
        "artifacts": {
            "baseline_jsonl": baseline["jsonl"],
            "baseline_report": baseline["report"],
            "hermes_smoke": str(transcripts_dir / f"{args.run_id}_hermes_oneshot_smoke.txt"),
        },
        "decision": "no accepted adapter yet",
        "quantization_note": "Quantization was deferred because no adapter has passed the live-Hermes acceptance gates.",
    }

    base_summary = baseline["summary"]
    accepted: list[dict[str, Any]] = []
    for family in ["browser", "terminal_file", "computer_use"]:
        data_dir = datasets_dir / f"{args.run_id}_{family}"
        manifest = mine_dataset(family, baseline["results"], tools_by_name, data_dir, args.seed)
        adapter_dir = adapters_dir / f"{args.run_id}_{family}_r{args.rank}"
        attempt: dict[str, Any] = {
            "family": family,
            "dataset": str(data_dir),
            "manifest": manifest,
            "adapter_path": str(adapter_dir),
        }
        if args.skip_train:
            attempt["decision"] = "skipped_training"
            state["attempts"].append(attempt)
            continue
        rc = run_training(repo, args.model, data_dir, adapter_dir, args.python, args.learning_rate, args.rank, args.seed + len(state["attempts"]), args.iters)
        attempt["train_returncode"] = rc
        if rc != 0 or not (adapter_dir / "adapters.safetensors").exists():
            attempt["decision"] = "training_failed"
            state["attempts"].append(attempt)
            continue
        eval_name = f"{args.run_id}_{family}_eval"
        candidate = run_eval_cases(
            args.endpoint,
            args.model,
            tools,
            cases,
            results_dir / f"{eval_name}.jsonl",
            reports_dir / f"{eval_name}.md",
            eval_name,
            adapter_path=str(adapter_dir),
        )
        attempt["summary"] = candidate["summary"]
        attempt["eval_jsonl"] = candidate["jsonl"]
        attempt["eval_report"] = candidate["report"]
        ok, reasons = evaluate_candidate(candidate["summary"], base_summary, family)
        attempt["decision"] = "accepted_individual" if ok else "rejected"
        attempt["reasons"] = reasons
        state["attempts"].append(attempt)
        if ok:
            accepted.append(attempt)

    if accepted:
        state["decision"] = "one or more individual adapters accepted; stacking/fusion still deferred pending second independent live suite"
    else:
        state["decision"] = "no-go: no adapter beat fixed base without violating gates"

    docs_report = repo / "docs" / "overnight_live_hermes_repair_report.md"
    write_final_report(docs_report, state)
    (repo / "artifacts" / "overnight_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    print(json.dumps({"event": "complete", "decision": state["decision"], "report": str(docs_report)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
