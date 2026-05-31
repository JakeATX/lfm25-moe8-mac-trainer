#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are Hermes. Use the provided tools when they are needed to satisfy the user. "
    "If no tool is needed, answer normally. Do not invent tool names or arguments."
)


def stable_split(case_id: str) -> str:
    bucket = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 85:
        return "train"
    if bucket < 93:
        return "valid"
    return "test"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def assistant_tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                }
            ],
        }


def row_for_tool(case: dict[str, Any], tools: list[dict[str, Any]], suffix: str = "") -> dict[str, Any]:
    tool = case["expected_tool"]
    args = case.get("expected_args") or {}
    row_tools = select_training_tools(tools, primary=tool, prompt=case["prompt"])
    return {
        "id": f"live_repair_{case['case_id']}{suffix}",
        "source_case_id": case["case_id"],
        "failure": case.get("failure"),
        "category": case.get("category"),
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": case["prompt"]},
            assistant_tool_call(tool, args, f"call_{case['case_id']}{suffix}"),
        ],
        "tools": row_tools,
    }


def row_for_text(case_id: str, prompt: str, answer: str, tools: list[dict[str, Any]], category: str = "normal_retention") -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "tools": select_training_tools(tools, primary=None, prompt=prompt),
    }


def select_training_tools(tools: list[dict[str, Any]], primary: str | None, prompt: str) -> list[dict[str, Any]]:
    """Keep exact Hermes schemas, but not the whole 13K-token live catalog.

    MLX LoRA truncates at the sequence cap. With the full CLI catalog, the
    assistant target often falls outside 4K tokens. These row-local tool
    subsets preserve exact names/parameters/descriptions for the tools the row
    is teaching while adding a few realistic contrast tools.
    """
    by_name = {tool["function"]["name"]: tool for tool in tools}
    wanted: list[str] = []
    if primary:
        wanted.append(primary)
    lowered = prompt.lower()
    if any(word in lowered for word in ["weather", "online", "web", "browser", "open", "latest", "current"]):
        wanted.extend(["browser_navigate", "x_search"])
    if any(word in lowered for word in ["terminal", "shell", "run", "execute", "command", "pwd", "echo"]):
        wanted.extend(["terminal", "execute_code"])
    if any(word in lowered for word in ["file", "repo", "read", "write", "search", "patch"]):
        wanted.extend(["read_file", "search_files", "write_file", "patch"])
    if any(word in lowered for word in ["computer", "desktop", "chrome", "tab", "click", "type", "scroll"]):
        wanted.extend(["computer_use", "browser_navigate"])
    if primary is None:
        wanted.extend(["browser_navigate", "terminal", "read_file"])

    # Keep the subset compact and deterministic.
    compact: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in wanted:
        if name in by_name and name not in seen:
            compact.append(by_name[name])
            seen.add(name)
    return compact


def hard_negatives(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row_for_text("live_negative_no_tools_command", "Do not run anything; what command prints the current directory?", "`pwd` prints the current directory.", tools, "hard_negative"),
        row_for_text("live_negative_browser_word", "Write a sentence using the word browser.", "The browser opened to a page full of technical notes.", tools, "hard_negative"),
        row_for_text("live_negative_tool_call_concept", "Explain what a tool call is without making one.", "A tool call is a structured request from the model to an external capability, such as reading a file or opening a browser.", tools, "hard_negative"),
        row_for_text("live_negative_apollo13", "Which moon landing was nearly a disaster?", "Apollo 13 was the Moon mission that became a near-disaster after an oxygen tank exploded, forcing NASA and the crew to abort the landing and improvise a safe return to Earth.", tools, "normal_retention"),
        row_for_text("live_negative_hello", "hello", "Hello! How can I help you today?", tools, "normal_retention"),
        row_for_text("live_negative_no", "no", "Got it. Tell me what you want changed or corrected, and I will adjust.", tools, "normal_retention"),
        row_for_text("live_negative_moon_detail", "Tell me about the first Moon landing in detail.", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin descended in the Lunar Module Eagle while Michael Collins remained in lunar orbit. Armstrong became the first person to step onto the lunar surface, followed by Aldrin, and they collected samples, deployed experiments, and photographed the site before returning safely to orbit.", tools, "normal_retention"),
    ]


def build_rows(results: list[dict[str, Any]], tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in results:
        if case.get("expected_tool"):
            if not case.get("passed") or case.get("failure") in {"wrong_tool", "refusal_when_tool_available", "invalid_args", "invented_action", "text_tool_leak"}:
                rows.append(row_for_tool(case, tools))
                # Recovery contrast: the exact wording after user correction should
                # still map to the same valid tool call.
                rows.append(row_for_tool({**case, "prompt": f"No, use the right Hermes tool for this: {case['prompt']}"}, tools, "_correction"))
            elif case.get("passed"):
                rows.append(row_for_tool({**case, "failure": None}, tools, "_success_retention"))
        elif case.get("failure") in {"normal_chat_regression", "over_tooling_no_tool_prompt", "text_tool_leak"}:
            content = case.get("assistant_visible_text") or "I can answer that directly without using tools."
            rows.append(row_for_text(f"live_retention_{case['case_id']}", case["prompt"], content, tools, case.get("category") or "normal_retention"))
        elif case.get("passed") and case.get("category") == "normal_chat" and case.get("assistant_visible_text"):
            rows.append(
                row_for_text(
                    f"live_passed_retention_{case['case_id']}",
                    case["prompt"],
                    case["assistant_visible_text"].strip(),
                    tools,
                    "normal_retention",
                )
            )

    rows.extend(hard_negatives(tools))
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = json.dumps(row["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine live Hermes eval failures into MLX chat-tool repair data.")
    parser.add_argument("--results-jsonl", type=Path, required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    results = load_jsonl(args.results_jsonl)
    tool_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = tool_payload["tools"]
    rows = build_rows(results, tools)
    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        splits[stable_split(row["id"])].append(row)
    # Guarantee validation/test are non-empty for small mined batches.
    if not splits["valid"] and len(splits["train"]) > 2:
        splits["valid"].append(splits["train"].pop())
    if not splits["test"] and len(splits["train"]) > 2:
        splits["test"].append(splits["train"].pop())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        with (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "source_results": str(args.results_jsonl),
        "tools_source": tool_payload.get("source"),
        "tool_names": tool_payload.get("tool_names"),
        "row_count": len(rows),
        "splits": {key: len(value) for key, value in splits.items()},
        "failure_counts": {},
    }
    for row in results:
        if row.get("failure"):
            manifest["failure_counts"][row["failure"]] = manifest["failure_counts"].get(row["failure"], 0) + 1
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
