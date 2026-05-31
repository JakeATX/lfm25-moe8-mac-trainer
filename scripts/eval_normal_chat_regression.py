#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TEXT_TOOL_RE = re.compile(r"<\|tool_call_start\|>|<tool_call>|\[[a-zA-Z_]\w*\([^)]*=.*\)\]")


@dataclass
class Case:
    name: str
    prompt: str
    min_completion_tokens: int
    required: tuple[str, ...] = ()
    max_tokens: int = 220
    tools: bool = False


SYSTEM_TOOL = "Use available tools when the user's request requires the browser, terminal, or local files. Otherwise answer normally."
SYSTEM_CHAT = "You are a helpful assistant. Answer directly and completely."


def cases() -> list[Case]:
    return [
        Case("greeting", "hi there", 3, ("hello",), 64),
        Case("moon_landing", "tell me about the moon landing", 55, ("Apollo", "Moon"), 220, True),
        Case("photosynthesis", "explain photosynthesis in plain English", 45, ("light", "oxygen"), 200, True),
        Case("airplanes", "how do airplanes fly?", 45, ("pressure", "wing"), 200),
        Case("black_holes", "what are black holes?", 45, ("gravity", "event"), 200),
        Case("compiler", "what is the difference between a compiler and an interpreter?", 45, ("compiler", "interpreter"), 200, True),
        Case("weather_vs_climate", "what is the difference between weather and climate?", 45, ("short", "long"), 200, True),
        Case("battery_capacity", "why do batteries lose capacity over time?", 45, ("capacity", "chem"), 200),
        Case("do_not_run", "do not run anything; what command prints the current directory?", 5, ("pwd",), 80, True),
        Case("word_browser", "write one sentence using the word browser", 6, ("browser",), 80, True),
    ]


def request_json(endpoint: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def load_tools(path: Path | None) -> list[dict[str, Any]]:
    if not path:
        return []
    return json.loads(path.read_text(encoding="utf-8"))["tools"]


def evaluate_case(endpoint: str, model: str, tools: list[dict[str, Any]], case: Case, adapter_path: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_TOOL if case.tools else SYSTEM_CHAT},
            {"role": "user", "content": case.prompt},
        ],
        "temperature": 0,
        "max_tokens": case.max_tokens,
    }
    if case.tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    if adapter_path:
        payload["adapters"] = adapter_path
    start = time.time()
    resp = request_json(endpoint, payload)
    elapsed = time.time() - start
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    completion_tokens = int(resp.get("usage", {}).get("completion_tokens") or 0)
    failure = None
    if tool_calls:
        failure = f"unexpected_tool_call:{tool_calls[0].get('function', {}).get('name')}"
    elif TEXT_TOOL_RE.search(content):
        failure = "text_tool_leak"
    elif completion_tokens < case.min_completion_tokens:
        failure = f"too_short:{completion_tokens}<{case.min_completion_tokens}"
    else:
        missing = [needle for needle in case.required if needle.lower() not in content.lower()]
        if missing:
            failure = f"missing_required:{missing}"
    return {
        "name": case.name,
        "passed": failure is None,
        "failure": failure,
        "prompt": case.prompt,
        "elapsed_s": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": completion_tokens,
        "content": content,
        "tool_calls": tool_calls,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "rate": round(sum(r["passed"] for r in results) / len(results), 4),
        "too_short": sum(1 for r in results if str(r["failure"]).startswith("too_short")),
        "unexpected_tool_calls": sum(1 for r in results if str(r["failure"]).startswith("unexpected_tool_call")),
        "text_tool_leaks": sum(1 for r in results if r["failure"] == "text_tool_leak"),
        "failures": {r["name"]: r["failure"] for r in results if not r["passed"]},
        "acceptance": {
            "all_cases_pass": all(r["passed"] for r in results),
            "zero_unexpected_tools": not any(str(r["failure"]).startswith("unexpected_tool_call") for r in results),
            "zero_text_tool_leaks": not any(r["failure"] == "text_tool_leak" for r in results),
            "zero_too_short": not any(str(r["failure"]).startswith("too_short") for r in results),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()
    tools = load_tools(args.tools_json)
    results = [evaluate_case(args.endpoint, args.model, tools, case, args.adapter_path) for case in cases()]
    summary = summarize(results)
    report = {"endpoint": args.endpoint, "model": args.model, "adapter_path": args.adapter_path, "summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.allow_fail and not all(summary["acceptance"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
