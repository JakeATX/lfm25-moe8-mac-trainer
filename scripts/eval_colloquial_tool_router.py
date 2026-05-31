#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Open a URL or web search in the browser.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command in a terminal.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "workdir": {"type": "string"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a local text file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text to a local file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search local files for a text pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate an exact arithmetic expression.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


SYSTEM_GENERIC = "You are a helpful assistant with tools."
SYSTEM_STRICT = (
    "Use tools when needed. Tool calls must be returned through the API tool-calling "
    "mechanism, not written as plain text. If no tool is needed, answer normally."
)


@dataclass
class Case:
    name: str
    category: str
    messages: list[dict[str, Any]]
    expect_tool: str | None = None
    required_args: tuple[str, ...] = ()
    arg_contains: dict[str, str] | None = None
    min_content_contains: str | None = None
    max_tokens: int = 192


def request_json(endpoint: str, payload: dict[str, Any], timeout: int = 160) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')}") from e


def parse_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}


def looks_like_text_tool_call(text: str) -> bool:
    return bool(re.search(r"<\|tool_call_start\|>|<tool_call>|\[[a-zA-Z_]\w*\([^)]*=.*\)\]", text or ""))


def cases() -> list[Case]:
    def sys_user(system: str, user: str) -> list[dict[str, str]]:
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    return [
        Case("generic_weather_austin", "browser", sys_user(SYSTEM_GENERIC, "whats the weather today in Austin TX"), "browser_navigate", ("url",), {"url": "weather"}),
        Case("generic_latest_release", "browser", sys_user(SYSTEM_GENERIC, "check online for the latest mlx-lm tool parser docs"), "browser_navigate", ("url",), {"url": "mlx"}),
        Case("generic_search_phrase", "browser", sys_user(SYSTEM_GENERIC, "look up current Liquid AI LFM2.5 context length"), "browser_navigate", ("url",), {"url": "Liquid"}),
        Case("strict_browser_open", "browser", sys_user(SYSTEM_STRICT, "open a search for NOAA Austin forecast"), "browser_navigate", ("url",), {"url": "Austin"}),
        Case("generic_terminal_echo", "terminal", sys_user(SYSTEM_GENERIC, "run echo hermes-tool-test"), "terminal", ("command",), {"command": "echo hermes-tool-test"}),
        Case("generic_terminal_pwd", "terminal", sys_user(SYSTEM_GENERIC, "pwd in terminal"), "terminal", ("command",), {"command": "pwd"}),
        Case("generic_terminal_ls", "terminal", sys_user(SYSTEM_GENERIC, "list files here from shell"), "terminal", ("command",), {"command": "ls"}),
        Case("strict_terminal_python", "terminal", sys_user(SYSTEM_STRICT, "use terminal to show python version"), "terminal", ("command",), {"command": "python"}),
        Case("generic_read_file", "file", sys_user(SYSTEM_GENERIC, "read /tmp/hermes_eval_marker.txt"), "read_file", ("path",), {"path": "/tmp/hermes_eval_marker.txt"}),
        Case("generic_search_repo", "file", sys_user(SYSTEM_GENERIC, "search this repo for browser_navigate"), "search_files", ("pattern",), {"pattern": "browser_navigate"}),
        Case("generic_write_file", "file", sys_user(SYSTEM_GENERIC, "write /tmp/hermes_router_note.txt with hello"), "write_file", ("path", "content"), {"path": "/tmp/hermes_router_note.txt", "content": "hello"}),
        Case("generic_calculator", "calculator", sys_user(SYSTEM_GENERIC, "what is 18.5 times 42 exactly"), "calculator", ("expression",), {"expression": "18.5"}),
        Case("generic_calculator_2", "calculator", sys_user(SYSTEM_GENERIC, "compute (23 + 19) * 2"), "calculator", ("expression",), {"expression": "(23 + 19)"}),
        Case("no_tool_greeting", "no_tool", sys_user(SYSTEM_GENERIC, "hello!"), min_content_contains="hello"),
        Case("no_tool_explain_router", "no_tool", sys_user(SYSTEM_GENERIC, "explain what a router does in a MoE model in one sentence"), min_content_contains="expert"),
        Case("no_tool_do_not_use", "no_tool", sys_user(SYSTEM_GENERIC, "do not use tools, just say done"), min_content_contains="done"),
        Case("no_tool_word_browser", "no_tool", sys_user(SYSTEM_GENERIC, "write a sentence using the word browser"), min_content_contains="browser"),
        Case(
            "finalize_browser_weather",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "whats the weather today in Austin TX"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "browser_navigate",
                                "arguments": json.dumps({"url": "https://www.google.com/search?q=Austin+TX+weather"}),
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_weather", "name": "browser_navigate", "content": "Austin weather is 82 F and sunny."},
            ],
            min_content_contains="82",
        ),
        Case(
            "finalize_terminal_echo",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "run echo hermes-tool-test"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_echo",
                            "type": "function",
                            "function": {"name": "terminal", "arguments": json.dumps({"command": "echo hermes-tool-test"})},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_echo", "name": "terminal", "content": "hermes-tool-test"},
            ],
            min_content_contains="hermes-tool-test",
        ),
        Case(
            "finalize_calculator_trust",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "what is 18.5 times 42 exactly"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_calc",
                            "type": "function",
                            "function": {"name": "calculator", "arguments": json.dumps({"expression": "18.5 * 42"})},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_calc", "name": "calculator", "content": "777"},
            ],
            min_content_contains="777",
        ),
    ]


def evaluate_case(endpoint: str, model: str, case: Case) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": case.messages,
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": case.max_tokens,
        "temperature": 0,
    }
    start = time.time()
    resp = request_json(endpoint, payload)
    elapsed = time.time() - start
    choice = resp["choices"][0]
    msg = choice["message"]
    content = msg.get("content") or ""
    tool_calls = msg.get("tool_calls") or []
    text_leak = looks_like_text_tool_call(content)
    failure: str | None = None
    args: dict[str, Any] = {}
    tool_name: str | None = None

    if case.expect_tool:
        if not tool_calls:
            failure = "missing_tool_call"
        elif text_leak:
            failure = "text_tool_leak"
        else:
            first = tool_calls[0]
            fn = first.get("function", {})
            tool_name = fn.get("name")
            args = parse_args(fn.get("arguments"))
            if tool_name != case.expect_tool:
                failure = f"wrong_tool:{tool_name}"
            elif any(required not in args for required in case.required_args):
                missing = sorted(set(case.required_args) - set(args))
                failure = f"missing_args:{missing}"
            elif case.arg_contains:
                for key, needle in case.arg_contains.items():
                    if needle.lower() not in str(args.get(key, "")).lower():
                        failure = f"arg_mismatch:{key}"
                        break
            if failure is None and choice.get("finish_reason") != "tool_calls":
                failure = f"bad_finish_reason:{choice.get('finish_reason')}"
    else:
        if tool_calls:
            tool_name = tool_calls[0].get("function", {}).get("name")
            failure = f"false_positive_tool_call:{tool_name}"
        elif text_leak:
            failure = "text_tool_false_positive"
        elif case.min_content_contains and case.min_content_contains.lower() not in content.lower():
            failure = "content_mismatch"

    return {
        "name": case.name,
        "category": case.category,
        "passed": failure is None,
        "failure": failure,
        "elapsed_s": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "tool_calls": tool_calls,
        "tool_name": tool_name,
        "parsed_args": args,
        "text_tool_leak": text_leak,
        "usage": resp.get("usage"),
    }


def summarize(results: list[dict[str, Any]], eval_cases: list[Case]) -> dict[str, Any]:
    by_name = {case.name: case for case in eval_cases}
    tool_results = [r for r in results if by_name[r["name"]].expect_tool]
    no_tool_results = [r for r in results if by_name[r["name"]].expect_tool is None and by_name[r["name"]].category == "no_tool"]
    finalization_results = [r for r in results if by_name[r["name"]].category == "finalization"]
    category_metrics = {}
    for category in sorted(set(case.category for case in eval_cases)):
        cat = [r for r in results if r["category"] == category]
        category_metrics[category] = {
            "passed": sum(r["passed"] for r in cat),
            "total": len(cat),
            "rate": round(sum(r["passed"] for r in cat) / len(cat), 4) if cat else 0,
        }
    false_positive_rate = (sum(not r["passed"] for r in no_tool_results) / len(no_tool_results)) if no_tool_results else 0
    return {
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "overall_rate": round(sum(r["passed"] for r in results) / len(results), 4),
        "tool_cases": len(tool_results),
        "structured_tool_cases_passed": sum(r["passed"] for r in tool_results),
        "structured_tool_rate": round(sum(r["passed"] for r in tool_results) / len(tool_results), 4) if tool_results else 0,
        "no_tool_cases": len(no_tool_results),
        "no_tool_cases_passed": sum(r["passed"] for r in no_tool_results),
        "false_positive_tool_rate": round(false_positive_rate, 4),
        "finalization_cases": len(finalization_results),
        "finalization_passed": sum(r["passed"] for r in finalization_results),
        "finalization_rate": round(sum(r["passed"] for r in finalization_results) / len(finalization_results), 4) if finalization_results else 0,
        "finish_reason_tool_calls": sum(1 for r in results if r["finish_reason"] == "tool_calls"),
        "content_text_tool_leaks": sum(1 for r in results if r["text_tool_leak"]),
        "category_metrics": category_metrics,
        "failures": {r["name"]: r["failure"] for r in results if not r["passed"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", default="release_work/model_upload")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    eval_cases = cases()
    results = [evaluate_case(args.endpoint, args.model, case) for case in eval_cases]
    summary = summarize(results, eval_cases)
    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "tools": [tool["function"]["name"] for tool in TOOLS],
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not args.allow_fail and summary["passed"] != summary["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
