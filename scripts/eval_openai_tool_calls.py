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
            "description": "Open a URL in the browser",
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
            "name": "calculator",
            "description": "Evaluate a math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "terminal",
            "description": "Run a shell command",
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
            "description": "Read a local file",
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
            "name": "search_files",
            "description": "Search file contents",
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
]


@dataclass
class Case:
    name: str
    messages: list[dict[str, Any]]
    expect_tool: str | None = None
    required_args: tuple[str, ...] = ()
    arg_contains: dict[str, str] | None = None
    max_tokens: int = 192
    min_content_contains: str | None = None


SYSTEM = (
    "You are a Hermes-style assistant. Use tools when needed. If a tool is needed, "
    "return a tool call through the API tool-calling mechanism. Do not write tool "
    "calls as plain prose."
)


def request_json(endpoint: str, payload: dict[str, Any], timeout: int = 120) -> dict[str, Any]:
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
    return json.loads(raw)


def tool_calls_from_response(resp: dict[str, Any]) -> list[dict[str, Any]]:
    msg = resp["choices"][0]["message"]
    return msg.get("tool_calls") or []


def content_from_response(resp: dict[str, Any]) -> str:
    msg = resp["choices"][0]["message"]
    return msg.get("content") or ""


def looks_like_text_tool_call(text: str) -> bool:
    return bool(re.search(r"<\\|tool_call_start\\|>|<tool_call>|\\w+\\([^)]*=.*\\)", text or ""))


def build_cases() -> list[Case]:
    return [
        Case(
            name="browser_weather",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Use the browser to look up current weather in Austin, TX."},
            ],
            expect_tool="browser_navigate",
            required_args=("url",),
            arg_contains={"url": "weather"},
        ),
        Case(
            name="browser_docs",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Open a browser search for MLX LM tool parser documentation."},
            ],
            expect_tool="browser_navigate",
            required_args=("url",),
            arg_contains={"url": "MLX"},
        ),
        Case(
            name="calculator_simple",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Use the calculator for (23 + 19) * 2."},
            ],
            expect_tool="calculator",
            required_args=("expression",),
            arg_contains={"expression": "(23 + 19) * 2"},
        ),
        Case(
            name="calculator_decimal",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Calculate 18.5 * 42 with the calculator."},
            ],
            expect_tool="calculator",
            required_args=("expression",),
            arg_contains={"expression": "18.5"},
        ),
        Case(
            name="terminal_pwd",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Use terminal to run pwd."},
            ],
            expect_tool="terminal",
            required_args=("command",),
            arg_contains={"command": "pwd"},
        ),
        Case(
            name="terminal_echo",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Use terminal to run: echo hermes-tool-test"},
            ],
            expect_tool="terminal",
            required_args=("command",),
            arg_contains={"command": "echo hermes-tool-test"},
        ),
        Case(
            name="read_file",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Read the file /tmp/hermes_eval_marker.txt."},
            ],
            expect_tool="read_file",
            required_args=("path",),
            arg_contains={"path": "/tmp/hermes_eval_marker.txt"},
        ),
        Case(
            name="search_files",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Search files for the string hermes-tool-test."},
            ],
            expect_tool="search_files",
            required_args=("pattern",),
            arg_contains={"pattern": "hermes-tool-test"},
        ),
        Case(
            name="no_tool_greeting",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Say hello in one short sentence. Do not use tools."},
            ],
            min_content_contains="hello",
        ),
        Case(
            name="no_tool_explain",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "In one sentence, explain what a router does in a MoE model. Do not use tools."},
            ],
            min_content_contains="expert",
        ),
        Case(
            name="tool_result_final_browser",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Use the browser to look up current weather in Austin, TX."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_eval_weather",
                            "type": "function",
                            "function": {
                                "name": "browser_navigate",
                                "arguments": json.dumps({"url": "https://www.google.com/search?q=Austin+TX+weather"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_eval_weather",
                    "name": "browser_navigate",
                    "content": "Search result: Austin, TX weather is 82 F and sunny.",
                },
            ],
            min_content_contains="82",
        ),
        Case(
            name="tool_result_final_calc",
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Use calculator for 18.5 * 42."},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_eval_calc",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": json.dumps({"expression": "18.5 * 42"}),
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_eval_calc",
                    "name": "calculator",
                    "content": "777",
                },
            ],
            min_content_contains="777",
        ),
    ]


def evaluate_case(endpoint: str, model: str, case: Case) -> dict[str, Any]:
    payload: dict[str, Any] = {
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
    tool_calls = tool_calls_from_response(resp)
    content = content_from_response(resp)
    failure: str | None = None

    if case.expect_tool:
        if not tool_calls:
            failure = "missing_structured_tool_call"
        elif content and looks_like_text_tool_call(content):
            failure = "text_tool_call_leaked_in_content"
        else:
            first = tool_calls[0]
            fn = first.get("function", {})
            args = parse_args(fn.get("arguments"))
            if fn.get("name") != case.expect_tool:
                failure = f"wrong_tool:{fn.get('name')}"
            elif any(arg not in args for arg in case.required_args):
                failure = f"missing_args:{sorted(set(case.required_args) - set(args))}"
            elif case.arg_contains:
                for k, needle in case.arg_contains.items():
                    if needle.lower() not in str(args.get(k, "")).lower():
                        failure = f"arg_mismatch:{k}"
                        break
            if choice.get("finish_reason") != "tool_calls" and failure is None:
                failure = f"bad_finish_reason:{choice.get('finish_reason')}"
    else:
        if tool_calls:
            failure = "false_positive_tool_call"
        elif looks_like_text_tool_call(content):
            failure = "text_tool_call_false_positive"
        elif case.min_content_contains and case.min_content_contains.lower() not in content.lower():
            failure = "content_mismatch"

    return {
        "name": case.name,
        "passed": failure is None,
        "failure": failure,
        "elapsed_s": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "tool_calls": tool_calls,
        "usage": resp.get("usage"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", default="release_work/model_upload")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    results = [evaluate_case(args.endpoint, args.model, case) for case in build_cases()]
    tool_cases = [r for r in results if r["tool_calls"] or r["name"].startswith(("browser", "calculator", "terminal", "read_", "search_"))]
    summary = {
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "tool_cases": len([c for c in build_cases() if c.expect_tool]),
        "structured_tool_cases_passed": sum(r["passed"] for r in results if any(c.name == r["name"] and c.expect_tool for c in build_cases())),
        "no_tool_cases": len([c for c in build_cases() if not c.expect_tool]),
        "no_tool_cases_passed": sum(r["passed"] for r in results if any(c.name == r["name"] and not c.expect_tool for c in build_cases())),
        "finish_reason_tool_calls": sum(1 for r in results if r["finish_reason"] == "tool_calls"),
        "content_text_tool_leaks": sum(1 for r in results if looks_like_text_tool_call(r["content"])),
        "failures": {r["name"]: r["failure"] for r in results if not r["passed"]},
    }
    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passed"] == summary["total"] else 1)


if __name__ == "__main__":
    main()
