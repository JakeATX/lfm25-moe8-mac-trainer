#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from mlx_lm import generate, load


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "browser_navigate",
            "description": "Navigate the browser to a URL.",
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
            "description": "Run a shell command.",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}, "workdir": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate arithmetic.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string"}},
                "required": ["expression"],
            },
        },
    },
]


CASES = [
    {
        "id": "weather_browser",
        "kind": "tool_call",
        "tool": "browser_navigate",
        "required": ["url"],
        "user": "Search for Austin TX weather using the browser.",
    },
    {
        "id": "calculator",
        "kind": "tool_call",
        "tool": "calculator",
        "required": ["expression"],
        "user": "Use the calculator tool to compute 18.5 * 42.",
    },
    {
        "id": "terminal_echo",
        "kind": "tool_call",
        "tool": "terminal",
        "required": ["command"],
        "user": "Use the terminal tool to run: echo hermes-tool-test",
    },
    {
        "id": "no_tool_hello",
        "kind": "no_tool",
        "user": "Say hello in one short sentence.",
    },
    {
        "id": "no_tool_refusal",
        "kind": "no_tool",
        "user": "Tell me a one-sentence fun fact about Austin.",
    },
    {
        "id": "tool_response_final",
        "kind": "final_answer",
        "messages": [
            ("system", "You are a tool-using assistant. After tool results, answer directly."),
            ("user", "What is the weather in Austin?"),
            ("assistant", '<|tool_call_start|>[browser_navigate(url="https://www.google.com/search?q=Austin+TX+weather")]<|tool_call_end|>'),
            ("tool", '<tool_response>{"temperature_f":82,"condition":"sunny","location":"Austin, TX"}</tool_response>'),
        ],
    },
]


def system_prompt() -> str:
    return (
        "You are a Hermes-style tool-using assistant. Use tools only when needed. "
        "When a tool is needed, emit exactly one native LFM tool call in this format: "
        "<|tool_call_start|>[tool_name(arg=\"value\")]<|tool_call_end|>. "
        "Do not wrap native LFM calls in markdown. Available tools:\n"
        + json.dumps(TOOLS, ensure_ascii=False)
    )


def chatml(messages: list[tuple[str, str]]) -> str:
    text = "<|startoftext|>"
    for role, content in messages:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


CALL_RE = re.compile(r"<\|tool_call_start\|>\s*\[(.*?)\]\s*<\|tool_call_end\|>", re.S)
NAME_RE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.S)


def parse_lfm_call(text: str) -> list[dict]:
    calls = []
    for match in CALL_RE.finditer(text or ""):
        body = match.group(1).strip()
        name_match = NAME_RE.match(body)
        if not name_match:
            calls.append({"raw": body, "parse_ok": False, "name": None, "args": {}})
            continue
        name, args_raw = name_match.groups()
        args = {}
        for key, _quote, value in re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(['\"])(.*?)\2", args_raw, re.S):
            args[key] = value
        calls.append({"raw": body, "parse_ok": True, "name": name, "args": args})
    return calls


def classify_failure(case: dict, output: str, calls: list[dict]) -> str | None:
    if case["kind"] == "tool_call":
        if not calls:
            if "don't have access" in output.lower() or "cannot" in output.lower():
                return "tool_refusal"
            return "prose_instead_of_call"
        if not calls[0]["parse_ok"]:
            return "invalid_lfm_syntax"
        if calls[0]["name"] != case["tool"]:
            return "wrong_tool_name"
        missing = [k for k in case.get("required", []) if not calls[0]["args"].get(k)]
        if missing:
            return "missing_required_args"
        if len(calls) > 1:
            return "repeated_tool_calls"
    elif case["kind"] == "no_tool" and calls:
        return "false_positive_tool_call"
    elif case["kind"] == "final_answer":
        if calls:
            return "tool_call_after_tool_response"
        if not any(token in output.lower() for token in ("82", "sunny", "austin")):
            return "bad_tool_response_finalization"
    return None


def score(case: dict, output: str) -> dict:
    calls = parse_lfm_call(output)
    failure = classify_failure(case, output, calls)
    return {
        "passed": failure is None,
        "failure_type": failure,
        "lfm_call_count": len(calls),
        "parse_ok": bool(calls) and all(c["parse_ok"] for c in calls),
        "correct_tool": None if case["kind"] != "tool_call" or not calls else calls[0]["name"] == case["tool"],
        "required_args_ok": None
        if case["kind"] != "tool_call" or not calls
        else all(calls[0]["args"].get(k) for k in case.get("required", [])),
        "calls": calls,
    }


def case_messages(case: dict) -> list[tuple[str, str]]:
    if "messages" in case:
        return case["messages"]
    return [("system", system_prompt()), ("user", case["user"])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    model, tokenizer = load(args.model, adapter_path=args.adapter_path) if args.adapter_path else load(args.model)
    results = []
    for case in CASES:
        prompt = chatml(case_messages(case))
        start = time.time()
        output = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
        results.append(
            {
                "case_id": case["id"],
                "kind": case["kind"],
                "expected_tool": case.get("tool"),
                "output": output,
                "elapsed_s": time.time() - start,
                "score": score(case, output),
            }
        )
    summary = {
        "passed": sum(r["score"]["passed"] for r in results),
        "total": len(results),
        "tool_cases": sum(1 for r in results if r["kind"] == "tool_call"),
        "tool_cases_passed": sum(1 for r in results if r["kind"] == "tool_call" and r["score"]["passed"]),
        "parse_ok": sum(1 for r in results if r["score"]["parse_ok"]),
        "failures": {},
    }
    for r in results:
        ft = r["score"].get("failure_type")
        if ft:
            summary["failures"][ft] = summary["failures"].get(ft, 0) + 1
    report = {"model": args.model, "adapter_path": args.adapter_path, "summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
