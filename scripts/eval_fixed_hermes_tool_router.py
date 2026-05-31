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


TEXT_TOOL_RE = re.compile(r"<\|tool_call_start\|>|<tool_call>|\[[a-zA-Z_]\w*\([^)]*=.*\)\]")


SYSTEM_GENERIC = "You are a helpful assistant with tools."
SYSTEM_STRICT = "Use available tools when needed. If no tool is needed, answer normally."


@dataclass
class Case:
    name: str
    suite: str
    category: str
    messages: list[dict[str, Any]]
    expect_tool: str | None = None
    required_args: tuple[str, ...] = ()
    arg_contains: dict[str, str] | None = None
    content_contains: str | None = None
    max_tokens: int = 192


def request_json(endpoint: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')}") from exc


def parse_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}


def sys_user(system: str, user: str) -> list[dict[str, str]]:
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tool_message(tool_name: str, call_id: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": tool_name, "arguments": json.dumps(args)},
            }
        ],
    }


def cases() -> list[Case]:
    return [
        Case("a_weather_austin", "suite_a", "browser", sys_user(SYSTEM_GENERIC, "what's the weather today in Austin TX"), "browser_navigate", ("url",), {"url": "weather"}),
        Case("a_latest_mlx", "suite_a", "browser", sys_user(SYSTEM_GENERIC, "check online for the latest mlx-lm release notes"), "browser_navigate", ("url",), {"url": "mlx"}),
        Case("a_noaa_forecast", "suite_a", "browser", sys_user(SYSTEM_STRICT, "open a search for NOAA Austin forecast"), "browser_navigate", ("url",), {"url": "Austin"}),
        Case("a_echo", "suite_a", "terminal", sys_user(SYSTEM_GENERIC, "run echo hermes-tool-test"), "terminal", ("command",), {"command": "echo hermes-tool-test"}),
        Case("a_pwd", "suite_a", "terminal", sys_user(SYSTEM_GENERIC, "pwd in terminal"), "terminal", ("command",), {"command": "pwd"}),
        Case("a_ls", "suite_a", "terminal", sys_user(SYSTEM_GENERIC, "list files here from shell"), "terminal", ("command",), {"command": "ls"}),
        Case("a_python_version", "suite_a", "terminal", sys_user(SYSTEM_STRICT, "use terminal to show python version"), "terminal", ("command",), {"command": "python"}),
        Case("a_read_file", "suite_a", "file", sys_user(SYSTEM_GENERIC, "read /tmp/hermes_eval_marker.txt"), "read_file", ("path",), {"path": "/tmp/hermes_eval_marker.txt"}),
        Case("a_search_repo", "suite_a", "file", sys_user(SYSTEM_GENERIC, "search this repo for browser_navigate"), "search_files", ("pattern",), {"pattern": "browser_navigate"}),
        Case("a_write_file", "suite_a", "file", sys_user(SYSTEM_GENERIC, "write /tmp/hermes_router_note.txt with hello"), "write_file", ("path", "content"), {"path": "/tmp/hermes_router_note.txt", "content": "hello"}),
        Case("a_no_tool_greeting", "suite_a", "no_tool", sys_user(SYSTEM_GENERIC, "hello!"), content_contains="hello"),
        Case("a_no_tool_explain", "suite_a", "no_tool", sys_user(SYSTEM_GENERIC, "explain what a router does in a MoE model in one sentence"), content_contains="expert"),
        Case("a_no_tool_do_not", "suite_a", "no_tool", sys_user(SYSTEM_GENERIC, "do not use tools, just say done"), content_contains="done"),
        Case("a_no_tool_word_browser", "suite_a", "no_tool", sys_user(SYSTEM_GENERIC, "write a sentence using the word browser"), content_contains="browser"),
        Case(
            "a_final_browser",
            "suite_a",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "what's the weather today in Austin TX"},
                tool_message("browser_navigate", "call_weather", {"url": "https://www.google.com/search?q=Austin+TX+weather+today"}),
                {"role": "tool", "tool_call_id": "call_weather", "name": "browser_navigate", "content": "Austin is 82 F and sunny."},
            ],
            content_contains="82",
        ),
        Case(
            "a_final_terminal",
            "suite_a",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "run echo hermes-tool-test"},
                tool_message("terminal", "call_echo", {"command": "echo hermes-tool-test"}),
                {"role": "tool", "tool_call_id": "call_echo", "name": "terminal", "content": "hermes-tool-test"},
            ],
            content_contains="hermes-tool-test",
        ),
        Case(
            "a_final_file",
            "suite_a",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "read /tmp/hermes_eval_marker.txt"},
                tool_message("read_file", "call_read", {"path": "/tmp/hermes_eval_marker.txt"}),
                {"role": "tool", "tool_call_id": "call_read", "name": "read_file", "content": "marker: fixed-schema"},
            ],
            content_contains="fixed-schema",
        ),
        Case("b_weather_plain", "suite_b", "browser", sys_user(SYSTEM_GENERIC, "can you check the current weather for Austin"), "browser_navigate", ("url",), {"url": "weather"}),
        Case("b_web_search", "suite_b", "browser", sys_user(SYSTEM_GENERIC, "look up current Liquid AI LFM2.5 context length"), "browser_navigate", ("url",), {"url": "Liquid"}),
        Case("b_terminal_uname", "suite_b", "terminal", sys_user(SYSTEM_GENERIC, "run uname -m in the terminal"), "terminal", ("command",), {"command": "uname"}),
        Case("b_terminal_whoami", "suite_b", "terminal", sys_user(SYSTEM_GENERIC, "print the current username in shell"), "terminal", ("command",), {"command": "whoami"}),
        Case("b_file_search_local", "suite_b", "file", sys_user(SYSTEM_GENERIC, "find where HERMES_PARSE_TEXT_TOOL_CALLS appears locally"), "search_files", ("pattern",), {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        Case("b_file_write", "suite_b", "file", sys_user(SYSTEM_GENERIC, "create /tmp/hermes_fixed_schema_probe.txt containing fixed schema"), "write_file", ("path", "content"), {"path": "/tmp/hermes_fixed_schema_probe.txt", "content": "fixed schema"}),
        Case("b_no_tool_command", "suite_b", "no_tool", sys_user(SYSTEM_GENERIC, "what command prints the current directory? do not run it"), content_contains="pwd"),
        Case("b_no_tool_browser", "suite_b", "no_tool", sys_user(SYSTEM_GENERIC, "what is a browser in one sentence?"), content_contains="web"),
    ]


def holdout_cases() -> list[Case]:
    return [
        Case("h_weather_san_antonio", "holdout", "browser", sys_user(SYSTEM_GENERIC, "can you check online for San Antonio weather right now"), "browser_navigate", ("url",), {"url": "weather"}),
        Case("h_current_docs", "holdout", "browser", sys_user(SYSTEM_GENERIC, "find the current docs for mlx_lm server tool calling"), "browser_navigate", ("url",), {"url": "mlx"}),
        Case("h_online_latest", "holdout", "browser", sys_user(SYSTEM_GENERIC, "look on the web for the latest LiquidAI LFM2.5 GGUF info"), "browser_navigate", ("url",), {"url": "Liquid"}),
        Case("h_noaa", "holdout", "browser", sys_user(SYSTEM_STRICT, "search online for NOAA forecast Austin Texas"), "browser_navigate", ("url",), {"url": "NOAA"}),
        Case("h_terminal_date", "holdout", "terminal", sys_user(SYSTEM_GENERIC, "run date in the terminal"), "terminal", ("command",), {"command": "date"}),
        Case("h_terminal_pwd_plain", "holdout", "terminal", sys_user(SYSTEM_GENERIC, "show current directory with a shell command"), "terminal", ("command",), {"command": "pwd"}),
        Case("h_terminal_echo", "holdout", "terminal", sys_user(SYSTEM_GENERIC, "execute this locally: echo fixed-hermes"), "terminal", ("command",), {"command": "echo fixed-hermes"}),
        Case("h_terminal_files", "holdout", "terminal", sys_user(SYSTEM_GENERIC, "use shell to list the current folder"), "terminal", ("command",), {"command": "ls"}),
        Case("h_read_absolute", "holdout", "file", sys_user(SYSTEM_GENERIC, "open and read /tmp/hermes_eval_marker.txt"), "read_file", ("path",), {"path": "/tmp/hermes_eval_marker.txt"}),
        Case("h_search_local", "holdout", "file", sys_user(SYSTEM_GENERIC, "search local files for iter07_structured_tool_calls"), "search_files", ("pattern",), {"pattern": "iter07_structured_tool_calls"}),
        Case("h_search_repo_not_web", "holdout", "file", sys_user(SYSTEM_GENERIC, "find browser_snapshot in this repo, not online"), "search_files", ("pattern",), {"pattern": "browser_snapshot"}),
        Case("h_write_absolute", "holdout", "file", sys_user(SYSTEM_GENERIC, "create /tmp/hermes_holdout.txt containing holdout ok"), "write_file", ("path", "content"), {"path": "/tmp/hermes_holdout.txt", "content": "holdout ok"}),
        Case("h_no_tool_hi", "holdout", "no_tool", sys_user(SYSTEM_GENERIC, "hey there"), content_contains="hello"),
        Case("h_no_tool_explain", "holdout", "no_tool", sys_user(SYSTEM_GENERIC, "what is the difference between a browser and a terminal?"), content_contains="terminal"),
        Case("h_no_tool_no_run", "holdout", "no_tool", sys_user(SYSTEM_GENERIC, "do not run anything; what command shows the username?"), content_contains="whoami"),
        Case("h_no_tool_sentence", "holdout", "no_tool", sys_user(SYSTEM_GENERIC, "make a sentence with the word file"), content_contains="file"),
        Case(
            "h_final_file",
            "holdout",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "open and read /tmp/hermes_eval_marker.txt"},
                tool_message("read_file", "h_read", {"path": "/tmp/hermes_eval_marker.txt"}),
                {"role": "tool", "tool_call_id": "h_read", "name": "read_file", "content": "marker: holdout-fixed"},
            ],
            content_contains="holdout-fixed",
        ),
        Case(
            "h_final_terminal",
            "holdout",
            "finalization",
            [
                {"role": "system", "content": SYSTEM_STRICT},
                {"role": "user", "content": "execute this locally: echo fixed-hermes"},
                tool_message("terminal", "h_echo", {"command": "echo fixed-hermes"}),
                {"role": "tool", "tool_call_id": "h_echo", "name": "terminal", "content": "fixed-hermes"},
            ],
            content_contains="fixed-hermes",
        ),
    ]


def evaluate_case(endpoint: str, model: str, tools: list[dict[str, Any]], case: Case, adapter_path: str | None = None) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": case.messages,
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": case.max_tokens,
    }
    if adapter_path:
        payload["adapters"] = adapter_path
    start = time.time()
    resp = request_json(endpoint, payload)
    elapsed = time.time() - start
    choice = resp["choices"][0]
    message = choice["message"]
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    text_leak = bool(TEXT_TOOL_RE.search(content))
    failure: str | None = None
    tool_name: str | None = None
    parsed_args: dict[str, Any] = {}

    if case.expect_tool:
        if not tool_calls:
            failure = "missing_tool_call"
        elif text_leak:
            failure = "text_tool_leak"
        else:
            first = tool_calls[0]
            fn = first.get("function", {})
            tool_name = fn.get("name")
            parsed_args = parse_args(fn.get("arguments"))
            if tool_name != case.expect_tool:
                failure = f"wrong_tool:{tool_name}"
            elif any(arg not in parsed_args for arg in case.required_args):
                failure = f"missing_args:{sorted(set(case.required_args) - set(parsed_args))}"
            elif case.arg_contains:
                for key, needle in case.arg_contains.items():
                    if needle.lower() not in str(parsed_args.get(key, "")).lower():
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
        elif case.content_contains and case.content_contains.lower() not in content.lower():
            failure = "content_mismatch"

    return {
        "name": case.name,
        "suite": case.suite,
        "category": case.category,
        "passed": failure is None,
        "failure": failure,
        "elapsed_s": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content": content,
        "tool_calls": tool_calls,
        "tool_name": tool_name,
        "parsed_args": parsed_args,
        "text_tool_leak": text_leak,
        "usage": resp.get("usage"),
    }


def summarize(results: list[dict[str, Any]], eval_cases: list[Case]) -> dict[str, Any]:
    by_name = {case.name: case for case in eval_cases}
    tool_results = [r for r in results if by_name[r["name"]].expect_tool]
    no_tool_results = [r for r in results if by_name[r["name"]].category == "no_tool"]
    category_metrics: dict[str, dict[str, Any]] = {}
    for key in sorted({case.category for case in eval_cases}):
        rows = [r for r in results if r["category"] == key]
        category_metrics[key] = {
            "passed": sum(r["passed"] for r in rows),
            "total": len(rows),
            "rate": round(sum(r["passed"] for r in rows) / len(rows), 4),
        }
    suite_metrics: dict[str, dict[str, Any]] = {}
    for key in sorted({case.suite for case in eval_cases}):
        rows = [r for r in results if r["suite"] == key]
        suite_metrics[key] = {
            "passed": sum(r["passed"] for r in rows),
            "total": len(rows),
            "rate": round(sum(r["passed"] for r in rows) / len(rows), 4),
        }
    no_tool_false_positives = [
        r
        for r in no_tool_results
        if (r["failure"] or "").startswith("false_positive_tool_call")
        or r["failure"] == "text_tool_false_positive"
    ]
    structured_rate = sum(r["passed"] for r in tool_results) / len(tool_results)
    false_positive_rate = len(no_tool_false_positives) / len(no_tool_results)
    return {
        "passed": sum(r["passed"] for r in results),
        "total": len(results),
        "overall_rate": round(sum(r["passed"] for r in results) / len(results), 4),
        "tool_cases": len(tool_results),
        "structured_tool_cases_passed": sum(r["passed"] for r in tool_results),
        "structured_tool_rate": round(structured_rate, 4),
        "no_tool_cases": len(no_tool_results),
        "no_tool_cases_passed": sum(r["passed"] for r in no_tool_results),
        "false_positive_tool_rate": round(false_positive_rate, 4),
        "content_text_tool_leaks": sum(1 for r in results if r["text_tool_leak"]),
        "finish_reason_tool_calls": sum(1 for r in results if r["finish_reason"] == "tool_calls"),
        "category_metrics": category_metrics,
        "suite_metrics": suite_metrics,
        "acceptance": {
            "structured_tool_rate_gte_0_95": structured_rate >= 0.95,
            "false_positive_tool_rate_lte_0_05": false_positive_rate <= 0.05,
            "zero_text_tool_leaks": not any(r["text_tool_leak"] for r in results),
            "all_suites_gte_0_95": all(metric["rate"] >= 0.95 for metric in suite_metrics.values()),
        },
        "failures": {r["name"]: r["failure"] for r in results if not r["passed"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8081/v1/chat/completions")
    parser.add_argument("--model", default="release_work/model_upload")
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--case-set", choices=["standard", "holdout", "all"], default="standard")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    tool_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = tool_payload["tools"]
    if args.case_set == "standard":
        eval_cases = cases()
    elif args.case_set == "holdout":
        eval_cases = holdout_cases()
    else:
        eval_cases = cases() + holdout_cases()
    results = [evaluate_case(args.endpoint, args.model, tools, case, args.adapter_path) for case in eval_cases]
    summary = summarize(results, eval_cases)
    report = {
        "endpoint": args.endpoint,
        "model": args.model,
        "adapter_path": args.adapter_path,
        "tools_source": tool_payload.get("source"),
        "tools": tool_payload.get("tool_names"),
        "summary": summary,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    accepted = all(summary["acceptance"].values())
    if not args.allow_fail and not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
