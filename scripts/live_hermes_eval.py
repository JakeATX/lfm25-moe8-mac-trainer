#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TEXT_TOOL_RE = re.compile(
    r"<\|tool_call_start\|>|<\|tool_call_end\|>|<tool_call>|</tool_call>|\[[a-zA-Z_]\w*\([^)]*\)\]"
)
SYSTEM = (
    "You are Hermes. Use the provided tools when they are needed to satisfy the user. "
    "If no tool is needed, answer normally. Do not invent tool names or arguments."
)


@dataclass
class Case:
    case_id: str
    category: str
    prompt: str
    expect_tool: str | None = None
    expected_args: dict[str, Any] = field(default_factory=dict)
    required_args: tuple[str, ...] = ()
    content_contains: str | None = None
    turns: list[dict[str, Any]] | None = None
    safe_to_execute_live: bool = False
    notes: str = ""


def request_json(endpoint: str, payload: dict[str, Any], timeout: int = 600) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(endpoint, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def parse_call_args(raw: str | dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}


def messages_for(case: Case) -> list[dict[str, Any]]:
    if case.turns:
        return case.turns
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": case.prompt}]


def tool_turn(name: str, call_id: str, args: dict[str, Any], result: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ],
        },
        {"role": "tool", "tool_call_id": call_id, "name": name, "content": result},
    ]


def build_cases(tool_names: set[str]) -> list[Case]:
    cases: list[Case] = []

    def add(case: Case) -> None:
        if case.expect_tool is None or case.expect_tool in tool_names:
            cases.append(case)

    browser_tool = "browser_navigate" if "browser_navigate" in tool_names else None
    # Hermes' x_search tool is specifically for X/Twitter. General current
    # info, weather, docs, and latest-web tasks should route to browser tools.
    search_tool = browser_tool
    file_search_tool = "search_files" if "search_files" in tool_names else None

    normal_prompts = [
        ("normal_hello", "hello", "hello"),
        ("normal_moon", "what happened on Apollo 13?", "Apollo"),
        ("normal_moe", "explain MoE routing in two paragraphs", "expert"),
        ("normal_do_not_run", "do not use tools; what command prints the current folder?", "pwd"),
        ("normal_browser_word", "write a sentence using the word browser", "browser"),
        ("normal_followup_context", "In one sentence, what was Apollo 11?", "Apollo"),
        ("normal_no_reset", "no, answer the original question with more detail", None),
        ("normal_math_no_tool", "without tools, what is 12 times 13?", "156"),
        ("normal_concept", "what is the difference between a browser and a terminal?", "terminal"),
        ("normal_summary", "summarize why long context uses more memory", "context"),
        ("normal_greeting2", "are you there?", "here"),
        ("normal_safety", "do not open anything; just say ready", "ready"),
        ("normal_history", "which moon landing was nearly a disaster?", "Apollo 13"),
        ("normal_poem", "write a two-line poem about Austin rain", None),
        ("normal_json", "return a tiny JSON object with status ok", "ok"),
        ("normal_define", "define quantization in one paragraph", "precision"),
        ("normal_tool_mentions", "Explain what a tool call is without making one.", "tool"),
        ("normal_correction", "I meant generally, not on this computer. What is Chrome?", "browser"),
        ("normal_short", "say good morning", "morning"),
        ("normal_detail", "tell me about the moon landing in detail", "1969"),
    ]
    for case_id, prompt, contains in normal_prompts:
        add(Case(case_id, "normal_chat", prompt, content_contains=contains))

    if search_tool:
        web_prompts = [
            ("web_weather_austin", "what's the weather in Austin today", "weather Austin today"),
            ("web_weather_plain", "can you check the current weather for Austin TX", "current weather Austin TX"),
            ("web_latest_lfm", "look up the latest LiquidAI LFM2.5 GGUF info", "latest LiquidAI LFM2.5 GGUF"),
            ("web_current_mlx", "check online for current mlx-lm server tool calling docs", "current mlx-lm server tool calling docs"),
            ("web_news", "find current news about NASA Artemis", "current news NASA Artemis"),
            ("web_search_noaa", "search online for NOAA Austin forecast", "NOAA Austin forecast"),
            ("web_weather_follow", "yes use your browser tools to get Austin weather", "Austin weather"),
            ("web_latest_python", "check the web for the latest Python release", "latest Python release"),
            ("web_current_time", "find the current time in Tokyo online", "current time Tokyo"),
            ("web_search_hf", "look online for sjakek LFM-2.5 Hermes tuned model", "sjakek LFM-2.5 Hermes tuned model"),
            ("web_current_llama", "find the latest llama.cpp release notes", "latest llama.cpp release notes"),
            ("web_weather_satx", "what is the current weather in San Antonio", "current weather San Antonio"),
            ("web_search_unsloth", "search for Unsloth LFM2.5 GGUF quants", "Unsloth LFM2.5 GGUF quants"),
            ("web_latest_austin", "what's happening in Austin today, check online", "Austin today"),
            ("web_search_liquid", "find current Liquid AI model information", "Liquid AI model information"),
            ("web_query_weather", "use the internet to answer Austin TX weather", "Austin TX weather"),
            ("web_check_online", "check online whether MLX supports pythonic tool calls", "MLX pythonic tool calls"),
            ("web_x_search", "search X for recent posts about LFM2.5", "recent posts about LFM2.5"),
        ]
        for case_id, prompt, query in web_prompts:
            if case_id == "web_x_search" and "x_search" in tool_names:
                add(Case(case_id, "browser_search_current", prompt, "x_search", {"query": query}, ("query",)))
            elif search_tool == "x_search":
                add(Case(case_id, "browser_search_current", prompt, search_tool, {"query": query}, ("query",)))
            else:
                add(Case(case_id, "browser_search_current", prompt, search_tool, {"url": f"https://www.google.com/search?q={query.replace(' ', '+')}"}, ("url",)))
        add(Case("web_open_x", "browser_search_current", "open x.com in the browser", "browser_navigate", {"url": "https://x.com"}, ("url",)))
        add(Case("web_open_google", "browser_search_current", "navigate the browser to google.com", "browser_navigate", {"url": "https://www.google.com"}, ("url",)))
        add(Case("web_open_docs", "browser_search_current", "open the MLX documentation website", "browser_navigate", {"url": "https://ml-explore.github.io"}, ("url",)))

    terminal_cases = [
        ("term_echo", "run echo hermes-live-test in the terminal", {"command": "echo hermes-live-test"}),
        ("term_pwd", "show current directory with a shell command", {"command": "pwd"}),
        ("term_uname", "execute uname -m locally", {"command": "uname -m"}),
        ("term_date", "run date in terminal", {"command": "date"}),
        ("term_python", "use terminal to print python version", {"command": "python3 --version"}),
        ("term_git_status", "run git status --short", {"command": "git status --short"}),
        ("term_env", "shell print $SHELL", {"command": "printf '%s\\n' \"$SHELL\""}),
        ("term_process", "show current llama server process with ps", {"command": "ps aux | grep llama"}),
    ]
    for case_id, prompt, args in terminal_cases:
        add(Case(case_id, "terminal_file_patch", prompt, "terminal", args, ("command",), safe_to_execute_live=True))

    if file_search_tool:
        file_cases = [
            ("file_search_patch", "search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", file_search_tool, {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}, ("pattern",)),
            ("file_search_browser", "find browser_navigate in local files", file_search_tool, {"pattern": "browser_navigate"}, ("pattern",)),
            ("file_search_iter", "search local files for iter13_llamacpp_chat_retention", file_search_tool, {"pattern": "iter13_llamacpp_chat_retention"}, ("pattern",)),
            ("file_find_py", "find Python files named live_hermes_eval", file_search_tool, {"pattern": "live_hermes_eval", "target": "files"}, ("pattern",)),
        ]
        for case_id, prompt, tool, args, required in file_cases:
            add(Case(case_id, "terminal_file_patch", prompt, tool, args, required))
    for case in [
        Case("file_read_tmp", "terminal_file_patch", "read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}, ("path",)),
        Case("file_write_tmp", "terminal_file_patch", "write /tmp/hermes_live_eval_write.txt with live ok", "write_file", {"path": "/tmp/hermes_live_eval_write.txt", "content": "live ok"}, ("path", "content")),
        Case("file_patch_tmp", "terminal_file_patch", "replace old with new in /tmp/hermes_live_eval_patch.txt", "patch", {"path": "/tmp/hermes_live_eval_patch.txt", "old_string": "old", "new_string": "new"}, ("path", "old_string", "new_string")),
        Case("code_exec_sum", "terminal_file_patch", "use Python execution to sum numbers 1 through 5", "execute_code", {"code": "print(sum(range(1, 6)))"}, ("code",)),
        Case("code_exec_json", "terminal_file_patch", "run Python code to print JSON status ok", "execute_code", {"code": "import json; print(json.dumps({'status':'ok'}))"}, ("code",)),
        Case("browser_capture", "computer_use_browser_control", "capture the desktop with computer use", "computer_use", {"action": "capture", "mode": "som"}, ("action",)),
        Case("computer_list_apps", "computer_use_browser_control", "list running apps using computer use", "computer_use", {"action": "list_apps"}, ("action",)),
        Case("computer_focus_chrome", "computer_use_browser_control", "focus Chrome in the background with computer use", "computer_use", {"action": "focus_app", "app": "Google Chrome", "raise_window": False}, ("action",)),
        Case("computer_key_new_tab", "computer_use_browser_control", "with computer use, press command l in Chrome", "computer_use", {"action": "key", "keys": "cmd+l", "app": "Google Chrome"}, ("action", "keys")),
        Case("computer_type_url", "computer_use_browser_control", "with computer use, type https://x.com", "computer_use", {"action": "type", "text": "https://x.com"}, ("action", "text")),
        Case("computer_wait", "computer_use_browser_control", "wait one second using computer use", "computer_use", {"action": "wait", "seconds": 1}, ("action",)),
        Case("computer_scroll", "computer_use_browser_control", "use computer use to scroll down", "computer_use", {"action": "scroll", "direction": "down", "amount": 3}, ("action", "direction")),
        Case("computer_wrong_navigate_repair", "computer_use_browser_control", "use computer use to navigate to x.com", "computer_use", {"action": "key", "keys": "cmd+l"}, ("action",), notes="Should not invent computer_use(action='navigate')."),
        Case("browser_control_navigate", "computer_use_browser_control", "open a headed browser tab to x.com", "browser_navigate", {"url": "https://x.com"}, ("url",)),
        Case("browser_control_google", "computer_use_browser_control", "open Google in the browser", "browser_navigate", {"url": "https://www.google.com"}, ("url",)),
    ]:
        add(case)

    correction_cases = [
        ("corr_use_browser", "yes use your browser tools to do it", search_tool, {"query": "Austin weather today"} if search_tool == "x_search" else {"url": "https://www.google.com/search?q=Austin+weather+today"}, ("query",) if search_tool == "x_search" else ("url",)),
        ("corr_that_failed", "that didn't work, try another browser search for Austin weather", search_tool, {"query": "Austin weather today"} if search_tool == "x_search" else {"url": "https://www.google.com/search?q=Austin+weather+today"}, ("query",) if search_tool == "x_search" else ("url",)),
        ("corr_use_terminal", "no, use the terminal instead and run pwd", "terminal", {"command": "pwd"}, ("command",)),
        ("corr_search_repo", "no, search local files, not the web", file_search_tool, {"pattern": "browser_navigate"}, ("pattern",)),
        ("corr_computer_capture", "use computer use, start by capturing the screen", "computer_use", {"action": "capture", "mode": "som"}, ("action",)),
        ("corr_no_tool", "no, don't use tools, just answer in words", None, {}, ()),
        ("corr_retry_file", "that file attempt failed; search for the filename locally", file_search_tool, {"pattern": "hermes_live_eval"}, ("pattern",)),
        ("corr_bad_action", "computer_use navigate is invalid; use capture first", "computer_use", {"action": "capture", "mode": "som"}, ("action",)),
        ("corr_browser_specific", "I mean browser_navigate to https://x.com", "browser_navigate", {"url": "https://x.com"}, ("url",)),
        ("corr_terminal_echo", "try another way: terminal echo recovery-ok", "terminal", {"command": "echo recovery-ok"}, ("command",)),
    ]
    for case_id, prompt, tool, args, required in correction_cases:
        if tool is None:
            add(Case(case_id, "correction_recovery", prompt, content_contains="answer"))
        else:
            add(Case(case_id, "correction_recovery", prompt, tool, args, required))

    final_cases = [
        ("final_weather", "what's Austin weather?", search_tool or "browser_navigate", {"query": "Austin weather"} if search_tool == "x_search" else {"url": "https://www.google.com/search?q=Austin+weather"}, "Austin is 87 F and partly cloudy.", "87"),
        ("final_terminal", "run echo final-ok", "terminal", {"command": "echo final-ok"}, "final-ok", "final-ok"),
        ("final_read", "read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}, "marker: live-hermes", "live-hermes"),
        ("final_code", "sum 1 through 5", "execute_code", {"code": "print(sum(range(1,6)))"}, "15", "15"),
        ("final_failed_tool", "open x.com", "browser_navigate", {"url": "https://x.com"}, "Error: navigation timed out", "timed out"),
    ]
    for case_id, prompt, tool, args, result, contains in final_cases:
        if tool in tool_names:
            turns = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
            turns.extend(tool_turn(tool, f"call_{case_id}", args, result))
            add(Case(case_id, "tool_result_finalization", prompt, content_contains=contains, turns=turns))

    return cases


def classify(case: Case, message: dict[str, Any], finish_reason: str | None, tool_names: set[str]) -> tuple[bool, str | None, dict[str, Any]]:
    content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []
    text_leak = bool(TEXT_TOOL_RE.search(content))
    detail: dict[str, Any] = {
        "text_tool_leak": text_leak,
        "tool_name": None,
        "parsed_args": {},
    }
    if case.expect_tool:
        if text_leak:
            return False, "text_tool_leak", detail
        if not tool_calls:
            lowered = content.lower()
            if "don't have access" in lowered or "cannot access" in lowered or "no access" in lowered:
                return False, "refusal_when_tool_available", detail
            return False, "refusal_when_tool_available", detail
        first = tool_calls[0]
        fn = first.get("function", {})
        name = fn.get("name")
        args = parse_call_args(fn.get("arguments"))
        detail["tool_name"] = name
        detail["parsed_args"] = args
        if name not in tool_names:
            return False, "invalid_tool_name", detail
        if name != case.expect_tool:
            return False, "wrong_tool", detail
        missing = [key for key in case.required_args if key not in args]
        if missing:
            return False, "invalid_args", {**detail, "missing_args": missing}
        if name == "computer_use" and args.get("action") == "navigate":
            return False, "invented_action", detail
        if name == "computer_use":
            action = args.get("action")
            allowed = {
                "capture",
                "click",
                "double_click",
                "right_click",
                "middle_click",
                "drag",
                "scroll",
                "type",
                "key",
                "set_value",
                "wait",
                "list_apps",
                "focus_app",
            }
            if action not in allowed:
                return False, "invented_action", detail
        for key, expected in case.expected_args.items():
            actual = str(args.get(key, ""))
            if key == "url":
                if str(expected).split("?")[0].lower() not in actual.lower() and not any(
                    token.lower() in actual.lower() for token in str(expected).replace("+", " ").split()[:3]
                ):
                    return False, "invalid_args", detail
            elif str(expected).lower() not in actual.lower():
                return False, "invalid_args", detail
        if finish_reason != "tool_calls":
            return False, "text_tool_leak" if content else "invalid_args", detail
        return True, None, detail

    if tool_calls:
        name = tool_calls[0].get("function", {}).get("name")
        return False, "over_tooling_no_tool_prompt", {**detail, "tool_name": name}
    if text_leak:
        return False, "text_tool_leak", detail
    if case.category == "tool_result_finalization":
        if case.content_contains and case.content_contains.lower() not in content.lower():
            return False, "bad_finalization", detail
    elif case.content_contains and case.content_contains.lower() not in content.lower():
        return False, "normal_chat_regression", detail
    return True, None, detail


def evaluate_case(endpoint: str, model: str, tools: list[dict[str, Any]], case: Case, adapter_path: str | None) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages_for(case),
        "tools": tools,
        "tool_choice": "auto",
        "temperature": 0,
        "max_tokens": 512,
    }
    if adapter_path:
        payload["adapters"] = adapter_path
    start = time.time()
    try:
        resp = request_json(endpoint, payload)
        elapsed = time.time() - start
        choice = resp["choices"][0]
        message = choice.get("message", {})
        finish_reason = choice.get("finish_reason")
        passed, failure, detail = classify(case, message, finish_reason, {tool["function"]["name"] for tool in tools})
        return {
            "case_id": case.case_id,
            "category": case.category,
            "prompt": case.prompt,
            "expected_tool": case.expect_tool,
            "expected_args": case.expected_args,
            "required_args": list(case.required_args),
            "passed": passed,
            "failure": failure,
            "finish_reason": finish_reason,
            "elapsed_s": round(elapsed, 3),
            "assistant_visible_text": message.get("content") or "",
            "tool_calls": message.get("tool_calls") or [],
            "tool_name": detail.get("tool_name"),
            "parsed_args": detail.get("parsed_args"),
            "text_tool_leak": detail.get("text_tool_leak", False),
            "usage": resp.get("usage"),
            "notes": case.notes,
        }
    except Exception as exc:
        return {
            "case_id": case.case_id,
            "category": case.category,
            "prompt": case.prompt,
            "expected_tool": case.expect_tool,
            "expected_args": case.expected_args,
            "required_args": list(case.required_args),
            "passed": False,
            "failure": "runtime_error",
            "error": str(exc),
            "elapsed_s": round(time.time() - start, 3),
        }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    tool_cases = [r for r in results if r["expected_tool"]]
    no_tool = [r for r in results if not r["expected_tool"] and r["category"] != "tool_result_finalization"]
    labels: dict[str, int] = {}
    cats: dict[str, dict[str, Any]] = {}
    for row in results:
        if row.get("failure"):
            labels[row["failure"]] = labels.get(row["failure"], 0) + 1
        cat = cats.setdefault(row["category"], {"passed": 0, "total": 0})
        cat["total"] += 1
        cat["passed"] += int(row["passed"])
    for metric in cats.values():
        metric["rate"] = round(metric["passed"] / metric["total"], 4) if metric["total"] else 0
    structured_pass = sum(1 for r in tool_cases if r["passed"])
    no_tool_fp = sum(1 for r in no_tool if r.get("failure") == "over_tooling_no_tool_prompt")
    return {
        "passed": sum(1 for r in results if r["passed"]),
        "total": total,
        "overall_rate": round(sum(1 for r in results if r["passed"]) / total, 4) if total else 0,
        "tool_required_cases": len(tool_cases),
        "valid_structured_tool_calls": structured_pass,
        "valid_structured_tool_rate": round(structured_pass / len(tool_cases), 4) if tool_cases else 0,
        "no_tool_cases": len(no_tool),
        "no_tool_false_positive_rate": round(no_tool_fp / len(no_tool), 4) if no_tool else 0,
        "text_tool_leaks": sum(1 for r in results if r.get("text_tool_leak")),
        "invented_tool_names": labels.get("invalid_tool_name", 0),
        "invented_computer_actions": labels.get("invented_action", 0),
        "failure_labels": labels,
        "category_metrics": cats,
        "acceptance": {
            "overall_rate_gte_0_90": (sum(1 for r in results if r["passed"]) / total) >= 0.90 if total else False,
            "valid_structured_tool_rate_gte_0_95": (structured_pass / len(tool_cases)) >= 0.95 if tool_cases else False,
            "zero_text_tool_leaks": not any(r.get("text_tool_leak") for r in results),
            "zero_invented_tool_names": labels.get("invalid_tool_name", 0) == 0,
            "zero_invented_computer_actions": labels.get("invented_action", 0) == 0,
            "no_tool_false_positive_lte_0_05": (no_tool_fp / len(no_tool)) <= 0.05 if no_tool else False,
        },
    }


def write_report(path: Path, name: str, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [f"# Live Hermes Eval: {name}", "", "## Summary", ""]
    for key in ["passed", "total", "overall_rate", "valid_structured_tool_rate", "no_tool_false_positive_rate", "text_tool_leaks"]:
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    lines.extend(["", "## Category Metrics", ""])
    for cat, metric in sorted(summary["category_metrics"].items()):
        lines.append(f"- `{cat}`: {metric['passed']}/{metric['total']} ({metric['rate']})")
    lines.extend(["", "## Failures", ""])
    failures = [r for r in results if not r["passed"]]
    if not failures:
        lines.append("No failures.")
    else:
        for row in failures:
            text = (row.get("assistant_visible_text") or "").replace("\n", " ")[:240]
            lines.append(f"- `{row['case_id']}` `{row['category']}` `{row.get('failure')}` expected `{row.get('expected_tool')}` got `{row.get('tool_name')}`: {text}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run direct OpenAI-compatible live Hermes tool-router eval.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--name", default="live_iter01")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    tool_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = tool_payload["tools"]
    tool_names = {tool["function"]["name"] for tool in tools}
    cases = build_cases(tool_names)
    if len(cases) < 80:
        raise SystemExit(f"Live suite generated only {len(cases)} cases; expected at least 80.")
    if args.limit:
        cases = cases[: args.limit]

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    Path("/tmp/hermes_live_eval_marker.txt").write_text("marker: live-hermes\n", encoding="utf-8")
    Path("/tmp/hermes_live_eval_patch.txt").write_text("old\n", encoding="utf-8")
    results = []
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, case in enumerate(cases, start=1):
            row = evaluate_case(args.endpoint, args.model, tools, case, args.adapter_path)
            row["index"] = idx
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"index": idx, "case_id": row["case_id"], "passed": row["passed"], "failure": row.get("failure")}))

    summary = summarize(results)
    summary_path = args.out_jsonl.with_suffix(".summary.json")
    summary_path.write_text(json.dumps({"summary": summary, "tools_source": tool_payload.get("source")}, indent=2), encoding="utf-8")
    write_report(args.out_report, args.name, summary, results)
    print(json.dumps(summary, indent=2))
    accepted = all(summary["acceptance"].values())
    if not args.allow_fail and not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
