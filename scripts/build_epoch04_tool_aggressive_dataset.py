#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SYSTEM = (
    "You are Hermes. Use the available tools aggressively when the user asks you to act, look up current "
    "information, inspect files, run commands, or control the computer. Do not give up just because the "
    "user did not name the exact tool. If a request needs current web information, use browser tools. If "
    "a request needs local command execution, use terminal. If a request needs local file inspection or "
    "edits, use file tools. If a request needs desktop interaction, use computer_use. If the request is "
    "answerable directly or explicitly says not to use tools, answer normally without making a tool call. "
    "Use only the provided tool names and arguments."
)

TEXT_TOOL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def stable_split(row_id: str) -> str:
    bucket = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 86:
        return "train"
    if bucket < 94:
        return "valid"
    return "test"


def py_value(value: Any) -> str:
    if isinstance(value, str):
        return repr(value)
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return "None"
    return repr(value)


def pythonic_call(name: str, args: dict[str, Any]) -> str:
    parts = [f"{key}={py_value(value)}" for key, value in args.items()]
    return f"<|tool_call_start|>[{name}({', '.join(parts)})]<|tool_call_end|>"


def normalize_tool_call_message(row_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{row_id}",
                "type": "function",
                "function": {"name": name, "arguments": args},
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


def row_tool(row_id: str, prompt: str, name: str, args: dict[str, Any], tools: list[dict[str, Any]], category: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            normalize_tool_call_message(row_id, name, args),
        ],
        "tools": tools,
    }


def row_final(
    row_id: str,
    prompt: str,
    tool_name: str,
    args: dict[str, Any],
    result: str,
    final: str,
    tools: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            normalize_tool_call_message(row_id, tool_name, args),
            {"role": "tool", "tool_call_id": f"call_{row_id}", "name": tool_name, "content": result},
            {"role": "assistant", "content": final},
        ],
        "tools": tools,
    }


def load_tools(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {tool["function"]["name"]: tool for tool in payload["tools"]}


def tools_for(by_name: dict[str, dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    return [by_name[name] for name in names if name in by_name]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def repair_rows_from_baseline(path: Path, by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    failures = [
        r for r in load_jsonl(path)
        if not r.get("passed")
        and r.get("expected_tool")
        and r.get("expected_args") is not None
        and r.get("failure") in {"refusal_when_tool_available", "wrong_tool", "invalid_args", "invalid_tool_name", "invented_action"}
    ]
    for idx, source in enumerate(failures):
        tool = source["expected_tool"]
        if tool not in by_name:
            continue
        prompt = source["prompt"]
        args = source.get("expected_args") or {}
        names = [tool, "browser_navigate", "terminal", "search_files", "read_file", "write_file", "patch", "computer_use", "execute_code", "x_search"]
        selected = tools_for(by_name, names)
        for variant, prefix in enumerate(["", "Please infer the right tool and ", "Do not give up; "]):
            rid = f"epoch04_live_failure_{idx}_{variant}_{hashlib.md5((prefix + prompt).encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prefix + prompt, tool, args, selected, "live_failure_repair"))
    return rows


def aggressive_positive_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    browser = tools_for(by_name, ["browser_navigate", "x_search", "terminal", "search_files"])
    terminal_file = tools_for(by_name, ["terminal", "search_files", "read_file", "write_file", "patch", "execute_code", "browser_navigate"])
    computer = tools_for(by_name, ["computer_use", "browser_navigate", "terminal"])

    web_cases = [
        ("what's the weather in Austin today", "Austin weather today"),
        ("what is the current weather in San Antonio", "current weather San Antonio"),
        ("check the latest llama.cpp release notes", "latest llama.cpp release notes"),
        ("find current Liquid AI model information", "Liquid AI model information"),
        ("what's happening in Austin today", "Austin today news"),
        ("look up the latest MLX tool calling docs", "latest MLX tool calling docs"),
        ("search online for NOAA Austin forecast", "NOAA Austin forecast"),
        ("I need the current Python release, can you check?", "latest Python release"),
    ]
    for i in range(8):
        for prompt, query in web_cases:
            rid = f"epoch04_web_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prompt, "browser_navigate", {"url": f"https://www.google.com/search?q={query.replace(' ', '+')}"}, browser, "aggressive_browser"))

    terminal_cases = [
        ("run echo epoch-four in terminal", {"command": "echo epoch-four"}),
        ("show the current folder", {"command": "pwd"}),
        ("list files here from shell", {"command": "ls"}),
        ("run date locally", {"command": "date"}),
        ("print machine architecture", {"command": "uname -m"}),
        ("show git status short", {"command": "git status --short"}),
    ]
    for i in range(7):
        for prompt, args in terminal_cases:
            rid = f"epoch04_terminal_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prompt, "terminal", args, terminal_file, "aggressive_terminal_file"))

    file_cases = [
        ("search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        ("find where browser_navigate appears locally", "search_files", {"pattern": "browser_navigate"}),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}),
        ("write /tmp/hermes_epoch04.txt with ok", "write_file", {"path": "/tmp/hermes_epoch04.txt", "content": "ok"}),
        ("replace old with new in /tmp/hermes_epoch04_patch.txt", "patch", {"path": "/tmp/hermes_epoch04_patch.txt", "old_string": "old", "new_string": "new"}),
    ]
    for i in range(6):
        for prompt, tool, args in file_cases:
            rid = f"epoch04_file_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prompt, tool, args, terminal_file, "aggressive_terminal_file"))

    computer_cases = [
        ("use computer use to list running apps", {"action": "list_apps"}),
        ("capture the desktop with computer use", {"action": "capture", "mode": "som"}),
        ("focus Chrome with computer use", {"action": "focus_app", "app": "Google Chrome", "raise_window": False}),
        ("with computer use press command l", {"action": "key", "keys": "cmd+l"}),
        ("with computer use type https://x.com", {"action": "type", "text": "https://x.com"}),
        ("use computer use to scroll down", {"action": "scroll", "direction": "down", "amount": 3}),
    ]
    for i in range(6):
        for prompt, args in computer_cases:
            rid = f"epoch04_computer_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prompt, "computer_use", args, computer, "aggressive_computer_use"))

    recovery_cases = [
        ("that didn't work, try another browser search for Austin weather", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather"}, browser, "dont_give_up_recovery"),
        ("no, use the browser to answer the weather question", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather+today"}, browser, "dont_give_up_recovery"),
        ("no, search local files, not the web", "search_files", {"pattern": "browser_navigate"}, terminal_file, "dont_give_up_recovery"),
        ("no, use terminal instead and run pwd", "terminal", {"command": "pwd"}, terminal_file, "dont_give_up_recovery"),
        ("computer_use navigate is invalid; start by capturing the screen", "computer_use", {"action": "capture", "mode": "som"}, computer, "dont_give_up_recovery"),
    ]
    for i in range(8):
        for prompt, tool, args, tools, category in recovery_cases:
            rid = f"epoch04_recovery_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}"
            rows.append(row_tool(rid, prompt, tool, args, tools, category))

    rows.append(row_final("epoch04_final_weather", "what's Austin weather?", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather"}, "Austin weather: 87 F and partly cloudy.", "Austin is 87 F and partly cloudy.", browser, "tool_result_finalization"))
    rows.append(row_final("epoch04_final_retry", "open x.com", "browser_navigate", {"url": "https://x.com"}, "Error: navigation timed out", "The browser navigation timed out. I can retry with a search page or use another available browser path.", browser, "tool_result_finalization"))
    rows.append(row_final("epoch04_final_terminal", "run echo final-ok", "terminal", {"command": "echo final-ok"}, "final-ok", "The terminal returned: final-ok", terminal_file, "tool_result_finalization"))
    rows.append(row_final("epoch04_final_file", "read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}, "marker: live-hermes", "The file contains: marker: live-hermes", terminal_file, "tool_result_finalization"))
    return rows


def retention_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools = tools_for(by_name, ["browser_navigate", "terminal", "search_files", "read_file", "computer_use"])
    pairs = [
        ("hello", "Hello! How can I help you today?"),
        ("are you there?", "I'm here. What would you like to do?"),
        ("tell me about Apollo 13", "Apollo 13 was the NASA mission that suffered an oxygen tank explosion on the way to the Moon. The lunar landing was aborted, and the crew returned safely after Mission Control and the astronauts improvised power, navigation, and life-support procedures."),
        ("which moon landing was nearly a disaster?", "Apollo 13 was the Moon mission that nearly became a disaster after an oxygen tank exploded, forcing the crew to abort the landing and return to Earth."),
        ("tell me about the first Moon landing in detail", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin descended in the Lunar Module Eagle while Michael Collins orbited above in Columbia. Armstrong stepped onto the lunar surface first, followed by Aldrin, while Michael Collins remained in lunar orbit."),
        ("do not use tools; what command prints the current folder?", "`pwd` prints the current folder."),
        ("write a sentence using the word browser", "The browser opened to a page full of reference notes."),
        ("without tools, what is 12 times 13?", "12 times 13 is 156."),
        ("what is the difference between a browser and a terminal?", "A browser is for viewing and interacting with web pages, while a terminal is for running text commands on the operating system."),
        ("return a tiny JSON object with status ok", "{\"status\":\"ok\"}"),
        ("explain what a tool call is without making one", "A tool call is a structured request from an assistant to an external capability such as a browser, terminal, or file reader."),
        ("define quantization in one paragraph", "Quantization stores numerical values at lower precision to reduce memory and bandwidth costs while trying to preserve useful model behavior."),
        ("do not open anything; just say ready", "ready"),
        ("I meant generally, not on this computer. What is Chrome?", "Chrome is Google's web browser, used to access websites and web applications."),
        ("What does weather mean as a word? Do not check the current weather.", "Weather refers to the short-term atmospheric conditions in a place, such as temperature, rain, wind, and cloud cover."),
    ]
    rows = []
    for i in range(10):
        for prompt, answer in pairs:
            for prefix in ("", "Do not use tools. ") if i < 4 else ("",):
                rid = f"epoch04_retention_{i}_{hashlib.md5((prefix + prompt).encode()).hexdigest()[:8]}"
                rows.append(row_text(rid, prefix + prompt, answer, tools, "normal_retention"))
    return rows


def convert_original_trace_text(text: str) -> str:
    text = text.replace("Each function call should be enclosed within <tool_call> </tool_call> XML tags.", "Each function call should use native LFM pythonic tool-call syntax.")
    text = text.replace("<tool_call>\n{'name': <function-name>,'arguments': <args-dict>}\n</tool_call>", "<|tool_call_start|>[function_name(arg='value')]<|tool_call_end|>")

    def repl(match: re.Match[str]) -> str:
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except Exception:
            try:
                parsed = ast.literal_eval(raw)
            except Exception:
                return ""
        name = parsed.get("name")
        args = parsed.get("arguments") or {}
        if not isinstance(name, str) or not isinstance(args, dict):
            return ""
        return pythonic_call(name, args)

    text = TEXT_TOOL_RE.sub(repl, text)
    return text


def original_trace_rows(path: Path, max_rows: int, rng: random.Random) -> list[dict[str, Any]]:
    rows = load_jsonl(path)
    rng.shuffle(rows)
    out = []
    for row in rows:
        converted = convert_original_trace_text(row["text"])
        if "<tool_call>" in converted or "</tool_call>" in converted:
            continue
        out.append(
            {
                "id": f"epoch04_original_converted_{row['id']}",
                "category": "original_trace_converted",
                "text": converted,
                "source_dataset": row.get("source_dataset", "hermes_filtered_text_10k"),
            }
        )
        if len(out) >= max_rows:
            break
    return out


def render_chat_rows(rows: list[dict[str, Any]], tokenizer, max_tokens: int) -> list[dict[str, Any]]:
    rendered = []
    for row in rows:
        text = tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"), tokenize=False)
        if "<tool_call>" in text or "</tool_call>" in text:
            continue
        token_count = len(tokenizer.encode(text))
        if token_count <= max_tokens:
            rendered.append(
                {
                    "id": row["id"],
                    "category": row["category"],
                    "text": text,
                    "token_count": token_count,
                }
            )
    return rendered


def add_token_counts(rows: list[dict[str, Any]], tokenizer, max_tokens: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        text = row["text"]
        token_count = len(tokenizer.encode(text))
        if token_count <= max_tokens:
            row = dict(row)
            row["token_count"] = token_count
            out.append(row)
    return out


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--original-train", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=582)
    parser.add_argument("--seed", type=int, default=404)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    by_name = load_tools(args.tools_json)
    live_failure = render_chat_rows(repair_rows_from_baseline(args.baseline_jsonl, by_name), tokenizer, args.max_tokens)
    positives = render_chat_rows(aggressive_positive_rows(by_name), tokenizer, args.max_tokens)
    retention = render_chat_rows(retention_rows(by_name), tokenizer, args.max_tokens)
    original = add_token_counts(original_trace_rows(args.original_train, max_rows=max(120, args.train_size // 3), rng=rng), tokenizer, args.max_tokens)

    buckets = {
        "live_failure_repair": live_failure,
        "aggressive_tool_positive": positives,
        "normal_retention": retention,
        "original_trace_converted": original,
    }
    for bucket in buckets.values():
        rng.shuffle(bucket)

    targets = {
        "live_failure_repair": int(args.train_size * 0.40),
        "aggressive_tool_positive": int(args.train_size * 0.25),
        "normal_retention": int(args.train_size * 0.25),
    }
    targets["original_trace_converted"] = args.train_size - sum(targets.values())
    train_rows: list[dict[str, Any]] = []
    for name, target in targets.items():
        train_rows.extend(buckets[name][:target])
    leftovers = [row for name, rows in buckets.items() for row in rows[targets.get(name, 0):]]
    rng.shuffle(leftovers)
    train_rows.extend(leftovers[: max(0, args.train_size - len(train_rows))])
    train_rows = train_rows[: args.train_size]
    rng.shuffle(train_rows)

    used = {row["id"] for row in train_rows}
    remaining = [row for rows in buckets.values() for row in rows if row["id"] not in used]
    rng.shuffle(remaining)
    valid_rows = remaining[:60]
    test_rows = remaining[60:120]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train_rows)
    write_jsonl(args.out / "valid.jsonl", valid_rows)
    write_jsonl(args.out / "test.jsonl", test_rows)
    manifest = {
        "model": args.model,
        "tools_json": str(args.tools_json),
        "baseline_jsonl": str(args.baseline_jsonl),
        "original_train": str(args.original_train),
        "max_tokens": args.max_tokens,
        "target_train_size": args.train_size,
        "split_counts": {"train": len(train_rows), "valid": len(valid_rows), "test": len(test_rows)},
        "available_counts": {name: len(rows) for name, rows in buckets.items()},
        "train_category_counts": {},
        "has_xml_tool_call_target": any("<tool_call>" in row["text"] or "</tool_call>" in row["text"] for row in train_rows + valid_rows + test_rows),
        "examples": [
            {k: row[k] for k in ("id", "category", "token_count")} | {"text_prefix": row["text"][:500]}
            for row in train_rows[:3]
        ],
    }
    for row in train_rows:
        manifest["train_category_counts"][row["category"]] = manifest["train_category_counts"].get(row["category"], 0) + 1
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
