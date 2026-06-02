#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_epoch05_tool_repair_masked_dataset import (  # noqa: E402
    SYSTEM,
    SYSTEM_CHAT,
    assistant_tool_call,
    final_row,
    load_tools,
    read_jsonl,
    stable_id,
    text_row,
    tool_row,
    tool_subset,
    validate_render,
    valid_args,
    write_jsonl,
)


PREFERRED_TOOLS = [
    "browser_navigate",
    "x_search",
    "terminal",
    "search_files",
    "read_file",
    "write_file",
    "patch",
    "execute_code",
    "computer_use",
]


def expected_tools(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return tool_subset(by_name, PREFERRED_TOOLS)


def google(query: str) -> str:
    return "https://www.google.com/search?q=" + query.replace(" ", "+")


def repair_rows_from_failures(paths: list[Path], by_name: dict[str, dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    tools = expected_tools(by_name)
    useful_failures = {
        "wrong_tool",
        "refusal_when_tool_available",
        "invalid_args",
        "invalid_tool_name",
        "invented_action",
        "bad_finalization",
    }
    rows: list[dict[str, Any]] = []
    weighted: list[dict[str, Any]] = []
    for path in paths:
        for row in read_jsonl(path):
            expected = row.get("expected_tool")
            args = row.get("expected_args") or {}
            failure = row.get("failure")
            if not expected or failure not in useful_failures:
                continue
            if not isinstance(args, dict) or not valid_args(by_name, expected, args):
                continue
            weight = 6 if failure in {"wrong_tool", "refusal_when_tool_available"} else 3
            if row.get("category") in {"browser_search_current", "correction_recovery"}:
                weight += 3
            weighted.extend([row] * weight)
    rng.shuffle(weighted)
    for idx, row in enumerate(weighted):
        expected = row["expected_tool"]
        args = row.get("expected_args") or {}
        base_prompt = row["prompt"]
        variants = [
            base_prompt,
            f"Infer the right Hermes tool and do not give up: {base_prompt}",
            f"Use tools when the request needs current info, files, commands, or desktop control: {base_prompt}",
        ]
        for variant, prompt in enumerate(variants):
            row_id = f"epoch06_failure_{idx:05d}_{variant}_{stable_id(prompt, expected, json.dumps(args, sort_keys=True))}"
            rows.append(tool_row(row_id, prompt, expected, args, tools, "live_failure_repair", str(row.get("case_id", "eval_failure"))))
    return rows


def browser_current_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools(by_name)
    web_queries = [
        ("what's the weather in Austin today", "Austin weather today"),
        ("weather in Austin right now", "Austin TX weather right now"),
        ("can you check the current weather for Austin TX", "current weather Austin TX"),
        ("find the latest NOAA forecast for Austin", "NOAA Austin forecast today"),
        ("what is happening in Austin today, check online", "Austin today current news"),
        ("look up the latest LiquidAI LFM2.5 GGUF info", "latest LiquidAI LFM2.5 GGUF"),
        ("check online for current mlx-lm server tool calling docs", "current mlx-lm server tool calling docs"),
        ("find current news about NASA Artemis", "current news NASA Artemis"),
        ("search online for Unsloth LFM2.5 GGUF quants", "Unsloth LFM2.5 GGUF quants"),
        ("what's the current time in Tokyo", "current time Tokyo"),
        ("look online for sjakek LFM-2.5 Hermes tuned model", "sjakek LFM-2.5 Hermes tuned model"),
        ("find the latest llama.cpp release notes", "latest llama.cpp release notes"),
        ("use the internet to answer Austin TX weather", "Austin TX weather"),
        ("check online whether MLX supports pythonic tool calls", "MLX pythonic tool calls"),
        ("find current Liquid AI model information", "Liquid AI current model information"),
        ("search the web for today's NBA news", "today NBA news"),
        ("look up current Apple Silicon MLX performance info", "current Apple Silicon MLX performance"),
        ("search for the latest Python release", "latest Python release"),
        ("open a browser search for weather radar Austin", "weather radar Austin"),
        ("check Google for current weather in San Antonio", "current weather San Antonio"),
    ]
    browser_open = [
        ("open x.com in the browser", "https://x.com"),
        ("navigate the browser to google.com", "https://www.google.com"),
        ("open the MLX documentation website", "https://ml-explore.github.io"),
        ("open Hugging Face", "https://huggingface.co"),
        ("open the llama.cpp GitHub page", "https://github.com/ggml-org/llama.cpp"),
    ]
    x_queries = [
        ("search X for recent posts about LFM2.5", "recent posts about LFM2.5"),
        ("look on X for posts about llama.cpp", "llama.cpp"),
        ("check X for recent Liquid AI posts", "Liquid AI"),
        ("search Twitter/X for MLX LFM2.5", "MLX LFM2.5"),
    ]
    for repeat in range(28):
        for prompt, query in web_queries:
            args = {"url": google(query)}
            if "browser_navigate" in by_name and valid_args(by_name, "browser_navigate", args):
                row_id = f"epoch06_browser_web_{repeat}_{stable_id(prompt, query)}"
                rows.append(tool_row(row_id, prompt, "browser_navigate", args, tools, "browser_current_info", "epoch06_synthetic_web"))
        for prompt, url in browser_open:
            args = {"url": url}
            if "browser_navigate" in by_name and valid_args(by_name, "browser_navigate", args):
                row_id = f"epoch06_browser_open_{repeat}_{stable_id(prompt, url)}"
                rows.append(tool_row(row_id, prompt, "browser_navigate", args, tools, "browser_current_info", "epoch06_synthetic_open"))
    if "x_search" in by_name:
        for repeat in range(24):
            for prompt, query in x_queries:
                args = {"query": query}
                if valid_args(by_name, "x_search", args):
                    row_id = f"epoch06_x_search_{repeat}_{stable_id(prompt, query)}"
                    rows.append(tool_row(row_id, prompt, "x_search", args, tools, "browser_current_info", "epoch06_synthetic_x"))
    return rows


def computer_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools(by_name)
    cases = [
        ("capture the desktop with computer use", {"action": "capture", "mode": "som"}),
        ("use computer use to inspect the screen", {"action": "capture", "mode": "som"}),
        ("list running apps using computer use", {"action": "list_apps"}),
        ("focus Chrome in the background with computer use", {"action": "focus_app", "app": "Google Chrome", "raise_window": False}),
        ("with computer use, press command l in Chrome", {"action": "key", "keys": "cmd+l", "app": "Google Chrome"}),
        ("with computer use, type https://x.com", {"action": "type", "text": "https://x.com"}),
        ("use computer use to scroll down", {"action": "scroll", "direction": "down", "amount": 3}),
        ("wait one second using computer use", {"action": "wait", "seconds": 1}),
        ("computer_use navigate is invalid; start by capturing the screen", {"action": "capture", "mode": "som"}),
        ("do not invent a navigate action; use capture first", {"action": "capture", "mode": "som"}),
        ("use computer use to click element 1", {"action": "click", "element": 1}),
        ("use computer use to type hello into the active field", {"action": "type", "text": "hello"}),
    ]
    for repeat in range(22):
        for prompt, args in cases:
            if "computer_use" in by_name and valid_args(by_name, "computer_use", args):
                row_id = f"epoch06_computer_{repeat}_{stable_id(prompt, json.dumps(args, sort_keys=True))}"
                rows.append(tool_row(row_id, prompt, "computer_use", args, tools, "computer_use_exact_schema", "epoch06_synthetic_computer"))
    return rows


def terminal_file_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools(by_name)
    cases = [
        ("run echo hermes-live-test in the terminal", "terminal", {"command": "echo hermes-live-test"}),
        ("show current directory with a shell command", "terminal", {"command": "pwd"}),
        ("execute uname -m locally", "terminal", {"command": "uname -m"}),
        ("run date in terminal", "terminal", {"command": "date"}),
        ("use terminal to print python version", "terminal", {"command": "python3 --version"}),
        ("run git status --short", "terminal", {"command": "git status --short"}),
        ("shell print $SHELL", "terminal", {"command": "printf '%s\\n' \"$SHELL\""}),
        ("search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        ("find browser_navigate in local files", "search_files", {"pattern": "browser_navigate"}),
        ("find Python files named live_hermes_eval", "search_files", {"pattern": "live_hermes_eval", "target": "files"}),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}),
        ("write /tmp/hermes_live_eval_write.txt with live ok", "write_file", {"path": "/tmp/hermes_live_eval_write.txt", "content": "live ok"}),
        ("replace old with new in /tmp/hermes_live_eval_patch.txt", "patch", {"mode": "replace", "path": "/tmp/hermes_live_eval_patch.txt", "old_string": "old", "new_string": "new"}),
        ("use Python execution to sum numbers 1 through 5", "execute_code", {"code": "print(sum(range(1, 6)))"}),
    ]
    for repeat in range(10):
        for prompt, name, args in cases:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"epoch06_terminal_file_{repeat}_{stable_id(prompt, name)}"
                rows.append(tool_row(row_id, prompt, name, args, tools, "terminal_file_patch", "epoch06_synthetic_terminal_file"))
    return rows


def retention_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools = expected_tools(by_name)
    pairs = [
        ("hello", "Hello! How can I help you today?"),
        ("are you there?", "I'm here. What would you like to do?"),
        ("what happened on Apollo 13?", "Apollo 13 was the NASA mission that suffered an oxygen tank explosion on the way to the Moon. The landing was aborted, but the crew returned safely after Mission Control and the astronauts improvised power, navigation, and life-support procedures."),
        ("which moon landing was nearly a disaster?", "Apollo 13 was the Moon mission that nearly became a disaster after an oxygen tank exploded, forcing the crew to abort the landing and return to Earth."),
        ("tell me about the moon landing in detail", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin descended in the Lunar Module Eagle while Michael Collins orbited above in Columbia. Armstrong stepped onto the lunar surface first, followed by Aldrin, and the crew collected samples, took photographs, and set up experiments."),
        ("no, answer the original question with more detail", "Apollo 13 was nearly disastrous because an oxygen tank exploded in the service module, disabling the planned Moon landing and forcing the crew to conserve power, water, and oxygen while using the lunar module as a lifeboat."),
        ("return a tiny JSON object with status ok", "{\"status\":\"ok\"}"),
        ("do not use tools; what command prints the current folder?", "`pwd` prints the current folder."),
        ("write a sentence using the word browser", "The browser opened to a page full of reference notes."),
        ("I meant generally, not on this computer. What is Chrome?", "Chrome is Google's web browser, used to access websites and web applications."),
        ("without tools, what is 12 times 13?", "12 times 13 is 156."),
        ("explain what a tool call is without making one", "A tool call is a structured request from an assistant to an external capability such as a browser, terminal, or file reader."),
        ("What does weather mean as a word? Do not check the current weather.", "Weather means the short-term atmospheric conditions in a place, such as temperature, rain, wind, humidity, and cloud cover."),
        ("define quantization in one paragraph", "Quantization stores numerical values at lower precision to reduce memory and bandwidth costs while trying to preserve useful model behavior."),
        ("what is the difference between a browser and a terminal?", "A browser is used to view and interact with websites. A terminal is used to run commands on a local system."),
        ("explain MoE routing in two paragraphs", "In a mixture-of-experts model, a router chooses which expert layers should process each token. This lets the model keep many total parameters while activating only a smaller subset per token.\n\nGood routing matters because the selected experts shape the model's answer. During fine-tuning, router behavior can change which experts are used for tasks such as tool calls, code, or ordinary chat."),
        ("do not open anything; just say ready", "Ready."),
        ("say good morning", "Good morning."),
        ("write a two-line poem about Austin rain", "Austin rain taps roofs at dawn,\nWarm streets shine, then clouds move on."),
        ("summarize why long context uses more memory", "Long context uses more memory because the runtime must keep more token state in the KV cache and process longer sequences during attention or recurrent-style layers."),
    ]
    rows: list[dict[str, Any]] = []
    for repeat in range(36):
        for prompt, answer in pairs:
            row_id = f"epoch06_retention_{repeat}_{stable_id(prompt)}"
            rows.append(text_row(row_id, prompt, answer, tools, "normal_chat_retention", "epoch06_retention"))
            if repeat < 18:
                rows.append(text_row(f"{row_id}_notools", "Do not use tools. " + prompt, answer, tools, "normal_chat_retention", "epoch06_retention"))
    return rows


def recovery_finalization_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools(by_name)
    correction_cases = [
        ("yes use your browser tools to do it", "browser_navigate", {"url": google("Austin weather today")}),
        ("that didn't work, try another browser search for Austin weather", "browser_navigate", {"url": google("Austin weather today")}),
        ("no, use the terminal instead and run pwd", "terminal", {"command": "pwd"}),
        ("no, search local files, not the web", "search_files", {"pattern": "browser_navigate"}),
        ("use computer use, start by capturing the screen", "computer_use", {"action": "capture", "mode": "som"}),
        ("computer_use navigate is invalid; use capture first", "computer_use", {"action": "capture", "mode": "som"}),
        ("I mean browser_navigate to https://x.com", "browser_navigate", {"url": "https://x.com"}),
        ("try another way: terminal echo recovery-ok", "terminal", {"command": "echo recovery-ok"}),
    ]
    for repeat in range(9):
        for prompt, name, args in correction_cases:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"epoch06_recovery_{repeat}_{stable_id(prompt, name)}"
                rows.append(tool_row(row_id, prompt, name, args, tools, "correction_recovery", "epoch06_recovery"))
    final_cases = [
        ("what's Austin weather?", "browser_navigate", {"url": google("Austin weather")}, "Austin weather: 87 F and partly cloudy.", "Austin is 87 F and partly cloudy."),
        ("run echo final-ok", "terminal", {"command": "echo final-ok"}, "final-ok", "The terminal returned: final-ok"),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}, "marker: live-hermes", "The file contains: marker: live-hermes"),
        ("open x.com", "browser_navigate", {"url": "https://x.com"}, "Error: navigation timed out", "The browser navigation timed out. I can retry with a search page or use another available browser path."),
        ("sum 1 through 5", "execute_code", {"code": "print(sum(range(1,6)))"}, "15", "The code returned 15."),
    ]
    for repeat in range(8):
        for prompt, name, args, result, final in final_cases:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"epoch06_final_{repeat}_{stable_id(prompt, name)}"
                rows.append(final_row(row_id, prompt, name, args, result, final, tools, "tool_result_finalization"))
    return rows


def pick_rows(buckets: dict[str, list[dict[str, Any]]], train_size: int, rng: random.Random) -> list[dict[str, Any]]:
    targets = {
        "browser_current_info": int(train_size * 0.35),
        "normal_chat_retention": int(train_size * 0.25),
        "live_failure_repair": int(train_size * 0.20),
        "computer_use_exact_schema": int(train_size * 0.10),
        "correction_recovery": int(train_size * 0.03),
        "tool_result_finalization": int(train_size * 0.02),
    }
    targets["terminal_file_patch"] = train_size - sum(targets.values())
    for rows in buckets.values():
        rng.shuffle(rows)
    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for name, target in targets.items():
        for row in buckets.get(name, [])[:target]:
            if row["case_id"] not in used:
                picked.append(row)
                used.add(row["case_id"])
    leftovers = [row for rows in buckets.values() for row in rows if row["case_id"] not in used]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, train_size - len(picked))])
    rng.shuffle(picked)
    return picked[:train_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--valid-size", type=int, default=240)
    parser.add_argument("--test-size", type=int, default=240)
    parser.add_argument("--seed", type=int, default=606)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _, by_name = load_tools(args.tools_json)
    buckets = {
        "browser_current_info": browser_current_rows(by_name),
        "normal_chat_retention": retention_rows(by_name),
        "live_failure_repair": repair_rows_from_failures(args.baseline_jsonl, by_name, rng),
        "computer_use_exact_schema": computer_rows(by_name),
        "terminal_file_patch": terminal_file_rows(by_name),
        "correction_recovery": [],
        "tool_result_finalization": [],
    }
    for row in recovery_finalization_rows(by_name):
        buckets.setdefault(row["category"], []).append(row)

    valid_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for row in [item for rows in buckets.values() for item in rows]:
        ok, token_count, reason = validate_render(row, tokenizer, args.max_tokens)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        row = dict(row)
        row["token_count"] = token_count
        valid_rows.append(row)
    by_id = {row["case_id"]: row for row in valid_rows}
    valid_rows = list(by_id.values())
    buckets = {}
    for row in valid_rows:
        buckets.setdefault(row["category"], []).append(row)

    train = pick_rows(buckets, args.train_size, rng)
    used = {row["case_id"] for row in train}
    remaining = [row for row in valid_rows if row["case_id"] not in used]
    rng.shuffle(remaining)
    valid = remaining[: args.valid_size]
    test = remaining[args.valid_size : args.valid_size + args.test_size]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    write_jsonl(args.out / "test.jsonl", test)
    manifest: dict[str, Any] = {
        "name": "semi_epoch06_tool_repair_expanded_10k",
        "model": args.model,
        "tools_json": str(args.tools_json),
        "baseline_jsonl": [str(path) for path in args.baseline_jsonl],
        "max_tokens": args.max_tokens,
        "target_train_size": args.train_size,
        "split_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "available_counts": {name: len(rows) for name, rows in buckets.items()},
        "train_category_counts": {},
        "rejected_counts": rejected,
        "has_xml_tool_call_target": False,
        "format": "messages+tools JSONL for prompt-masked MLX semi-full grouped training",
        "system_policy": SYSTEM,
        "chat_retention_policy": SYSTEM_CHAT,
    }
    for row in train:
        manifest["train_category_counts"][row["category"]] = manifest["train_category_counts"].get(row["category"], 0) + 1
    manifest["examples"] = [
        {
            "case_id": row["case_id"],
            "category": row["category"],
            "kind": row["kind"],
            "expect_tool": row.get("expect_tool"),
            "token_count": row["token_count"],
        }
        for row in train[:8]
    ]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
