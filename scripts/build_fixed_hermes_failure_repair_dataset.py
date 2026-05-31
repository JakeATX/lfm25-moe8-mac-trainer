#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from build_fixed_hermes_contrast_router_dataset import tool_call, write_jsonl


SYSTEMS = [
    None,
    "You are a helpful assistant with tools.",
    "Use available tools when the user's request requires current web information, local shell execution, or local file search.",
]


def ua(user: str, assistant: str, system: str | None = None) -> list[dict[str, Any]]:
    msgs: list[dict[str, Any]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend([{"role": "user", "content": user}, {"role": "assistant", "content": assistant}])
    return msgs


def build_rows(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    hard_calls = [
        ("what's the weather today in Austin TX", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+TX+weather+today"}, "weather_browser"),
        ("can you check the current weather for Austin", "browser_navigate", {"url": "https://www.google.com/search?q=current+Austin+TX+weather"}, "weather_browser"),
        ("weather today", "browser_navigate", {"url": "https://www.google.com/search?q=weather+today"}, "weather_browser"),
        ("check online for the forecast in Austin", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+TX+forecast"}, "weather_browser"),
        ("look up current Liquid AI LFM2.5 context length", "browser_navigate", {"url": "https://www.google.com/search?q=Liquid+AI+LFM2.5+context+length"}, "web_current"),
        ("find current LFM2.5 context length online", "browser_navigate", {"url": "https://www.google.com/search?q=current+LFM2.5+context+length"}, "web_current"),
        ("pwd in terminal", "terminal", {"command": "pwd"}, "terminal_pwd"),
        ("run pwd", "terminal", {"command": "pwd"}, "terminal_pwd"),
        ("show the current directory from shell", "terminal", {"command": "pwd"}, "terminal_pwd"),
        ("list files here from shell", "terminal", {"command": "ls"}, "terminal_ls"),
        ("ls this directory", "terminal", {"command": "ls"}, "terminal_ls"),
        ("show files in this folder using terminal", "terminal", {"command": "ls"}, "terminal_ls"),
        ("find where HERMES_PARSE_TEXT_TOOL_CALLS appears locally", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS", "path": "."}, "local_search"),
        ("search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS", "path": "."}, "local_search"),
        ("search local files for browser_navigate", "search_files", {"pattern": "browser_navigate", "path": "."}, "local_search"),
    ]
    prefixes = ["", "please ", "can you ", "real quick ", "i need you to "]
    for system in SYSTEMS:
        for user, tool, kwargs, source in hard_calls:
            for prefix in prefixes:
                rows.append(
                    {
                        "messages": ua(prefix + user, tool_call(tool, **kwargs), system),
                        "tools": tools,
                        "category": "failure_repair",
                        "source": source,
                        "expect_tool": tool,
                    }
                )

    no_tool = [
        ("hello!", "Hello!"),
        ("what command prints the current directory? do not run it", "The command is `pwd`."),
        ("write a sentence using the word browser", "The browser opened a quiet page."),
        ("explain local file search in one sentence", "Local file search scans files on the machine for matching text or filenames."),
        ("do not use tools, just say done", "done"),
    ]
    for system in SYSTEMS:
        for user, assistant in no_tool:
            rows.append(
                {
                    "messages": ua(user, assistant, system),
                    "tools": tools,
                    "category": "no_tool_guard",
                    "source": "hard_negative",
                    "expect_tool": None,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=906)
    parser.add_argument("--train-size", type=int, default=2400)
    parser.add_argument("--valid-size", type=int, default=240)
    parser.add_argument("--test-size", type=int, default=240)
    args = parser.parse_args()

    tools_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = tools_payload["tools"]
    rows = build_rows(tools)
    rng = random.Random(args.seed)
    need = args.train_size + args.valid_size + args.test_size
    expanded = [rows[i % len(rows)] | {"case_id": f"failure_repair_{i:05d}"} for i in range(need)]
    rng.shuffle(expanded)
    train = expanded[: args.train_size]
    valid = expanded[args.train_size : args.train_size + args.valid_size]
    test = expanded[args.train_size + args.valid_size :]
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    write_jsonl(args.out / "test.jsonl", test)
    manifest = {
        "name": "iter06_fixed_hermes_failure_repair",
        "seed": args.seed,
        "sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "tool_names": tools_payload.get("tool_names"),
        "purpose": "Focused continuation set for iter05 missing-tool-call failures.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
