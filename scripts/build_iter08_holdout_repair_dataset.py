#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


SYSTEMS = [
    "You are a helpful assistant with tools.",
    "Use available tools when the user's request requires the browser, terminal, or local files. Otherwise answer normally.",
]


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def load_tools(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return raw["tools"] if isinstance(raw, dict) and "tools" in raw else raw


def tool_row(system: str, user: str, tool: str, args: dict[str, Any], tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{stable_id(system, user, tool)}",
                        "type": "function",
                        "function": {"name": tool, "arguments": args},
                    }
                ],
            },
        ],
        "tools": tools,
        "case_id": f"iter08_{stable_id(system, user, tool)}",
        "category": "holdout_repair",
        "kind": "tool_call",
        "expect_tool": tool,
        "source": "iter08_holdout_repair",
    }


def no_tool_row(system: str, user: str, assistant: str, tools: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        "tools": tools,
        "case_id": f"iter08_{stable_id(system, user, assistant)}",
        "category": "no_tool_guard",
        "kind": "no_tool",
        "expect_tool": None,
        "source": "iter08_holdout_repair",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    tools = load_tools(args.tools_json)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    list_prompts = [
        "use shell to list the current folder",
        "list the current folder with terminal",
        "show files here using shell",
        "run a shell command to list files here",
        "from terminal, show this directory's files",
        "use terminal to list this directory",
        "shell list current directory contents",
        "run ls for the current folder",
    ]
    pwd_prompts = [
        ("show current directory with a shell command", "pwd"),
        ("pwd in terminal", "pwd"),
        ("print working directory from terminal", "pwd"),
    ]
    no_tool = [
        ("hey there", "Hello!"),
        ("do not run anything; what command lists files?", "The command is usually `ls`."),
        ("what is the difference between pwd and ls?", "`pwd` prints the current directory, while `ls` lists directory contents."),
        ("write a sentence using the word terminal", "The terminal waited for the next command."),
    ]

    rows: list[dict[str, Any]] = []
    for system in SYSTEMS:
        for prompt in list_prompts:
            rows.append(tool_row(system, prompt, "terminal", {"command": "ls"}, tools))
            rows.append(tool_row(system, "please " + prompt, "terminal", {"command": "ls"}, tools))
        for prompt, command in pwd_prompts:
            rows.append(tool_row(system, prompt, "terminal", {"command": command}, tools))
        for prompt, answer in no_tool:
            rows.append(no_tool_row(system, prompt, answer, tools))

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {"train": rows * 4, "valid": rows, "test": rows}
    for name, split_rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                tokenizer.apply_chat_template(row["messages"], tools=tools, tokenize=False, add_generation_prompt=False)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "format": "Targeted structured assistant tool_calls repair for shell list vs pwd ambiguity.",
        "rows": {name: len(split_rows) for name, split_rows in splits.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
