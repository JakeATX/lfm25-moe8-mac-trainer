#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOOL_MARKUP_RE = re.compile(r"<\|tool_call_start\|>|<tool_call>|\[[a-zA-Z_]\w*\([^)]*=.*\)\]")


@dataclass(frozen=True)
class Example:
    case_id: str
    category: str
    kind: str
    messages: tuple[dict[str, Any], ...]
    expect_tool: str | None
    source: str


def stable_id(*parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:16]


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def tool_call(name: str, **kwargs: str | int | bool) -> str:
    args = ", ".join(f"{key}={quote(str(value))}" if not isinstance(value, bool) else f"{key}={str(value)}" for key, value in kwargs.items())
    return f"<|tool_call_start|>[{name}({args})]<|tool_call_end|>"


def required_args_by_tool(tools: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for tool in tools:
        fn = tool["function"]
        params = fn.get("parameters") or {}
        out[fn["name"]] = set(params.get("required") or [])
    return out


def validate_pythonic_call(target: str, required: dict[str, set[str]]) -> tuple[str, dict[str, Any]]:
    match = re.fullmatch(r"<\|tool_call_start\|>\[(.+)\]<\|tool_call_end\|>", target.strip())
    if not match:
        raise ValueError(f"not native LFM tool syntax: {target}")
    node = ast.parse(match.group(1), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"not a pythonic call: {target}")
    name = node.func.id
    if name not in required:
        raise ValueError(f"unknown Hermes tool {name}: {target}")
    kwargs: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError(f"star kwargs are not allowed: {target}")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    missing = required[name] - set(kwargs)
    if missing:
        raise ValueError(f"missing required args {sorted(missing)} for {name}: {target}")
    return name, kwargs


def make_example(
    category: str,
    kind: str,
    messages: list[dict[str, Any]],
    expect_tool: str | None,
    source: str,
) -> Example:
    sig = json.dumps(messages, sort_keys=True, ensure_ascii=False)
    return Example(
        case_id=f"{category}_{kind}_{stable_id(sig)}",
        category=category,
        kind=kind,
        messages=tuple(messages),
        expect_tool=expect_tool,
        source=source,
    )


def user_assistant(user: str, assistant: str, system: str | None = None) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    )
    return messages


def build_examples() -> list[Example]:
    examples: list[Example] = []
    systems = [
        None,
        "You are a helpful assistant with tools.",
        "Use available tools when the user's request requires the browser, terminal, or local files. Otherwise answer normally.",
    ]

    terminal_cases = [
        ("run pwd", "pwd"),
        ("pwd in terminal", "pwd"),
        ("show me the current directory from shell", "pwd"),
        ("execute echo hermes-tool-test", "echo hermes-tool-test"),
        ("run echo hermes-tool-test", "echo hermes-tool-test"),
        ("terminal: echo hermes-tool-test", "echo hermes-tool-test"),
        ("list files here from shell", "ls"),
        ("ls this directory", "ls"),
        ("run ls -la", "ls -la"),
        ("use terminal to show python version", "python3 --version"),
        ("what architecture is this machine? run uname -m", "uname -m"),
        ("print the current username in the terminal", "whoami"),
        ("in terminal, count files in this folder", "find . -maxdepth 1 -type f | wc -l"),
        ("shell out and show disk usage for this folder", "du -sh ."),
    ]
    terminal_prefixes = ["", "please ", "can you ", "real quick ", "i need you to "]
    for system in systems:
        for user, command in terminal_cases:
            for prefix in terminal_prefixes:
                prompt = prefix + user
                examples.append(
                    make_example(
                        "terminal",
                        "tool_call",
                        user_assistant(prompt, tool_call("terminal", command=command), system),
                        "terminal",
                        "contrast_terminal",
                    )
                )

    browser_cases = [
        ("what's the weather today in Austin TX", "https://www.google.com/search?q=Austin+TX+weather+today"),
        ("check online for Austin TX weather", "https://www.google.com/search?q=Austin+TX+weather"),
        ("look up the current forecast in Austin", "https://www.google.com/search?q=current+Austin+TX+forecast"),
        ("find the latest mlx-lm release notes online", "https://www.google.com/search?q=latest+mlx-lm+release+notes"),
        ("search the web for Liquid AI LFM2.5 context length", "https://www.google.com/search?q=Liquid+AI+LFM2.5+context+length"),
        ("open a search for NOAA Austin forecast", "https://www.google.com/search?q=NOAA+Austin+forecast"),
        ("check current news about llama.cpp LFM2.5 support", "https://www.google.com/search?q=llama.cpp+LFM2.5+support"),
        ("google the latest Hermes agent release", "https://www.google.com/search?q=latest+Hermes+agent+release"),
        ("pull up the weather.com page for Dallas weather", "https://www.google.com/search?q=weather.com+Dallas+weather"),
    ]
    browser_prefixes = ["", "please ", "can you ", "bro ", "hey "]
    for system in systems:
        for user, url in browser_cases:
            for prefix in browser_prefixes:
                examples.append(
                    make_example(
                        "browser",
                        "tool_call",
                        user_assistant(prefix + user, tool_call("browser_navigate", url=url), system),
                        "browser_navigate",
                        "contrast_browser",
                    )
                )

    file_cases = [
        ("read /tmp/hermes_eval_marker.txt", "read_file", {"path": "/tmp/hermes_eval_marker.txt"}),
        ("open the file /tmp/hermes_eval_marker.txt", "read_file", {"path": "/tmp/hermes_eval_marker.txt"}),
        ("show me the contents of /tmp/hermes_router_note.txt", "read_file", {"path": "/tmp/hermes_router_note.txt"}),
        ("search this repo for browser_navigate", "search_files", {"pattern": "browser_navigate", "path": "."}),
        ("find where hermes-tool-test appears locally", "search_files", {"pattern": "hermes-tool-test", "path": "."}),
        ("search my project for HERMES_PARSE_TEXT_TOOL_CALLS", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS", "path": "."}),
        ("write /tmp/hermes_router_note.txt with hello", "write_file", {"path": "/tmp/hermes_router_note.txt", "content": "hello"}),
        ("update /tmp/hermes_router_note.txt to say done", "write_file", {"path": "/tmp/hermes_router_note.txt", "content": "done"}),
        ("create /tmp/hermes_fixed_schema_probe.txt containing fixed schema", "write_file", {"path": "/tmp/hermes_fixed_schema_probe.txt", "content": "fixed schema"}),
    ]
    file_prefixes = ["", "please ", "can you ", "i need you to "]
    for system in systems:
        for user, name, kwargs in file_cases:
            for prefix in file_prefixes:
                examples.append(
                    make_example(
                        "file",
                        "tool_call",
                        user_assistant(prefix + user, tool_call(name, **kwargs), system),
                        name,
                        "contrast_file",
                    )
                )

    # Contrast pairs: nearly identical words, different correct tools.
    contrast_sets = [
        ("search this repo for browser_navigate", tool_call("search_files", pattern="browser_navigate", path="."), "search_files"),
        ("search the web for browser_navigate docs", tool_call("browser_navigate", url="https://www.google.com/search?q=browser_navigate+docs"), "browser_navigate"),
        ("run a search command for browser_navigate in this repo", tool_call("terminal", command="rg browser_navigate ."), "terminal"),
        ("open /tmp/hermes_eval_marker.txt", tool_call("read_file", path="/tmp/hermes_eval_marker.txt"), "read_file"),
        ("open a web search for /tmp/hermes_eval_marker.txt", tool_call("browser_navigate", url="https://www.google.com/search?q=%2Ftmp%2Fhermes_eval_marker.txt"), "browser_navigate"),
        ("write the word browser in /tmp/hermes_word.txt", tool_call("write_file", path="/tmp/hermes_word.txt", content="browser"), "write_file"),
        ("write a sentence using the word browser", "The browser opened a quiet page.", None),
        ("tell me what the terminal tool is for", "The terminal tool is for running shell commands when a task needs local command execution.", None),
        ("do not use tools; what command prints the current directory?", "The command is `pwd`.", None),
    ]
    for system in systems:
        for user, assistant, expect in contrast_sets:
            examples.append(
                make_example(
                    "contrast",
                    "tool_call" if expect else "no_tool",
                    user_assistant(user, assistant, system),
                    expect,
                    "minimal_pairs",
                )
            )

    no_tool_pairs = [
        ("hello!", "Hello!"),
        ("how are you", "I'm here and ready to help."),
        ("explain what a router does in a MoE model in one sentence", "A router chooses which expert networks should process each token."),
        ("what does tool calling mean conceptually?", "Tool calling means the model asks an external function or system to perform an action instead of only replying with text."),
        ("do not use tools, just say done", "done"),
        ("without using tools, tell me a joke about terminals", "A terminal walked into a shell and finally found its prompt."),
        ("what command would list files? do not run it", "The command is usually `ls`."),
        ("write a sentence using the word shell", "The shell waited for the next command."),
        ("what is a browser in one sentence?", "A browser is an application for navigating and interacting with web pages."),
        ("what should I use for current weather if tools are disabled?", "Use a weather website, weather app, or search engine."),
    ]
    for system in systems:
        for user, assistant in no_tool_pairs:
            examples.append(make_example("no_tool", "no_tool", user_assistant(user, assistant, system), None, "hard_negative"))

    finalization = [
        (
            "what's the weather today in Austin TX",
            "browser_navigate",
            {"url": "https://www.google.com/search?q=Austin+TX+weather+today"},
            "Austin is 82 F and sunny.",
            "Austin is 82 F and sunny.",
        ),
        (
            "run echo hermes-tool-test",
            "terminal",
            {"command": "echo hermes-tool-test"},
            "hermes-tool-test",
            "hermes-tool-test",
        ),
        (
            "read /tmp/hermes_eval_marker.txt",
            "read_file",
            {"path": "/tmp/hermes_eval_marker.txt"},
            "marker: fixed-schema",
            "marker: fixed-schema",
        ),
        (
            "search this repo for browser_navigate",
            "search_files",
            {"pattern": "browser_navigate", "path": "."},
            "release_work/hermes-agent-lfm-tool-parser/tools/browser_tool.py:1472: name browser_navigate",
            "I found browser_navigate in release_work/hermes-agent-lfm-tool-parser/tools/browser_tool.py.",
        ),
    ]
    for system in systems:
        for user, name, kwargs, result, assistant in finalization:
            messages: list[dict[str, Any]] = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.extend(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": tool_call(name, **kwargs)},
                    {"role": "tool", "name": name, "tool_call_id": f"call_{stable_id(user, name)}", "content": result},
                    {"role": "assistant", "content": assistant},
                ]
            )
            examples.append(make_example("finalization", "final_answer", messages, None, "tool_result_finalization"))

    corrections = [
        (
            "run echo hermes-tool-test",
            "browser_navigate",
            {"url": "https://www.google.com/search?q=echo+hermes-tool-test"},
            "That opened a browser search, but I meant run it locally.",
            "terminal",
            {"command": "echo hermes-tool-test"},
        ),
        (
            "search this repo for browser_navigate",
            "browser_navigate",
            {"url": "https://www.google.com/search?q=browser_navigate"},
            "No, search the local repo, not the web.",
            "search_files",
            {"pattern": "browser_navigate", "path": "."},
        ),
        (
            "pwd in terminal",
            "browser_navigate",
            {"url": "https://www.google.com/search?q=pwd+terminal"},
            "That did not run pwd. Use the terminal.",
            "terminal",
            {"command": "pwd"},
        ),
    ]
    for system in systems:
        for user, wrong_name, wrong_kwargs, correction, right_name, right_kwargs in corrections:
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.extend(
                [
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": tool_call(wrong_name, **wrong_kwargs)},
                    {"role": "tool", "name": wrong_name, "tool_call_id": f"wrong_{stable_id(user, wrong_name)}", "content": "The requested local action was not performed."},
                    {"role": "user", "content": correction},
                    {"role": "assistant", "content": tool_call(right_name, **right_kwargs)},
                ]
            )
            examples.append(make_example("correction", "tool_call", messages, right_name, "multi_turn_correction"))

    return examples


def validate_examples(examples: list[Example], tools: list[dict[str, Any]]) -> dict[str, Any]:
    required = required_args_by_tool(tools)
    tool_names = set(required)
    failures: list[str] = []
    counts = Counter()
    for example in examples:
        counts[(example.category, example.kind, example.expect_tool or "none")] += 1
        if example.expect_tool and example.expect_tool not in tool_names:
            failures.append(f"{example.case_id}: unknown expected tool {example.expect_tool}")
        last = example.messages[-1]
        if last["role"] != "assistant":
            failures.append(f"{example.case_id}: final message is not assistant")
            continue
        content = last.get("content") or ""
        if example.expect_tool:
            try:
                name, _ = validate_pythonic_call(content, required)
            except Exception as exc:
                failures.append(f"{example.case_id}: {exc}")
                continue
            if name != example.expect_tool:
                failures.append(f"{example.case_id}: expected {example.expect_tool}, got {name}")
        elif TOOL_MARKUP_RE.search(content):
            failures.append(f"{example.case_id}: no-tool/final answer contains tool markup")
    if failures:
        raise SystemExit("Dataset validation failed:\n" + "\n".join(failures[:50]))
    return {
        "total": len(examples),
        "by_category": Counter(example.category for example in examples),
        "by_expected_tool": Counter(example.expect_tool or "none" for example in examples),
        "by_source": Counter(example.source for example in examples),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=905)
    parser.add_argument("--train-size", type=int, default=4200)
    parser.add_argument("--valid-size", type=int, default=360)
    parser.add_argument("--test-size", type=int, default=360)
    args = parser.parse_args()

    tools_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = tools_payload["tools"]
    examples = build_examples()
    validation = validate_examples(examples, tools)

    rng = random.Random(args.seed)
    need = args.train_size + args.valid_size + args.test_size
    target_mix = {
        "terminal": 0.26,
        "file": 0.22,
        "browser": 0.18,
        "contrast": 0.10,
        "no_tool": 0.16,
        "finalization": 0.05,
        "correction": 0.03,
    }
    by_category: dict[str, list[Example]] = {}
    for example in examples:
        by_category.setdefault(example.category, []).append(example)
    expanded = []
    allocated = 0
    for category, fraction in target_mix.items():
        count = int(round(need * fraction))
        allocated += count
        pool = by_category[category]
        for i in range(count):
            source = pool[i % len(pool)]
            expanded.append(
                Example(
                    case_id=f"{source.case_id}_{i}",
                    category=source.category,
                    kind=source.kind,
                    messages=source.messages,
                    expect_tool=source.expect_tool,
                    source=source.source,
                )
            )
    while allocated < need:
        source = rng.choice(examples)
        expanded.append(
            Example(
                case_id=f"{source.case_id}_fill_{allocated}",
                category=source.category,
                kind=source.kind,
                messages=source.messages,
                expect_tool=source.expect_tool,
                source=source.source,
            )
        )
        allocated += 1
    expanded = expanded[:need]
    rng.shuffle(expanded)

    def row(example: Example) -> dict[str, Any]:
        return {
            "messages": list(example.messages),
            "tools": tools,
            "case_id": example.case_id,
            "category": example.category,
            "kind": example.kind,
            "expect_tool": example.expect_tool,
            "source": example.source,
        }

    train = [row(example) for example in expanded[: args.train_size]]
    valid = [row(example) for example in expanded[args.train_size : args.train_size + args.valid_size]]
    test = [row(example) for example in expanded[args.train_size + args.valid_size :]]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    write_jsonl(args.out / "test.jsonl", test)

    manifest = {
        "name": "iter05_fixed_hermes_contrast_router",
        "seed": args.seed,
        "source_tools": tools_payload.get("source"),
        "tool_names": tools_payload.get("tool_names"),
        "sizes": {"train": len(train), "valid": len(valid), "test": len(test)},
        "base_example_validation": {
            "total": validation["total"],
            "by_category": dict(validation["by_category"]),
            "by_expected_tool": dict(validation["by_expected_tool"]),
            "by_source": dict(validation["by_source"]),
        },
        "expanded_counts": {
            "train_by_category": dict(Counter(row["category"] for row in train)),
            "train_by_expected_tool": dict(Counter(row["expect_tool"] or "none" for row in train)),
            "valid_by_category": dict(Counter(row["category"] for row in valid)),
            "test_by_category": dict(Counter(row["category"] for row in test)),
        },
        "format": "MLX ChatDataset JSONL with messages plus exact Hermes tools; prompt masking required at training time.",
        "target_format": "Native LFM pythonic tool calls inside assistant content.",
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest["sizes"] | {"train_by_category": manifest["expanded_counts"]["train_by_category"]}, indent=2))


if __name__ == "__main__":
    main()
