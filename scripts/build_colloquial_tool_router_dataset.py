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


TOOL_SPECS = {
    "browser_navigate": ("url",),
    "terminal": ("command",),
    "read_file": ("path",),
    "write_file": ("path", "content"),
    "search_files": ("pattern",),
    "calculator": ("expression",),
}

TOOL_JSON = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "browser_navigate",
                "description": "Open a URL or web search in the browser.",
                "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminal",
                "description": "Run a shell command in a terminal.",
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
                "name": "read_file",
                "description": "Read a local text file.",
                "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
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
                    "properties": {"pattern": {"type": "string"}, "path": {"type": "string"}},
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
    ],
    separators=(",", ": "),
)

SYSTEM_PROMPTS = [
    (
        "You are a Hermes-style assistant with tools. Infer when the user needs a tool, "
        "even if they ask casually. When using a tool, output exactly one native LFM "
        "pythonic tool call and no prose: "
        "<|tool_call_start|>[tool_name(arg=\"value\")]<|tool_call_end|>. "
        "When no tool is needed, answer normally and do not emit tool syntax. "
        "Available tools: browser_navigate(url), terminal(command, workdir optional), "
        "read_file(path), write_file(path, content), search_files(pattern, path optional), "
        "calculator(expression).\nList of tools: "
        + TOOL_JSON
    ),
    (
        "Use tools when the request depends on the browser, shell, local files, search, "
        "or exact arithmetic. Return native LFM pythonic tool syntax only for tool calls. "
        "Do not describe the call in text. If the user explicitly says not to use tools, "
        "answer without a tool.\nList of tools: "
        + TOOL_JSON
    ),
    (
        "You are an agent. Choose the best available tool from the user's intent. "
        "Tool calls must be emitted as <|tool_call_start|>[name(arg=\"value\")]"
        "<|tool_call_end|>. Natural answers must not contain tool-call markup.\nList of tools: "
        + TOOL_JSON
    ),
    (
        "You are a helpful assistant with tools.\nList of tools: "
        + TOOL_JSON
    ),
]


@dataclass(frozen=True)
class Example:
    case_id: str
    category: str
    kind: str
    text: str
    tool: str | None = None
    source: str = "synthetic"


def stable_id(*parts: str) -> str:
    raw = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:14]


def quote(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def tool_call(name: str, **kwargs: str) -> str:
    args = ", ".join(f"{key}={quote(value)}" for key, value in kwargs.items())
    return f"<|tool_call_start|>[{name}({args})]<|tool_call_end|>"


def chatml(system: str, turns: list[tuple[str, str]]) -> str:
    out = ["<|startoftext|>", f"<|im_start|>system\n{system}<|im_end|>\n"]
    for role, content in turns:
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    return "".join(out)


def make_example(
    category: str,
    kind: str,
    system: str,
    turns: list[tuple[str, str]],
    tool: str | None = None,
    source: str = "synthetic",
) -> Example:
    user_bits = "|".join(content for role, content in turns if role == "user")
    case_id = f"{category}_{kind}_{stable_id(system, user_bits, turns[-1][1])}"
    return Example(case_id, category, kind, chatml(system, turns), tool, source)


def validate_pythonic_call(target: str) -> tuple[str, dict[str, Any]]:
    match = re.fullmatch(r"<\|tool_call_start\|>\[(.+)\]<\|tool_call_end\|>", target.strip())
    if not match:
        raise ValueError(f"not native LFM tool syntax: {target}")
    node = ast.parse(match.group(1), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"not a pythonic call: {target}")
    tool = node.func.id
    if tool not in TOOL_SPECS:
        raise ValueError(f"unknown tool {tool}: {target}")
    kwargs: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError(f"star kwargs are not allowed: {target}")
        kwargs[keyword.arg] = ast.literal_eval(keyword.value)
    missing = [name for name in TOOL_SPECS[tool] if name not in kwargs]
    if missing:
        raise ValueError(f"missing args {missing} for {tool}: {target}")
    return tool, kwargs


def build_base_examples() -> list[Example]:
    examples: list[Example] = []
    systems = SYSTEM_PROMPTS

    browser_intents = [
        ("what's the weather today in Austin TX", "https://www.google.com/search?q=Austin+TX+weather"),
        ("can you check online for the weather in Austin", "https://www.google.com/search?q=Austin+weather"),
        ("look up today's Austin forecast", "https://www.google.com/search?q=Austin+TX+forecast+today"),
        ("find the latest Liquid AI LFM 2.5 docs", "https://www.google.com/search?q=Liquid+AI+LFM+2.5+documentation"),
        ("open a search for mlx-lm tool parser pythonic", "https://www.google.com/search?q=mlx-lm+tool+parser+pythonic"),
        ("check the web for current Apple Silicon MLX release notes", "https://www.google.com/search?q=Apple+Silicon+MLX+release+notes"),
        ("google the NOAA forecast for Dallas", "https://www.google.com/search?q=NOAA+Dallas+forecast"),
        ("pull up the weather.com page for Chicago weather", "https://www.google.com/search?q=weather.com+Chicago+weather"),
    ]
    browser_prefixes = ["", "please ", "real quick, ", "bro ", "hey, "]
    for system in systems:
        for prefix in browser_prefixes:
            for user, url in browser_intents:
                prompt = prefix + user
                examples.append(make_example("browser", "tool_call", system, [("user", prompt), ("assistant", tool_call("browser_navigate", url=url))], "browser_navigate"))

    terminal_intents = [
        ("run pwd", "pwd"),
        ("pwd in the terminal", "pwd"),
        ("show me the current directory in shell", "pwd"),
        ("execute echo hermes-tool-test", "echo hermes-tool-test"),
        ("run ls -la", "ls -la"),
        ("list files here from terminal", "ls"),
        ("use shell to print hello", "echo hello"),
        ("terminal: uname -m", "uname -m"),
        ("in terminal, show python version", "python3 --version"),
        ("run a command to count files in this directory", "find . -maxdepth 1 -type f | wc -l"),
    ]
    for system in systems:
        for user, command in terminal_intents:
            variants = [user, f"can you {user}", f"please {user}", f"i need you to {user}"]
            for prompt in variants:
                examples.append(make_example("terminal", "tool_call", system, [("user", prompt), ("assistant", tool_call("terminal", command=command))], "terminal"))

    file_intents = [
        ("read /tmp/hermes_eval_marker.txt", "read_file", {"path": "/tmp/hermes_eval_marker.txt"}),
        ("open the file /Users/jkooker/README.md", "read_file", {"path": "/Users/jkooker/README.md"}),
        ("search this repo for hermes-tool-test", "search_files", {"pattern": "hermes-tool-test", "path": "."}),
        ("find where browser_navigate appears in the project", "search_files", {"pattern": "browser_navigate", "path": "."}),
        ("write /tmp/hermes_router_note.txt with the text hello", "write_file", {"path": "/tmp/hermes_router_note.txt", "content": "hello"}),
        ("update /tmp/hermes_router_note.txt to say done", "write_file", {"path": "/tmp/hermes_router_note.txt", "content": "done"}),
    ]
    for system in systems:
        for user, name, kwargs in file_intents:
            for prefix in ["", "please ", "can you "]:
                examples.append(make_example("file", "tool_call", system, [("user", prefix + user), ("assistant", tool_call(name, **kwargs))], name))

    calculator_intents = [
        ("what is 18.5 times 42 exactly", "18.5 * 42"),
        ("compute (23 + 19) * 2", "(23 + 19) * 2"),
        ("how much is 144 divided by 12", "144 / 12"),
        ("use a calculator for 999 - 123", "999 - 123"),
        ("calculate 3.14159 * 2", "3.14159 * 2"),
        ("exact math: 2**10", "2**10"),
    ]
    for system in systems:
        for user, expr in calculator_intents:
            for prefix in ["", "please ", "quickly "]:
                examples.append(make_example("calculator", "tool_call", system, [("user", prefix + user), ("assistant", tool_call("calculator", expression=expr))], "calculator"))

    no_tool_pairs = [
        ("hello!", "Hello!"),
        ("how are you", "I'm here and ready to help."),
        ("explain what a router does in a MoE model in one sentence", "A router chooses which expert networks should process each token."),
        ("what does terminal mean conceptually, no tools", "A terminal is a text interface for running commands."),
        ("do not use tools; just say done", "done"),
        ("without using tools, tell me what color the sky usually is", "The sky is usually blue on a clear day."),
        ("write a short sentence containing the word browser", "The browser opened a useful page."),
        ("what is a file search tool used for", "A file search tool finds matching text or filenames in local files."),
        ("say the word calculator twice", "calculator calculator"),
        ("I might ask you to search later, but for now just say ready", "ready"),
    ]
    for system in systems:
        for user, assistant in no_tool_pairs:
            examples.append(make_example("no_tool", "no_tool", system, [("user", user), ("assistant", assistant)]))

    finalizations = [
        ("browser_navigate", {"url": "https://www.google.com/search?q=Austin+TX+weather"}, "Search result: Austin weather is 82 F and sunny.", "Austin weather is 82 F and sunny."),
        ("terminal", {"command": "echo hermes-tool-test"}, "hermes-tool-test", "The terminal output is hermes-tool-test."),
        ("terminal", {"command": "pwd"}, "/Users/jkooker/Documents/Codex", "The current directory is /Users/jkooker/Documents/Codex."),
        ("read_file", {"path": "/tmp/hermes_eval_marker.txt"}, "marker contents: lfm router ok", "The file says: lfm router ok."),
        ("search_files", {"pattern": "hermes-tool-test", "path": "."}, "scripts/eval_openai_tool_calls.py: terminal echo hermes-tool-test", "I found hermes-tool-test in scripts/eval_openai_tool_calls.py."),
        ("calculator", {"expression": "18.5 * 42"}, "777", "The calculator result is 777."),
    ]
    for system in systems:
        for name, kwargs, tool_response, assistant in finalizations:
            call = tool_call(name, **kwargs)
            turns = [
                ("user", "Use the appropriate tool, then report the result."),
                ("assistant", call),
                ("tool", tool_response),
                ("assistant", assistant),
            ]
            examples.append(make_example("finalization", "final_answer", system, turns, name, "synthetic_finalization"))

    observed_failures = [
        ("whats the weather today", "browser_navigate", {"url": "https://www.google.com/search?q=weather+today"}),
        ("search for austin tx", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+TX+weather"}),
        ("yes you do what tools do you have that you could use to answer the request", None, None),
        ("run echo hermes-tool-test", "terminal", {"command": "echo hermes-tool-test"}),
        ("that didnt work, try the browser for Austin weather", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+TX+weather"}),
    ]
    for system in systems:
        for user, name, kwargs in observed_failures:
            if name is None:
                answer = "I can use the browser for current weather, terminal for commands, and file tools for local files."
                examples.append(make_example("observed_failure", "no_tool", system, [("user", user), ("assistant", answer)], source="observed_failure"))
            else:
                examples.append(make_example("observed_failure", "tool_call", system, [("user", user), ("assistant", tool_call(name, **kwargs))], name, "observed_failure"))

    return examples


def expand_examples(base: list[Example], target_train_rows: int, seed: int) -> list[Example]:
    rng = random.Random(seed)
    by_category: dict[str, list[Example]] = {}
    for example in base:
        by_category.setdefault(example.category, []).append(example)
    category_mix = [
        ("browser", 0.20),
        ("terminal", 0.20),
        ("file", 0.13),
        ("calculator", 0.09),
        ("no_tool", 0.21),
        ("finalization", 0.08),
        ("observed_failure", 0.09),
    ]
    missing = [category for category, _ in category_mix if category not in by_category]
    if missing:
        raise ValueError(f"missing categories in base examples: {missing}")
    expanded: list[Example] = []
    categories = [category for category, _ in category_mix]
    weights = [weight for _, weight in category_mix]
    while len(expanded) < target_train_rows + max(300, target_train_rows // 6):
        category = rng.choices(categories, weights=weights, k=1)[0]
        example = rng.choice(by_category[category])
        suffix = len(expanded)
        expanded.append(
            Example(
                case_id=f"{example.case_id}_{suffix:05d}",
                category=example.category,
                kind=example.kind,
                text=example.text,
                tool=example.tool,
                source=example.source,
            )
        )
    rng.shuffle(expanded)
    return expanded


def split_examples(examples: list[Example], seed: int) -> dict[str, list[Example]]:
    keyed = []
    for example in examples:
        digest = hashlib.sha1(f"{seed}:{example.case_id}".encode("utf-8")).hexdigest()
        keyed.append((digest, example))
    keyed.sort(key=lambda item: item[0])
    ordered = [example for _, example in keyed]
    train_cut = int(len(ordered) * 0.86)
    valid_cut = int(len(ordered) * 0.93)
    return {
        "train": ordered[:train_cut],
        "valid": ordered[train_cut:valid_cut],
        "test": ordered[valid_cut:],
    }


def validate_examples(examples: list[Example]) -> None:
    for example in examples:
        assistant_chunks = re.findall(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", example.text, re.S)
        if not assistant_chunks:
            raise ValueError(f"missing assistant turn: {example.case_id}")
        final_assistant = assistant_chunks[-1].strip()
        if example.kind == "tool_call":
            tool, _ = validate_pythonic_call(final_assistant)
            if tool != example.tool:
                raise ValueError(f"tool mismatch {tool} != {example.tool}: {example.case_id}")
        elif example.kind in {"no_tool", "final_answer"}:
            if "<|tool_call_start|>" in final_assistant or "<tool_call>" in final_assistant:
                raise ValueError(f"unexpected final text tool syntax: {example.case_id}")
        else:
            raise ValueError(f"unknown kind: {example.kind}")


def write_split(path: Path, rows: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps({"text": row.text}, ensure_ascii=False) + "\n")


def write_metadata(path: Path, rows: list[Example]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.__dict__, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--target-train-rows", type=int, default=2600)
    args = parser.parse_args()

    base = build_base_examples()
    expanded = expand_examples(base, args.target_train_rows, args.seed)
    splits = split_examples(expanded, args.seed)
    validate_examples([row for rows in splits.values() for row in rows])

    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        write_split(args.out / f"{split}.jsonl", rows)
        write_metadata(args.out / f"{split}_metadata.jsonl", rows)

    all_rows = [row for rows in splits.values() for row in rows]
    category_counts = Counter(row.category for row in all_rows)
    kind_counts = Counter(row.kind for row in all_rows)
    tool_counts = Counter(row.tool for row in all_rows if row.tool)
    control_rows = category_counts["no_tool"] + category_counts["finalization"]
    manifest = {
        "name": "iter02_colloquial_router",
        "format": "mlx_lm text jsonl",
        "target_tool_format": "native_lfm_pythonic",
        "tool_parser_type_required": "pythonic",
        "seed": args.seed,
        "base_examples": len(base),
        "splits": {split: len(rows) for split, rows in splits.items()},
        "category_counts": dict(sorted(category_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "no_tool_plus_finalization_fraction": round(control_rows / len(all_rows), 4),
        "validation": {
            "all_tool_targets_parse": True,
            "no_tool_final_turns_have_no_tool_syntax": True,
            "known_tools": sorted(TOOL_SPECS),
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
