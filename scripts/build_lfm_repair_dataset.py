#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


TOOLS = [
    "browser_navigate(url)",
    "calculator(expression)",
    "terminal(command, workdir optional)",
    "read_file(path)",
    "search_files(pattern, path optional)",
]


def chatml(system: str, user: str, assistant: str) -> str:
    return (
        "<|startoftext|>"
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
    )


def system_prompt() -> str:
    return (
        "You are a Hermes-style tool-using assistant. Use a tool only when it is needed. "
        "When using a tool, output exactly one native LFM tool call and no extra prose: "
        "<|tool_call_start|>[tool_name(arg=\"value\")]<|tool_call_end|>. "
        "When no tool is needed, answer naturally and do not emit tool syntax. "
        f"Available tools: {', '.join(TOOLS)}."
    )


def rows() -> list[dict]:
    sys = system_prompt()
    data = []
    browser_queries = [
        ("Search for Austin TX weather using the browser.", "https://www.google.com/search?q=Austin+TX+weather"),
        ("Open a search for Liquid AI LFM2.5 release notes.", "https://www.google.com/search?q=Liquid+AI+LFM2.5+release+notes"),
        ("Use the browser to find current weather in Chicago.", "https://www.google.com/search?q=Chicago+weather"),
        ("Navigate to the Python docs.", "https://docs.python.org/3/"),
        ("Look up the latest MLX documentation with the browser.", "https://www.google.com/search?q=MLX+documentation"),
    ]
    for user, url in browser_queries:
        data.append({"text": chatml(sys, user, f'<|tool_call_start|>[browser_navigate(url="{url}")]<|tool_call_end|>'), "kind": "tool_call", "tool": "browser_navigate"})
    calcs = [
        ("Use the calculator tool for 18.5 * 42.", "18.5 * 42"),
        ("Calculate 144 / 12 with the calculator.", "144 / 12"),
        ("Use calculator for (23 + 19) * 2.", "(23 + 19) * 2"),
        ("Please call the calculator for 9.8 * 11.", "9.8 * 11"),
    ]
    for user, expr in calcs:
        data.append({"text": chatml(sys, user, f'<|tool_call_start|>[calculator(expression="{expr}")]<|tool_call_end|>'), "kind": "tool_call", "tool": "calculator"})
    terminals = [
        ("Use the terminal tool to run: echo hermes-tool-test", "echo hermes-tool-test"),
        ("Run pwd in the terminal.", "pwd"),
        ("Use terminal to list the current directory.", "ls"),
    ]
    for user, command in terminals:
        data.append({"text": chatml(sys, user, f'<|tool_call_start|>[terminal(command="{command}")]<|tool_call_end|>'), "kind": "tool_call", "tool": "terminal"})
    finalizations = [
        ("The calculator returned 777.", "The result is 777."),
        ("The browser result says Austin is sunny and 82 F.", "Austin is sunny and 82 F."),
    ]
    for user, assistant in finalizations:
        text = (
            "<|startoftext|>"
            f"<|im_start|>system\n{sys}<|im_end|>\n"
            f"<|im_start|>user\nReport the tool result.<|im_end|>\n"
            f"<|im_start|>tool\n<tool_response>{user}</tool_response><|im_end|>\n"
            f"<|im_start|>assistant\n{assistant}<|im_end|>\n"
        )
        data.append({"text": text, "kind": "final_answer"})
    no_tools = [
        ("Say hello in one short sentence.", "Hello!"),
        ("Give me a one-sentence fun fact about Austin.", "Austin is home to the largest urban bat colony in North America."),
        ("What color is the sky on a clear day?", "The sky is usually blue on a clear day."),
        ("Write three comma-separated colors.", "red, blue, green"),
        ("Do not use a tool; just say done.", "done"),
    ]
    for user, assistant in no_tools:
        data.append({"text": chatml(sys, user, assistant), "kind": "no_tool"})
    return data


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--copies", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    base = rows()
    expanded = []
    for _ in range(args.copies):
        sample = list(base)
        rng.shuffle(sample)
        expanded.extend(sample)
    args.out.mkdir(parents=True, exist_ok=True)
    train_cut = int(len(expanded) * 0.86)
    valid_cut = int(len(expanded) * 0.93)
    splits = {
        "train": expanded[:train_cut],
        "valid": expanded[train_cut:valid_cut],
        "test": expanded[valid_cut:],
    }
    for split, split_rows in splits.items():
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in split_rows:
                f.write(json.dumps({"text": row["text"]}, ensure_ascii=False) + "\n")
    manifest = {
        "format": "mlx_lm text jsonl",
        "target_tool_format": "native_lfm",
        "base_templates": len(base),
        "copies": args.copies,
        "splits": {k: len(v) for k, v in splits.items()},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
