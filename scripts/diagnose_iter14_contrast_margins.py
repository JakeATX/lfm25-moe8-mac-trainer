#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mlx.core as mx
from mlx_lm import load
from mlx_lm.tuner.trainer import default_loss


SYSTEM = (
    "You are Hermes. Use the provided tools when they are needed to satisfy the user. "
    "If no tool is needed, answer normally. Do not invent tool names or arguments."
)

PREFERRED_TOOLS = {
    "browser_navigate",
    "x_search",
    "terminal",
    "search_files",
    "read_file",
    "write_file",
    "patch",
    "execute_code",
    "computer_use",
}


def google(query: str) -> str:
    return "https://www.google.com/search?q=" + query.replace(" ", "+")


def assistant_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_diag",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def assistant_text(text: str) -> dict[str, Any]:
    return {"role": "assistant", "content": text}


def candidates(prompt: str, correct_label: str, browser_query: str, x_query: str, file_pattern: str, chat: str) -> dict[str, Any]:
    return {
        "prompt": prompt,
        "correct": correct_label,
        "candidates": [
            ("browser", assistant_tool("browser_navigate", {"url": google(browser_query)})),
            ("x_search", assistant_tool("x_search", {"query": x_query})),
            ("search_files", assistant_tool("search_files", {"pattern": file_pattern})),
            ("refusal", assistant_text("I don't have access to that information.")),
            ("chat", assistant_text(chat)),
        ],
    }


def build_cases() -> list[dict[str, Any]]:
    return [
        candidates("search for LFM2.5 GGUF", "browser", "LFM2.5 GGUF", "LFM2.5 GGUF", "LFM2.5 GGUF", "LFM2.5 GGUF is a model artifact format reference."),
        candidates("search X for LFM2.5 GGUF", "x_search", "LFM2.5 GGUF", "LFM2.5 GGUF", "LFM2.5 GGUF", "You asked specifically about X posts."),
        candidates("search this repo for LFM2.5 GGUF", "search_files", "LFM2.5 GGUF", "LFM2.5 GGUF", "LFM2.5 GGUF", "That phrase may appear in local docs."),
        candidates("explain what LFM2.5 GGUF means; do not search", "chat", "LFM2.5 GGUF", "LFM2.5 GGUF", "LFM2.5 GGUF", "LFM2.5 GGUF refers to an LFM2.5 model checkpoint packaged in GGUF format."),
        candidates("what's the weather in Austin today", "browser", "Austin weather today", "Austin weather today", "Austin weather", "Weather changes over time, so current data is needed."),
        candidates("find X reactions to today's Austin weather", "x_search", "Austin weather today reactions", "Austin weather", "Austin weather", "You asked for reactions on X."),
        candidates("search local files, not online, for browser_navigate", "search_files", "browser_navigate", "browser_navigate", "browser_navigate", "`browser_navigate` is a tool name."),
        candidates("what is a browser in one sentence?", "chat", "browser definition", "browser", "browser", "A browser is an application for viewing and interacting with websites."),
        candidates("find the latest llama.cpp release notes", "browser", "latest llama.cpp release notes", "llama.cpp release notes", "llama.cpp release notes", "Release notes are current web information."),
        candidates("search Twitter/X for MLX LFM2.5", "x_search", "MLX LFM2.5", "MLX LFM2.5", "MLX LFM2.5", "The prompt explicitly asks for Twitter/X."),
        candidates("search my repo for tool_parser_type", "search_files", "tool_parser_type", "tool_parser_type", "tool_parser_type", "That is a local configuration key."),
        candidates("do not use tools; what command prints the current folder?", "chat", "command current folder", "current folder command", "pwd", "`pwd` prints the current folder."),
    ]


def render(tokenizer, tools: list[dict[str, Any]], prompt: str, assistant: dict[str, Any]) -> tuple[list[int], int, int, str]:
    prefix_messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
    full_messages = [*prefix_messages, assistant]
    prompt_text = tokenizer.apply_chat_template(prefix_messages, tools=tools, tokenize=False, add_generation_prompt=True)
    full_text = tokenizer.apply_chat_template(full_messages, tools=tools, tokenize=False, add_generation_prompt=False)
    prompt_ids = tokenizer.encode(prompt_text)
    full_ids = tokenizer.encode(full_text)
    return full_ids, len(prompt_ids), len(full_ids), full_text


def candidate_loss(model, ids: list[int], start: int, end: int) -> tuple[float, int]:
    batch = mx.array([ids])
    lengths = mx.array([[start, end]])
    loss, ntoks = default_loss(model, batch, lengths)
    mx.eval(loss, ntoks)
    return float(loss.item()), int(ntoks.item())


def main() -> None:
    parser = argparse.ArgumentParser(description="Score Iter14 correct-vs-wrong candidate losses.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--max-tokens", type=int, default=8192)
    args = parser.parse_args()

    tool_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = [tool for tool in tool_payload["tools"] if tool.get("function", {}).get("name") in PREFERRED_TOOLS]
    if args.adapter_path:
        model, tokenizer = load(args.model, adapter_path=args.adapter_path, lazy=True)
    else:
        model, tokenizer = load(args.model, lazy=True)

    results: list[dict[str, Any]] = []
    for case in build_cases():
        rows: list[dict[str, Any]] = []
        for label, assistant in case["candidates"]:
            ids, start, end, text = render(tokenizer, tools, case["prompt"], assistant)
            if len(ids) > args.max_tokens:
                rows.append({"candidate": label, "error": "too_long", "tokens": len(ids)})
                continue
            loss, ntoks = candidate_loss(model, ids, start, end)
            rows.append(
                {
                    "candidate": label,
                    "loss": loss,
                    "supervised_tokens": ntoks,
                    "total_tokens": len(ids),
                    "render_contains_xml": "<tool_call>" in text or "</tool_call>" in text,
                }
            )
        valid = [row for row in rows if "loss" in row]
        correct = next((row for row in valid if row["candidate"] == case["correct"]), None)
        wrong = [row for row in valid if row["candidate"] != case["correct"]]
        best = min(valid, key=lambda row: row["loss"]) if valid else None
        best_wrong = min(wrong, key=lambda row: row["loss"]) if wrong else None
        margin = (best_wrong["loss"] - correct["loss"]) if correct and best_wrong else None
        results.append(
            {
                "prompt": case["prompt"],
                "correct": case["correct"],
                "best_candidate": best["candidate"] if best else None,
                "correct_is_best": bool(best and best["candidate"] == case["correct"]),
                "margin_best_wrong_minus_correct": margin,
                "candidates": rows,
            }
        )

    margins = [row["margin_best_wrong_minus_correct"] for row in results if row["margin_best_wrong_minus_correct"] is not None]
    summary = {
        "model": args.model,
        "adapter_path": args.adapter_path,
        "tools_json": str(args.tools_json),
        "cases": len(results),
        "correct_best": sum(1 for row in results if row["correct_is_best"]),
        "correct_best_rate": round(sum(1 for row in results if row["correct_is_best"]) / len(results), 4) if results else 0,
        "mean_margin_best_wrong_minus_correct": round(sum(margins) / len(margins), 6) if margins else None,
        "positive_margin_cases": sum(1 for margin in margins if margin > 0),
        "positive_margin_rate": round(sum(1 for margin in margins if margin > 0) / len(margins), 4) if margins else 0,
        "best_candidate_counts": {},
    }
    for row in results:
        summary["best_candidate_counts"][row["best_candidate"]] = summary["best_candidate_counts"].get(row["best_candidate"], 0) + 1

    payload = {"summary": summary, "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
