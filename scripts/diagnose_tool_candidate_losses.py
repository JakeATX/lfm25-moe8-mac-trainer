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


def build_cases() -> list[dict[str, Any]]:
    return [
        {
            "case_id": "weather_austin",
            "prompt": "what's the weather in Austin today",
            "candidates": [
                ("browser_correct", assistant_tool("browser_navigate", {"url": google("Austin weather today")})),
                ("x_wrong", assistant_tool("x_search", {"query": "Austin weather today"})),
                ("refusal", assistant_text("I don't have access to real-time weather data.")),
                ("chat", assistant_text("Austin weather can vary, but I can explain how to check it.")),
            ],
        },
        {
            "case_id": "latest_lfm",
            "prompt": "look up the latest LiquidAI LFM2.5 GGUF info",
            "candidates": [
                ("browser_correct", assistant_tool("browser_navigate", {"url": google("latest LiquidAI LFM2.5 GGUF")})),
                ("x_wrong", assistant_tool("x_search", {"query": "latest LiquidAI LFM2.5 GGUF"})),
                ("refusal", assistant_text("I can't browse the web from here.")),
                ("chat", assistant_text("LFM2.5 is a Liquid AI model family with MoE-style architecture.")),
            ],
        },
        {
            "case_id": "terminal_pwd",
            "prompt": "show current directory with a shell command",
            "candidates": [
                ("terminal_correct", assistant_tool("terminal", {"command": "pwd"})),
                ("browser_wrong", assistant_tool("browser_navigate", {"url": google("current directory shell command")})),
                ("refusal", assistant_text("I cannot run local commands.")),
                ("chat", assistant_text("The `pwd` command prints the current directory.")),
            ],
        },
        {
            "case_id": "search_repo",
            "prompt": "search this repo for HERMES_PARSE_TEXT_TOOL_CALLS",
            "candidates": [
                ("search_files_correct", assistant_tool("search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"})),
                ("terminal_ok", assistant_tool("terminal", {"command": "rg HERMES_PARSE_TEXT_TOOL_CALLS ."})),
                ("browser_wrong", assistant_tool("browser_navigate", {"url": google("HERMES_PARSE_TEXT_TOOL_CALLS")})),
                ("chat", assistant_text("That string looks like an environment variable name.")),
            ],
        },
        {
            "case_id": "computer_capture",
            "prompt": "use computer use to inspect the screen",
            "candidates": [
                ("computer_correct", assistant_tool("computer_use", {"action": "capture", "mode": "som"})),
                ("computer_bad_action", assistant_tool("computer_use", {"action": "navigate"})),
                ("browser_wrong", assistant_tool("browser_navigate", {"url": google("inspect screen")})),
                ("chat", assistant_text("I can describe how screen inspection usually works.")),
            ],
        },
        {
            "case_id": "normal_no_tool",
            "prompt": "do not use tools; what command prints the current folder?",
            "candidates": [
                ("chat_correct", assistant_text("`pwd` prints the current folder.")),
                ("terminal_wrong", assistant_tool("terminal", {"command": "pwd"})),
                ("browser_wrong", assistant_tool("browser_navigate", {"url": google("command prints current folder")})),
                ("refusal", assistant_text("I can't help with that.")),
            ],
        },
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    tool_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = [tool for tool in tool_payload["tools"] if tool.get("function", {}).get("name") in PREFERRED_TOOLS]
    model, tokenizer = load(args.model, lazy=True)
    rows: list[dict[str, Any]] = []
    for case in build_cases():
        case_rows: list[dict[str, Any]] = []
        for label, assistant in case["candidates"]:
            ids, start, end, text = render(tokenizer, tools, case["prompt"], assistant)
            if len(ids) > args.max_tokens:
                case_rows.append({"candidate": label, "error": "too_long", "tokens": len(ids)})
                continue
            loss, ntoks = candidate_loss(model, ids, start, end)
            case_rows.append(
                {
                    "candidate": label,
                    "loss": loss,
                    "supervised_tokens": ntoks,
                    "total_tokens": len(ids),
                    "render_contains_xml": "<tool_call>" in text or "</tool_call>" in text,
                }
            )
        valid = [row for row in case_rows if "loss" in row]
        best = min(valid, key=lambda row: row["loss"])["candidate"] if valid else None
        rows.append({"case_id": case["case_id"], "prompt": case["prompt"], "best_candidate": best, "candidates": case_rows})

    summary = {
        "model": args.model,
        "tools_json": str(args.tools_json),
        "cases": len(rows),
        "best_candidate_counts": {},
    }
    for row in rows:
        summary["best_candidate_counts"][row["best_candidate"]] = summary["best_candidate_counts"].get(row["best_candidate"], 0) + 1
    payload = {"summary": summary, "results": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
