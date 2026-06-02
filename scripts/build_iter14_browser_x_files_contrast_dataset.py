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
    load_tools,
    stable_id,
    text_row,
    tool_row,
    tool_subset,
    validate_render,
    valid_args,
    write_jsonl,
)


CORE_TOOLS = [
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


def google(query: str) -> str:
    return "https://www.google.com/search?q=" + query.replace(" ", "+")


def expected_tools(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return tool_subset(by_name, CORE_TOOLS)


def add_tool(
    rows: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    prompt: str,
    name: str,
    args: dict[str, Any],
    category: str,
    source: str,
    salt: str,
) -> None:
    if name not in by_name or not valid_args(by_name, name, args):
        return
    row_id = f"iter14_{category}_{stable_id(prompt, name, json.dumps(args, sort_keys=True), salt)}"
    rows.append(tool_row(row_id, prompt, name, args, expected_tools(by_name), category, source))


def add_text(
    rows: list[dict[str, Any]],
    by_name: dict[str, dict[str, Any]],
    prompt: str,
    answer: str,
    category: str,
    source: str,
    salt: str,
) -> None:
    row_id = f"iter14_{category}_{stable_id(prompt, answer, salt)}"
    rows.append(text_row(row_id, prompt, answer, expected_tools(by_name), category, source))


def browser_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    query_cases = [
        ("search for LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("look up LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("search the web for LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("find current LFM2.5 GGUF information", "current LFM2.5 GGUF information"),
        ("what's the weather in Austin today", "Austin weather today"),
        ("check the current weather in Austin TX", "current weather Austin TX"),
        ("weather in Austin right now", "Austin weather right now"),
        ("find the latest NOAA forecast for Austin", "NOAA Austin forecast today"),
        ("what is happening in Austin today, check online", "Austin today current news"),
        ("look up the latest LiquidAI LFM2.5 information", "latest LiquidAI LFM2.5 information"),
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
        ("search the internet for a recent llama.cpp issue about tool calls", "recent llama.cpp tool calls issue"),
        ("look up current Hugging Face model card guidance", "current Hugging Face model card guidance"),
        ("find today's forecast for Chicago", "Chicago weather today"),
        ("can you check online for the latest MLX release", "latest MLX release"),
        ("search current web results for Hermes agent tools", "Hermes agent tools current"),
        ("find latest docs for llama-server OpenAI tools", "llama-server OpenAI tools latest docs"),
    ]
    open_cases = [
        ("open x.com in the browser", "https://x.com"),
        ("navigate the browser to google.com", "https://www.google.com"),
        ("open the MLX documentation website", "https://ml-explore.github.io"),
        ("open Hugging Face", "https://huggingface.co"),
        ("open the llama.cpp GitHub page", "https://github.com/ggml-org/llama.cpp"),
        ("open the Liquid AI website", "https://www.liquid.ai"),
        ("navigate to https://github.com/JakeATX", "https://github.com/JakeATX"),
        ("open https://huggingface.co/sjakek", "https://huggingface.co/sjakek"),
    ]
    for repeat in range(24):
        for prompt, query in query_cases:
            add_tool(rows, by_name, prompt, "browser_navigate", {"url": google(query)}, "browser_general_web", "iter14_browser", str(repeat))
        for prompt, url in open_cases:
            add_tool(rows, by_name, prompt, "browser_navigate", {"url": url}, "browser_general_web", "iter14_browser_open", str(repeat))
    return rows


def x_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("search X for LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("look on X for recent posts about LFM2.5", "recent posts about LFM2.5"),
        ("check X for recent Liquid AI posts", "Liquid AI"),
        ("search Twitter/X for MLX LFM2.5", "MLX LFM2.5"),
        ("find current X reactions to llama.cpp", "llama.cpp reactions"),
        ("look up what people on X are saying about Hermes agent", "Hermes agent"),
        ("search X for sjakek model posts", "sjakek model"),
        ("check Twitter for current Apple MLX discussion", "Apple MLX discussion"),
        ("find X threads about Unsloth dynamic GGUF quants", "Unsloth dynamic GGUF quants"),
        ("search X posts from Liquid AI about LFM2.5", "LFM2.5 Liquid AI"),
        ("look for recent X chatter about tool calling", "tool calling"),
        ("search Twitter for current discussion of Apollo 13", "Apollo 13"),
        ("find X reactions to today's Austin weather", "Austin weather"),
        ("search X for recent llama.cpp release discussion", "llama.cpp release"),
        ("check X for posts about Hugging Face GGUF uploads", "Hugging Face GGUF uploads"),
    ]
    handle_cases = [
        ("search X posts from @liquid_ai about LFM2.5", "LFM2.5", ["liquid_ai"]),
        ("check X posts from @huggingface about GGUF", "GGUF", ["huggingface"]),
    ]
    for repeat in range(16):
        for prompt, query in cases:
            add_tool(rows, by_name, prompt, "x_search", {"query": query}, "x_social_search", "iter14_x_search", str(repeat))
        for prompt, query, handles in handle_cases:
            add_tool(
                rows,
                by_name,
                prompt,
                "x_search",
                {"query": query, "allowed_x_handles": handles},
                "x_social_search",
                "iter14_x_search_handles",
                str(repeat),
            )
    return rows


def search_files_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("search this repo for LFM2.5 GGUF", {"pattern": "LFM2.5 GGUF"}),
        ("search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        ("find browser_navigate in local files", {"pattern": "browser_navigate"}),
        ("search local files for iter14", {"pattern": "iter14"}),
        ("search my repo for tool_parser_type", {"pattern": "tool_parser_type"}),
        ("find where x_search is mentioned in the project", {"pattern": "x_search"}),
        ("search the codebase for live_hermes_eval", {"pattern": "live_hermes_eval"}),
        ("find Python files named live_hermes_eval", {"pattern": "live_hermes_eval", "target": "files"}),
        ("search this workspace for epoch06_candidate_losses", {"pattern": "epoch06_candidate_losses"}),
        ("find files matching *contrast* in the repo", {"pattern": "*contrast*", "target": "files"}),
        ("search local files, not the web, for browser_navigate", {"pattern": "browser_navigate"}),
        ("look in this repo for model_runtime_step01746_pythonic", {"pattern": "model_runtime_step01746_pythonic"}),
        ("search source files for run_lora_repair", {"pattern": "run_lora_repair"}),
        ("find docs mentioning Q6KXL", {"pattern": "Q6KXL"}),
        ("search my local files for llama.cpp", {"pattern": "llama.cpp"}),
    ]
    for repeat in range(12):
        for prompt, args in cases:
            add_tool(rows, by_name, prompt, "search_files", args, "local_file_search", "iter14_search_files", str(repeat))
    return rows


def retention_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    pairs = [
        ("search means look carefully; explain the word, do not use tools", "To search means to look carefully for something."),
        ("search for meaning in this sentence, but do not use tools", "Here, `search for meaning` means interpret the sentence rather than looking something up."),
        ("explain what LFM2.5 GGUF means; do not search", "LFM2.5 GGUF refers to a Liquid AI LFM2.5 model checkpoint packaged in the GGUF format used by llama.cpp-style runtimes."),
        ("what does x_search do conceptually? Do not call it.", "`x_search` is a tool for searching X/Twitter posts, profiles, and threads."),
        ("what is a browser in one sentence?", "A browser is an application for viewing and interacting with websites."),
        ("what is a terminal in one sentence?", "A terminal is an interface for running local shell commands."),
        ("what is local file search?", "Local file search means looking through files on the current machine or repository."),
        ("do not use tools; what's the weather as a concept?", "Weather is the short-term condition of the atmosphere, including temperature, wind, clouds, and precipitation."),
        ("do not use tools; what command prints the current folder?", "`pwd` prints the current folder."),
        ("hello", "Hello! How can I help you today?"),
        ("are you there?", "I'm here. What would you like to do?"),
        ("what happened on Apollo 13?", "Apollo 13 suffered an oxygen tank explosion on the way to the Moon. The landing was aborted, but the crew returned safely after improvised power, navigation, and life-support work."),
        ("which moon landing was nearly a disaster?", "Apollo 13 was the Moon mission that nearly became a disaster after an oxygen tank exploded."),
        ("tell me about the moon landing in detail", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin descended in Eagle while Michael Collins orbited in Columbia, then Armstrong and Aldrin walked on the surface, collected samples, and deployed experiments."),
        ("no, answer the original question with more detail", "Apollo 13 was nearly disastrous because an oxygen tank explosion damaged the service module, forcing the crew to conserve resources and use the lunar module as a lifeboat before returning safely."),
        ("return a tiny JSON object with status ok", "{\"status\":\"ok\"}"),
        ("write a sentence using the word browser", "The browser displayed the documentation page."),
        ("write a sentence using the word search", "The search took longer because the notes were scattered."),
        ("without tools, what is 12 times 13?", "12 times 13 is 156."),
        ("explain what a tool call is without making one", "A tool call is a structured request from an assistant to an external capability such as a browser, terminal, or file reader."),
    ]
    for repeat in range(12):
        for prompt, answer in pairs:
            add_text(rows, by_name, prompt, answer, "no_tool_retention", "iter14_retention", str(repeat))
            if repeat < 6 and not prompt.lower().startswith("do not use tools"):
                add_text(rows, by_name, "Do not use tools. " + prompt, answer, "no_tool_retention", "iter14_retention_explicit", str(repeat))
    return rows


def correction_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cases = [
        ("no, search the web, not X", "browser_navigate", {"url": google("LFM2.5 GGUF")}),
        ("that was local files, I meant search online", "browser_navigate", {"url": google("LFM2.5 GGUF")}),
        ("yes use browser tools to check the weather", "browser_navigate", {"url": google("Austin weather today")}),
        ("no, search X specifically", "x_search", {"query": "LFM2.5 GGUF"}),
        ("I mean Twitter/X, not the web", "x_search", {"query": "LFM2.5 GGUF"}),
        ("no, search this repo instead", "search_files", {"pattern": "LFM2.5 GGUF"}),
        ("search local files, not online", "search_files", {"pattern": "browser_navigate"}),
        ("that didn't work, try a normal browser search", "browser_navigate", {"url": google("Austin weather today")}),
    ]
    for repeat in range(8):
        for prompt, name, args in cases:
            add_tool(rows, by_name, prompt, name, args, "correction_boundary", "iter14_correction", str(repeat))
    final_pairs = [
        ("what's Austin weather?", "browser_navigate", {"url": google("Austin weather")}, "Austin weather: 87 F and partly cloudy.", "Austin is 87 F and partly cloudy."),
        ("search X for LFM2.5", "x_search", {"query": "LFM2.5"}, "Recent X posts discuss LFM2.5 GGUF quants.", "Recent X posts discuss LFM2.5 GGUF quants."),
        ("search local files for browser_navigate", "search_files", {"pattern": "browser_navigate"}, "scripts/live_hermes_eval.py: browser_navigate", "I found `browser_navigate` in `scripts/live_hermes_eval.py`."),
    ]
    tools = expected_tools(by_name)
    from build_epoch05_tool_repair_masked_dataset import final_row

    for repeat in range(4):
        for prompt, name, args, result, final in final_pairs:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"iter14_finalization_{stable_id(prompt, name, str(repeat))}"
                rows.append(final_row(row_id, prompt, name, args, result, final, tools, "correction_boundary"))
    return rows


def pick(rows: list[dict[str, Any]], train_size: int, valid_size: int, test_size: int, rng: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    targets = {
        "browser_general_web": int(train_size * 0.40),
        "x_social_search": int(train_size * 0.20),
        "local_file_search": int(train_size * 0.15),
        "no_tool_retention": int(train_size * 0.20),
    }
    targets["correction_boundary"] = train_size - sum(targets.values())
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(row["category"], []).append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)
    picked: list[dict[str, Any]] = []
    used: set[str] = set()
    for category, target in targets.items():
        for row in buckets.get(category, [])[:target]:
            picked.append(row)
            used.add(row["case_id"])
    leftovers = [row for row in rows if row["case_id"] not in used]
    rng.shuffle(leftovers)
    picked.extend(leftovers[: max(0, train_size - len(picked))])
    rng.shuffle(picked)
    used = {row["case_id"] for row in picked}
    remaining = [row for row in rows if row["case_id"] not in used]
    rng.shuffle(remaining)
    valid = remaining[:valid_size]
    test = remaining[valid_size : valid_size + test_size]
    return picked[:train_size], valid, test


def main() -> None:
    parser = argparse.ArgumentParser(description="Build focused browser-vs-X-vs-local-files contrast dataset.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--train-size", type=int, default=1200)
    parser.add_argument("--valid-size", type=int, default=120)
    parser.add_argument("--test-size", type=int, default=120)
    parser.add_argument("--seed", type=int, default=1414)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _, by_name = load_tools(args.tools_json)
    rows = browser_rows(by_name) + x_rows(by_name) + search_files_rows(by_name) + retention_rows(by_name) + correction_rows(by_name)

    valid_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for row in rows:
        ok, token_count, reason = validate_render(row, tokenizer, args.max_tokens)
        if not ok:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        row = dict(row)
        row["token_count"] = token_count
        valid_rows.append(row)
    by_id = {row["case_id"]: row for row in valid_rows}
    valid_rows = list(by_id.values())
    train, valid, test = pick(valid_rows, args.train_size, args.valid_size, args.test_size, rng)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    write_jsonl(args.out / "test.jsonl", test)
    manifest: dict[str, Any] = {
        "name": "iter14_browser_x_files_contrast_router",
        "model": args.model,
        "tools_json": str(args.tools_json),
        "format": "messages+tools JSONL for prompt-masked MLX LoRA",
        "max_tokens": args.max_tokens,
        "target_split_counts": {"train": args.train_size, "valid": args.valid_size, "test": args.test_size},
        "split_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "available_rows": len(valid_rows),
        "rejected_counts": rejected,
        "has_xml_tool_call_target": False,
        "tool_policy": {
            "browser_navigate": "General web/current/latest/weather/open/navigate requests.",
            "x_search": "Only explicit X/Twitter/social-current requests.",
            "search_files": "Only local repo/workspace/file/code search requests.",
        },
        "system_policy": SYSTEM,
        "chat_retention_policy": SYSTEM_CHAT,
        "train_category_counts": {},
        "examples": [],
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
        for row in train[:12]
    ]
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
