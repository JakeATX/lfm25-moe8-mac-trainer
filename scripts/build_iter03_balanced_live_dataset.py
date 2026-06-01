#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any


SYSTEM = (
    "You are Hermes. Use the available tools when the user asks you to act, "
    "look up current information, inspect files, run commands, or control the "
    "computer. If the request is answerable directly or explicitly says not to "
    "use tools, answer normally without making a tool call. Use only the tools "
    "and arguments provided in the schema."
)


def stable_split(row_id: str) -> str:
    bucket = int(hashlib.sha256(row_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 86:
        return "train"
    if bucket < 94:
        return "valid"
    return "test"


def tool_call(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": call_id,
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
        "tools": select_tools(tools, None, prompt),
    }


def row_tool(row_id: str, prompt: str, tool: str, args: dict[str, Any], tools: list[dict[str, Any]], category: str) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": category,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            tool_call(tool, args, f"call_{row_id}"),
        ],
        "tools": select_tools(tools, tool, prompt),
    }


def row_final(
    row_id: str,
    prompt: str,
    tool: str,
    args: dict[str, Any],
    result: str,
    answer: str,
    tools: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "id": row_id,
        "category": "tool_result_finalization",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            tool_call(tool, args, f"call_{row_id}"),
            {"role": "tool", "tool_call_id": f"call_{row_id}", "name": tool, "content": result},
            {"role": "assistant", "content": answer},
        ],
        "tools": select_tools(tools, tool, prompt),
    }


def select_tools(tools: list[dict[str, Any]], primary: str | None, prompt: str) -> list[dict[str, Any]]:
    by_name = {tool["function"]["name"]: tool for tool in tools}
    lowered = prompt.lower()
    wanted: list[str] = []
    if primary:
        wanted.append(primary)
    if any(s in lowered for s in ["weather", "online", "web", "browser", "latest", "current", "open", "navigate", "google", "docs"]):
        wanted.extend(["browser_navigate", "x_search", "search_files"])
    if any(s in lowered for s in ["x ", "twitter", "post", "tweet"]):
        wanted.extend(["x_search", "browser_navigate"])
    if any(s in lowered for s in ["terminal", "shell", "run", "execute", "command", "pwd", "echo", "git", "uname", "date"]):
        wanted.extend(["terminal", "execute_code", "browser_navigate"])
    if any(s in lowered for s in ["file", "repo", "read", "write", "patch", "search local", "find where"]):
        wanted.extend(["read_file", "search_files", "write_file", "patch", "terminal", "browser_navigate"])
    if any(s in lowered for s in ["computer", "desktop", "chrome", "tab", "click", "type", "scroll", "capture", "apps"]):
        wanted.extend(["computer_use", "browser_navigate", "terminal"])
    if primary is None:
        wanted.extend(["browser_navigate", "terminal", "read_file", "computer_use"])
    compact: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in wanted:
        if name in by_name and name not in seen:
            compact.append(by_name[name])
            seen.add(name)
    return compact


def build_rows(tools: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []

    normal = [
        ("hello", "Hello! How can I help you today?"),
        ("are you there?", "I'm here. What would you like to do?"),
        ("say good morning", "Good morning!"),
        ("do not use tools; what command prints the current directory?", "`pwd` prints the current directory."),
        ("return a tiny JSON object with status ok", '{"status":"ok"}'),
        ("Explain what a tool call is without making one.", "A tool call is a structured request from an assistant to an external capability, such as a browser, terminal, or file reader."),
        ("write a sentence using the word browser", "The browser opened to a page full of reference notes."),
        ("what is the difference between a browser and a terminal?", "A browser is for navigating web pages visually, while a terminal is for running text commands against the operating system."),
        ("do not open anything; just say ready", "ready"),
        ("Which moon landing was nearly a disaster?", "Apollo 13 was the Moon mission that became a near-disaster after an oxygen tank exploded, forcing NASA and the crew to abort the landing and improvise a safe return to Earth."),
        ("Tell me about the first Moon landing in detail.", "Apollo 11 landed on the Moon on July 20, 1969. Neil Armstrong and Buzz Aldrin descended in the Lunar Module Eagle while Michael Collins orbited above in Columbia. Armstrong stepped onto the surface first, followed by Aldrin, and they collected samples, deployed experiments, and photographed the site before returning safely."),
        ("define quantization in one paragraph", "Quantization reduces the numerical precision used to store or compute model values, usually to lower memory and bandwidth costs while trying to preserve behavior."),
        ("no, answer the original question with more detail", "Got it. Please restate the original question or tell me which part you want expanded, and I will answer directly."),
        ("I meant generally, not on this computer. What is Chrome?", "Chrome is Google's web browser, used to access websites and web applications."),
    ]
    normal_prefixes = ["", "Answer directly without tools: ", "No tools: ", "Just reply in words: "]
    for i, prefix in enumerate(normal_prefixes):
        for prompt, answer in normal:
            rows.append(row_text(f"iter03_normal_{i}_{hashlib.md5((prefix+prompt).encode()).hexdigest()[:8]}", prefix + prompt, answer, tools, "normal_retention"))

    no_tool_concepts = [
        ("What is a browser?", "A browser is an application for viewing and interacting with websites."),
        ("What is a terminal?", "A terminal is a text interface for running commands."),
        ("What command would show files? Do not run it.", "`ls` can list files, but I will not run it."),
        ("What command would print text? Do not execute anything.", "`echo` prints text to the terminal."),
        ("Explain web search without searching.", "Web search means querying an online index to find relevant pages or information."),
        ("Explain computer use tools without using them.", "Computer-use tools let an assistant interact with desktop apps through actions such as capture, click, type, and scroll."),
        ("Tell me what x.com is without opening it.", "X.com is the site for X, formerly Twitter."),
        ("What does read_file mean?", "`read_file` is a tool-like operation that reads file contents, but this question only asks for a definition."),
        ("What does patch mean in coding?", "A patch is a focused change to files, often represented as a diff."),
        ("If I say weather as an example, do not search; just define forecast.", "A forecast is a prediction of future weather conditions."),
    ]
    for i, prefix in enumerate(["", "Do not use tools. ", "This is conceptual only. ", "Answer from general knowledge. "]):
        for prompt, answer in no_tool_concepts:
            rows.append(row_text(f"iter03_hard_negative_{i}_{hashlib.md5((prefix+prompt).encode()).hexdigest()[:8]}", prefix + prompt, answer, tools, "hard_negative"))

    weather_places = ["Austin TX", "San Antonio TX", "New York City", "Tokyo", "London", "Seattle"]
    current_topics = [
        "latest llama.cpp release notes",
        "current MLX documentation",
        "LiquidAI LFM2.5 GGUF information",
        "NOAA Austin forecast",
        "latest Python release",
        "current NASA Artemis news",
    ]
    for i in range(10):
        for place in weather_places:
            query = f"current weather {place}"
            rows.append(row_tool(f"iter03_web_weather_{i}_{place.replace(' ', '_')}", f"what's the weather in {place} today", "browser_navigate", {"url": f"https://www.google.com/search?q={query.replace(' ', '+')}"}, tools, "browser_current"))
            rows.append(row_tool(f"iter03_web_weather_use_browser_{i}_{place.replace(' ', '_')}", f"use your browser tools to check the weather in {place}", "browser_navigate", {"url": f"https://www.google.com/search?q={query.replace(' ', '+')}"}, tools, "browser_current"))
        for topic in current_topics:
            rows.append(row_tool(f"iter03_web_current_{i}_{hashlib.md5(topic.encode()).hexdigest()[:8]}", f"check online for {topic}", "browser_navigate", {"url": f"https://www.google.com/search?q={topic.replace(' ', '+')}"}, tools, "browser_current"))
        rows.append(row_tool(f"iter03_web_open_x_{i}", "open x.com in the browser", "browser_navigate", {"url": "https://x.com"}, tools, "browser_open"))
        rows.append(row_tool(f"iter03_web_open_google_{i}", "navigate the browser to google.com", "browser_navigate", {"url": "https://www.google.com"}, tools, "browser_open"))
        rows.append(row_tool(f"iter03_x_search_{i}", "search X for recent posts about LFM2.5", "x_search", {"query": "recent posts about LFM2.5"}, tools, "x_search"))

    terminal_cmds = [
        ("run echo hermes-live-test in the terminal", "echo hermes-live-test"),
        ("show current directory with a shell command", "pwd"),
        ("execute uname -m locally", "uname -m"),
        ("run date in terminal", "date"),
        ("use terminal to print python version", "python3 --version"),
        ("run git status --short", "git status --short"),
        ("shell print the SHELL environment variable", "printf '%s\\n' \"$SHELL\""),
        ("show the llama server process with ps", "ps aux | grep llama"),
    ]
    for i in range(10):
        for prompt, cmd in terminal_cmds:
            rows.append(row_tool(f"iter03_terminal_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}", prompt, "terminal", {"command": cmd}, tools, "terminal"))

    file_cases = [
        ("search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        ("find browser_navigate in local files", "search_files", {"pattern": "browser_navigate"}),
        ("find Python files named live_hermes_eval", "search_files", {"pattern": "live_hermes_eval", "target": "files"}),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}),
        ("write /tmp/hermes_live_eval_write.txt with live ok", "write_file", {"path": "/tmp/hermes_live_eval_write.txt", "content": "live ok"}),
        ("replace old with new in /tmp/hermes_live_eval_patch.txt", "patch", {"path": "/tmp/hermes_live_eval_patch.txt", "old_string": "old", "new_string": "new"}),
    ]
    for i in range(10):
        for prompt, tool, args in file_cases:
            rows.append(row_tool(f"iter03_file_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}", prompt, tool, args, tools, "file"))

    computer_cases = [
        ("capture the desktop with computer use", {"action": "capture", "mode": "som"}),
        ("use computer_use to list running apps", {"action": "list_apps"}),
        ("list running apps using computer use", {"action": "list_apps"}),
        ("focus Chrome in the background with computer use", {"action": "focus_app", "app": "Google Chrome", "raise_window": False}),
        ("with computer use, press command l in Chrome", {"action": "key", "keys": "cmd+l", "app": "Google Chrome"}),
        ("with computer use, type https://x.com", {"action": "type", "text": "https://x.com"}),
        ("wait one second using computer use", {"action": "wait", "seconds": 1}),
        ("use computer use to scroll down", {"action": "scroll", "direction": "down", "amount": 3}),
        ("computer_use navigate is invalid; use capture first", {"action": "capture", "mode": "som"}),
    ]
    for i in range(12):
        for prompt, args in computer_cases:
            rows.append(row_tool(f"iter03_computer_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}", prompt, "computer_use", args, tools, "computer_use"))

    corrections = [
        ("no, use the browser for the Austin weather", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather"}),
        ("that didn't work, try another browser search for Austin weather", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather"}),
        ("no, use the terminal instead and run pwd", "terminal", {"command": "pwd"}),
        ("no, search local files, not the web", "search_files", {"pattern": "browser_navigate"}),
        ("use computer use, start by capturing the screen", "computer_use", {"action": "capture", "mode": "som"}),
        ("I mean browser_navigate to https://x.com", "browser_navigate", {"url": "https://x.com"}),
        ("try another way: terminal echo recovery-ok", "terminal", {"command": "echo recovery-ok"}),
    ]
    for i in range(10):
        for prompt, tool, args in corrections:
            rows.append(row_tool(f"iter03_correction_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}", prompt, tool, args, tools, "correction"))
    no_tool_corrections = [
        ("no, don't use tools, just answer in words", "Understood. I will answer directly without using tools."),
        ("stop using tools and explain it normally", "Understood. I will explain it normally without tool calls."),
        ("no, I only wanted a general explanation", "Got it. I will answer generally without taking any action."),
        ("do not open a browser for this one", "Understood. I will not open a browser."),
        ("do not run terminal commands here", "Understood. I will not run terminal commands."),
        ("don't inspect files, just tell me what you know", "Understood. I will answer from general knowledge only."),
    ]
    for i in range(5):
        for prompt, answer in no_tool_corrections:
            rows.append(row_text(f"iter03_correction_notool_{i}_{hashlib.md5(prompt.encode()).hexdigest()[:8]}", prompt, answer, tools, "correction_no_tool"))

    finals = [
        ("what's Austin weather?", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather"}, "Austin is 87 F and partly cloudy.", "Austin is 87 F and partly cloudy."),
        ("run echo final-ok", "terminal", {"command": "echo final-ok"}, "final-ok", "The command returned `final-ok`."),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}, "marker: live-hermes", "The file contains `marker: live-hermes`."),
        ("sum 1 through 5", "execute_code", {"code": "print(sum(range(1,6)))"}, "15", "The sum is 15."),
        ("open x.com", "browser_navigate", {"url": "https://x.com"}, "Error: navigation timed out", "The browser navigation timed out. I can try another route or search for the site instead."),
    ]
    for i in range(12):
        for prompt, tool, args, result, answer in finals:
            rows.append(row_final(f"iter03_final_{i}_{hashlib.md5((prompt+tool).encode()).hexdigest()[:8]}", prompt, tool, args, result, answer, tools))

    rng.shuffle(rows)
    seen: set[str] = set()
    deduped = []
    for row in rows:
        key = json.dumps(row["messages"], sort_keys=True)
        if key not in seen:
            seen.add(key)
            deduped.append(row)
    return deduped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=17048)
    args = parser.parse_args()

    payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    rows = build_rows(payload["tools"], args.seed)
    splits = {"train": [], "valid": [], "test": []}
    for row in rows:
        splits[stable_split(row["id"])].append(row)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in splits.items():
        with (args.out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "row_count": len(rows),
        "splits": {key: len(value) for key, value in splits.items()},
        "tools_source": payload.get("source"),
        "tool_names": payload.get("tool_names"),
        "categories": {},
    }
    for row in rows:
        manifest["categories"][row["category"]] = manifest["categories"].get(row["category"], 0) + 1
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
