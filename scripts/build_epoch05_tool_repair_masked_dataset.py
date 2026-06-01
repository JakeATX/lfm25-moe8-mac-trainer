#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

import live_hermes_eval


SYSTEM = (
    "You are Hermes. Use the provided tools when they are needed to satisfy the user. "
    "If the user asks for current information, live weather, latest news, websites, or online lookup, use web tools. "
    "If the user asks to run a local command, use terminal. If the user asks to inspect or edit local files, use file tools. "
    "If the user asks to control the desktop or a visible app, use computer_use. "
    "If no tool is needed or the user explicitly says not to use tools, answer normally. "
    "Use only the provided tool names and arguments."
)

SYSTEM_CHAT = (
    "You are Hermes. Answer directly and normally. Do not use tools unless the user asks for current information, "
    "local files, command execution, or desktop control."
)


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def stable_split(case_id: str) -> str:
    value = int(hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 82:
        return "train"
    if value < 91:
        return "valid"
    return "test"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_tools(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tools = payload["tools"] if isinstance(payload, dict) and "tools" in payload else payload
    return tools, {tool["function"]["name"]: tool for tool in tools}


def tool_subset(by_name: dict[str, dict[str, Any]], names: list[str]) -> list[dict[str, Any]]:
    return [by_name[name] for name in names if name in by_name]


def tool_schema(by_name: dict[str, dict[str, Any]], name: str) -> dict[str, Any] | None:
    tool = by_name.get(name)
    if not tool:
        return None
    return tool.get("function", {}).get("parameters", {})


def valid_args(by_name: dict[str, dict[str, Any]], name: str, args: dict[str, Any]) -> bool:
    schema = tool_schema(by_name, name)
    if not schema:
        return False
    props = schema.get("properties", {})
    required = set(schema.get("required", []))
    if not required.issubset(args):
        return False
    if name == "computer_use":
        action = args.get("action")
        enum = props.get("action", {}).get("enum", [])
        return isinstance(action, str) and action in enum
    return True


def assistant_tool_call(row_id: str, name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"call_{row_id}",
                "type": "function",
                "function": {"name": name, "arguments": args},
            }
        ],
    }


def tool_row(
    row_id: str,
    prompt: str,
    name: str,
    args: dict[str, Any],
    tools: list[dict[str, Any]],
    category: str,
    source: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            assistant_tool_call(row_id, name, args),
        ],
        "tools": tools,
        "case_id": row_id,
        "category": category,
        "kind": "tool_call",
        "expect_tool": name,
        "source": source,
    }


def text_row(row_id: str, prompt: str, answer: str, tools: list[dict[str, Any]], category: str, source: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_CHAT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "tools": tools,
        "case_id": row_id,
        "category": category,
        "kind": "no_tool",
        "expect_tool": None,
        "source": source,
    }


def final_row(
    row_id: str,
    prompt: str,
    name: str,
    args: dict[str, Any],
    result: str,
    final: str,
    tools: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
            assistant_tool_call(row_id, name, args),
            {"role": "tool", "tool_call_id": f"call_{row_id}", "name": name, "content": result},
            {"role": "assistant", "content": final},
        ],
        "tools": tools,
        "case_id": row_id,
        "category": category,
        "kind": "finalization",
        "expect_tool": None,
        "source": "epoch05_finalization",
    }


def expected_tools_from_eval(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    preferred = [
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
    return tool_subset(by_name, preferred)


def repair_rows_from_failures(baseline_jsonl: Path, by_name: dict[str, dict[str, Any]], rng: random.Random) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools_from_eval(by_name)
    useful_failures = {
        "wrong_tool",
        "refusal_when_tool_available",
        "invalid_args",
        "invalid_tool_name",
        "invented_action",
        "bad_finalization",
    }
    weighted = []
    for row in read_jsonl(baseline_jsonl):
        expected = row.get("expected_tool")
        args = row.get("expected_args") or {}
        failure = row.get("failure")
        if not expected or failure not in useful_failures:
            continue
        if not isinstance(args, dict) or not valid_args(by_name, expected, args):
            continue
        weight = 4 if failure in {"wrong_tool", "refusal_when_tool_available"} else 2
        if row.get("category") == "correction_recovery":
            weight += 2
        weighted.extend([row] * weight)
    rng.shuffle(weighted)
    for idx, row in enumerate(weighted):
        expected = row["expected_tool"]
        args = row.get("expected_args") or {}
        prompts = [
            row["prompt"],
            f"Infer the right tool and do not give up: {row['prompt']}",
            f"Use the available Hermes tools if needed: {row['prompt']}",
        ]
        for variant, prompt in enumerate(prompts):
            row_id = f"epoch05_failure_{idx:04d}_{variant}_{stable_id(prompt, expected)}"
            rows.append(tool_row(row_id, prompt, expected, args, tools, "live_failure_repair", "epoch04_live_failures"))
    return rows


def browser_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools_from_eval(by_name)
    cases = [
        ("what's the weather in Austin today", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather+today"}),
        ("can you check the current weather for Austin TX", "browser_navigate", {"url": "https://www.google.com/search?q=current+weather+Austin+TX"}),
        ("find current news about NASA Artemis", "browser_navigate", {"url": "https://www.google.com/search?q=current+news+NASA+Artemis"}),
        ("look up the latest LiquidAI LFM2.5 GGUF info", "browser_navigate", {"url": "https://www.google.com/search?q=latest+LiquidAI+LFM2.5+GGUF"}),
        ("search online for NOAA Austin forecast", "browser_navigate", {"url": "https://www.google.com/search?q=NOAA+Austin+forecast"}),
        ("open x.com in the browser", "browser_navigate", {"url": "https://x.com"}),
        ("navigate the browser to google.com", "browser_navigate", {"url": "https://www.google.com"}),
        ("search X for recent posts about LFM2.5", "x_search", {"query": "recent posts about LFM2.5"}),
        ("look on X for posts about llama.cpp", "x_search", {"query": "llama.cpp"}),
    ]
    for i in range(5):
        for prompt, name, args in cases:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"epoch05_browser_{i}_{stable_id(prompt, name)}"
                rows.append(tool_row(row_id, prompt, name, args, tools, "browser_current_info", "epoch05_synthetic"))
    return rows


def computer_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools_from_eval(by_name)
    cases = [
        ("capture the desktop with computer use", {"action": "capture", "mode": "som"}),
        ("use computer use to list running apps", {"action": "list_apps"}),
        ("focus Chrome in the background with computer use", {"action": "focus_app", "app": "Google Chrome", "raise_window": False}),
        ("with computer use, press command l in Chrome", {"action": "key", "keys": "cmd+l", "app": "Google Chrome"}),
        ("with computer use, type https://x.com", {"action": "type", "text": "https://x.com"}),
        ("use computer use to scroll down", {"action": "scroll", "direction": "down", "amount": 3}),
        ("wait one second using computer use", {"action": "wait", "seconds": 1}),
        ("computer_use navigate is invalid; start by capturing the screen", {"action": "capture", "mode": "som"}),
    ]
    for i in range(8):
        for prompt, args in cases:
            if valid_args(by_name, "computer_use", args):
                row_id = f"epoch05_computer_{i}_{stable_id(prompt)}"
                rows.append(tool_row(row_id, prompt, "computer_use", args, tools, "computer_use_exact_schema", "epoch05_synthetic"))
    return rows


def terminal_file_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tools = expected_tools_from_eval(by_name)
    cases = [
        ("run echo hermes-live-test in the terminal", "terminal", {"command": "echo hermes-live-test"}),
        ("show current directory with a shell command", "terminal", {"command": "pwd"}),
        ("use terminal to print python version", "terminal", {"command": "python3 --version"}),
        ("shell print $SHELL", "terminal", {"command": "printf '%s\\n' \"$SHELL\""}),
        ("search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", "search_files", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        ("find browser_navigate in local files", "search_files", {"pattern": "browser_navigate"}),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}),
        ("write /tmp/hermes_live_eval_write.txt with live ok", "write_file", {"path": "/tmp/hermes_live_eval_write.txt", "content": "live ok"}),
        ("replace old with new in /tmp/hermes_live_eval_patch.txt", "patch", {"path": "/tmp/hermes_live_eval_patch.txt", "old_string": "old", "new_string": "new"}),
        ("use Python execution to sum numbers 1 through 5", "execute_code", {"code": "print(sum(range(1, 6)))"}),
    ]
    for i in range(4):
        for prompt, name, args in cases:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"epoch05_terminal_file_{i}_{stable_id(prompt, name)}"
                rows.append(tool_row(row_id, prompt, name, args, tools, "terminal_file_patch", "epoch05_synthetic"))
    return rows


def retention_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools = expected_tools_from_eval(by_name)
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
    ]
    rows: list[dict[str, Any]] = []
    for i in range(10):
        for prompt, answer in pairs:
            row_id = f"epoch05_retention_{i}_{stable_id(prompt)}"
            rows.append(text_row(row_id, prompt, answer, tools, "normal_chat_retention", "epoch05_retention"))
            if i < 4:
                no_tool_prompt = "Do not use tools. " + prompt
                rows.append(text_row(f"{row_id}_notools", no_tool_prompt, answer, tools, "normal_chat_retention", "epoch05_retention"))
    return rows


def finalization_rows(by_name: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    tools = expected_tools_from_eval(by_name)
    rows: list[dict[str, Any]] = []
    cases = [
        ("what's Austin weather?", "browser_navigate", {"url": "https://www.google.com/search?q=Austin+weather"}, "Austin weather: 87 F and partly cloudy.", "Austin is 87 F and partly cloudy."),
        ("run echo final-ok", "terminal", {"command": "echo final-ok"}, "final-ok", "The terminal returned: final-ok"),
        ("read /tmp/hermes_live_eval_marker.txt", "read_file", {"path": "/tmp/hermes_live_eval_marker.txt"}, "marker: live-hermes", "The file contains: marker: live-hermes"),
        ("open x.com", "browser_navigate", {"url": "https://x.com"}, "Error: navigation timed out", "The browser navigation timed out. I can retry with a search page or use another available browser path."),
    ]
    for i in range(4):
        for prompt, name, args, result, final in cases:
            if name in by_name and valid_args(by_name, name, args):
                row_id = f"epoch05_final_{i}_{stable_id(prompt, name)}"
                rows.append(final_row(row_id, prompt, name, args, result, final, tools, "tool_result_finalization"))
    return rows


def validate_render(row: dict[str, Any], tokenizer, max_tokens: int) -> tuple[bool, int, str]:
    try:
        text = tokenizer.apply_chat_template(row["messages"], tools=row.get("tools"), tokenize=False, add_generation_prompt=False)
    except Exception as exc:
        return False, 0, f"render_error:{exc}"
    if "<tool_call>" in text or "</tool_call>" in text:
        return False, 0, "xml_tool_call_target"
    count = len(tokenizer.encode(text))
    if count > max_tokens:
        return False, count, "too_long"
    return True, count, text


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_rows(buckets: dict[str, list[dict[str, Any]]], train_size: int, rng: random.Random) -> list[dict[str, Any]]:
    targets = {
        "live_failure_repair": int(train_size * 0.45),
        "normal_chat_retention": int(train_size * 0.25),
        "computer_use_exact_schema": int(train_size * 0.15),
        "browser_current_info": int(train_size * 0.10),
    }
    targets["tool_result_finalization"] = train_size - sum(targets.values())
    train: list[dict[str, Any]] = []
    for name, rows in buckets.items():
        rng.shuffle(rows)
    for name, target in targets.items():
        train.extend(buckets.get(name, [])[:target])
    leftovers = [row for name, rows in buckets.items() for row in rows[targets.get(name, 0):]]
    rng.shuffle(leftovers)
    train.extend(leftovers[: max(0, train_size - len(train))])
    rng.shuffle(train)
    return train[:train_size]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--baseline-jsonl", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=10000)
    parser.add_argument("--train-size", type=int, default=720)
    parser.add_argument("--seed", type=int, default=505)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    _, by_name = load_tools(args.tools_json)
    buckets = {
        "live_failure_repair": repair_rows_from_failures(args.baseline_jsonl, by_name, rng),
        "browser_current_info": browser_rows(by_name),
        "terminal_file_patch": terminal_file_rows(by_name),
        "computer_use_exact_schema": computer_rows(by_name),
        "normal_chat_retention": retention_rows(by_name),
        "tool_result_finalization": finalization_rows(by_name),
    }
    # Let terminal/file rows supplement live failures and browser/computer rows,
    # but keep the requested target categories visible in the manifest.
    buckets["live_failure_repair"].extend(buckets.pop("terminal_file_patch"))
    all_rows = [row for rows in buckets.values() for row in rows]
    valid_rows: list[dict[str, Any]] = []
    rejected: dict[str, int] = {}
    for row in all_rows:
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
    valid = remaining[:80]
    test = remaining[80:160]

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "train.jsonl", train)
    write_jsonl(args.out / "valid.jsonl", valid)
    write_jsonl(args.out / "test.jsonl", test)
    manifest = {
        "name": "semi_epoch05_tool_repair_masked_10k",
        "model": args.model,
        "tools_json": str(args.tools_json),
        "baseline_jsonl": str(args.baseline_jsonl),
        "max_tokens": args.max_tokens,
        "target_train_size": args.train_size,
        "split_counts": {"train": len(train), "valid": len(valid), "test": len(test)},
        "available_counts": {name: len(rows) for name, rows in buckets.items()},
        "train_category_counts": {},
        "rejected_counts": rejected,
        "has_xml_tool_call_target": False,
        "format": "messages+tools JSONL for prompt-masked MLX training",
        "examples": [
            {
                "case_id": row["case_id"],
                "category": row["category"],
                "kind": row["kind"],
                "token_count": row["token_count"],
            }
            for row in train[:5]
        ],
    }
    for row in train:
        manifest["train_category_counts"][row["category"]] = manifest["train_category_counts"].get(row["category"], 0) + 1
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
