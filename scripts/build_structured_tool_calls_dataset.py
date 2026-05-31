#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


TOOL_CALL_RE = re.compile(r"^<\|tool_call_start\|>\[(?P<call>.+)\]<\|tool_call_end\|>$")


def stable_id(*parts: str) -> str:
    return hashlib.sha1("\x1f".join(parts).encode("utf-8")).hexdigest()[:16]


def load_tools(path: Path) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tools = raw["tools"] if isinstance(raw, dict) and "tools" in raw else raw
    if not isinstance(tools, list):
        raise ValueError(f"Expected a tool list in {path}")
    return tools


def required_args(tools: list[dict[str, Any]]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for tool in tools:
        fn = tool["function"]
        params = fn.get("parameters") or {}
        out[fn["name"]] = set(params.get("required") or [])
    return out


def parse_native_tool_call(content: str) -> tuple[str, dict[str, Any]] | None:
    match = TOOL_CALL_RE.fullmatch(content.strip())
    if not match:
        return None
    node = ast.parse(match.group("call"), mode="eval").body
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        raise ValueError(f"Invalid pythonic tool call: {content}")
    args: dict[str, Any] = {}
    for keyword in node.keywords:
        if keyword.arg is None:
            raise ValueError(f"Star kwargs are not supported: {content}")
        args[keyword.arg] = ast.literal_eval(keyword.value)
    return node.func.id, args


def structured_tool_message(name: str, args: dict[str, Any], call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "tool_calls": [
            {
                "id": call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": args,
                },
            }
        ],
    }


def convert_messages(messages: list[dict[str, Any]], tool_names: set[str], required: dict[str, set[str]]) -> tuple[list[dict[str, Any]], int]:
    converted: list[dict[str, Any]] = []
    count = 0
    pending_tool_call_ids: list[str] = []
    for index, message in enumerate(messages):
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and isinstance(content, str):
            parsed = parse_native_tool_call(content)
            if parsed:
                name, args = parsed
                if name not in tool_names:
                    raise ValueError(f"Unknown tool in target: {name}")
                missing = required[name] - set(args)
                if missing:
                    raise ValueError(f"Missing required args for {name}: {sorted(missing)}")
                call_id = f"call_{stable_id(json.dumps(messages[: index + 1], sort_keys=True, ensure_ascii=False))}"
                converted.append(structured_tool_message(name, args, call_id))
                pending_tool_call_ids.append(call_id)
                count += 1
                continue
        if role == "tool":
            tool_msg = dict(message)
            if pending_tool_call_ids:
                tool_msg["tool_call_id"] = pending_tool_call_ids.pop(0)
            converted.append(tool_msg)
        else:
            converted.append(dict(message))
    return converted, count


def iter_jsonl(paths: list[Path], split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for root in paths:
        path = root / f"{split}.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
    return rows


def validate_template(tokenizer: Any, row: dict[str, Any]) -> None:
    rendered = tokenizer.apply_chat_template(
        row["messages"],
        tools=row["tools"],
        tokenize=False,
        add_generation_prompt=False,
    )
    if any(msg.get("tool_calls") for msg in row["messages"]) and "<|tool_call_start|>" not in rendered:
        raise ValueError("Structured tool_calls row did not render native tool-call syntax")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", action="append", type=Path, required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--validate-limit", type=int, default=250)
    args = parser.parse_args()

    tools = load_tools(args.tools_json)
    tool_names = {tool["function"]["name"] for tool in tools}
    required = required_args(tools)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    args.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "sources": [str(path) for path in args.source],
        "tools_json": str(args.tools_json),
        "model": args.model,
        "format": "MLX ChatDataset JSONL with assistant tool_calls dict arguments; tokenizer renders native LFM pythonic tool calls.",
        "splits": {},
    }
    category_counts: Counter[str] = Counter()
    for split in ("train", "valid", "test"):
        seen: set[str] = set()
        rows_out: list[dict[str, Any]] = []
        converted_count = 0
        for row in iter_jsonl(args.source, split):
            messages, count = convert_messages(row["messages"], tool_names, required)
            sig = json.dumps({"messages": messages, "tools": tools}, sort_keys=True, ensure_ascii=False)
            if sig in seen:
                continue
            seen.add(sig)
            new_row = {
                "messages": messages,
                "tools": tools,
                "case_id": row.get("case_id") or stable_id(sig),
                "category": row.get("category", "unknown"),
                "kind": row.get("kind", "unknown"),
                "expect_tool": row.get("expect_tool"),
                "source": f"structured:{row.get('source', 'unknown')}",
            }
            if len(rows_out) < args.validate_limit:
                validate_template(tokenizer, new_row)
            rows_out.append(new_row)
            converted_count += count
            category_counts.update([new_row["category"]])
        with (args.out / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows_out:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["splits"][split] = {
            "rows": len(rows_out),
            "converted_tool_call_messages": converted_count,
        }
    manifest["category_counts_all_splits"] = dict(sorted(category_counts.items()))
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
