#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from build_colloquial_tool_router_dataset import TOOL_JSON


TOOLS = json.loads(TOOL_JSON)
LIST_RE = re.compile(r"\nList of tools: \[.*\]\s*$", re.S)
TURN_RE = re.compile(r"<\|im_start\|>([a-zA-Z_]+)\n(.*?)<\|im_end\|>\n?", re.S)


def text_to_messages(text: str) -> list[dict[str, str]]:
    messages = []
    for role, content in TURN_RE.findall(text):
        if role == "system":
            content = LIST_RE.sub("", content).rstrip()
        messages.append({"role": role, "content": content})
    if not messages:
        raise ValueError("no chatml turns found")
    return messages


def convert_file(src: Path, dst: Path) -> int:
    count = 0
    with src.open("r", encoding="utf-8") as in_handle, dst.open("w", encoding="utf-8") as out_handle:
        for line in in_handle:
            if not line.strip():
                continue
            row = json.loads(line)
            messages = text_to_messages(row["text"])
            out_handle.write(json.dumps({"messages": messages, "tools": TOOLS}, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in ("train", "valid", "test"):
        src_file = args.src / f"{split}.jsonl"
        if not src_file.exists():
            continue
        splits[split] = convert_file(src_file, args.out / f"{split}.jsonl")
    manifest = {
        "format": "mlx_lm chat jsonl",
        "source": str(args.src),
        "tool_parser_type_required": "pythonic",
        "prompt_masking_supported": True,
        "splits": splits,
        "tools": [tool["function"]["name"] for tool in TOOLS],
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
