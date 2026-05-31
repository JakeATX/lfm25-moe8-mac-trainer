#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def message_text(row: dict) -> str:
    parts: list[str] = []
    for msg in row.get("messages", []):
        role = msg.get("role", "unknown")
        content = msg.get("content")
        if content:
            parts.append(f"{role}: {content}")
        if msg.get("tool_calls"):
            parts.append(f"{role}: {json.dumps(msg['tool_calls'], ensure_ascii=False)}")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-chars", type=int, default=6_000_000)
    parser.add_argument("--max-rows", type=int, default=50_000)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    written_chars = 0
    with args.out.open("w", encoding="utf-8") as out:
        for dataset in args.datasets:
            for split in ("train.jsonl", "valid.jsonl", "test.jsonl"):
                path = dataset / split
                if not path.exists():
                    continue
                for row in load_jsonl(path):
                    text = message_text(row)
                    if not text:
                        continue
                    if written_rows >= args.max_rows or written_chars + len(text) > args.max_chars:
                        break
                    out.write(text.replace("\r", "") + "\n\n")
                    written_rows += 1
                    written_chars += len(text) + 2
                if written_rows >= args.max_rows or written_chars >= args.max_chars:
                    break
            if written_rows >= args.max_rows or written_chars >= args.max_chars:
                break

    manifest = {
        "out": str(args.out),
        "source_datasets": [str(path) for path in args.datasets],
        "rows": written_rows,
        "chars": written_chars,
        "note": "Plain-text calibration prompts for llama.cpp imatrix/KL runs; built from Hermes tool-call and no-tool traces.",
    }
    (args.out.with_suffix(args.out.suffix + ".manifest.json")).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
