#!/usr/bin/env python3
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


ROLE_MAP = {
    "system": "system",
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "tool": "tool",
}


def stable_bucket(row_id: str) -> int:
    digest = hashlib.sha256(row_id.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % 100


def split_name(row_id: str) -> str:
    bucket = stable_bucket(row_id)
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "valid"
    return "test"


def balanced_tags(text: str, tag: str) -> bool:
    return text.count(f"<{tag}>") == text.count(f"</{tag}>")


def validate_row(row: dict) -> tuple[bool, str]:
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or not conversations:
        return False, "missing_conversations"
    assistant_count = 0
    full_text_parts = []
    for msg in conversations:
        if not isinstance(msg, dict):
            return False, "bad_message"
        role = ROLE_MAP.get(msg.get("from") or msg.get("role"))
        value = msg.get("value") if "value" in msg else msg.get("content")
        if role is None:
            return False, "bad_role"
        if not isinstance(value, str) or not value.strip():
            return False, "empty_message"
        if role == "assistant":
            assistant_count += 1
        full_text_parts.append(value)
    if assistant_count == 0:
        return False, "missing_assistant"
    joined = "\n".join(full_text_parts)
    for tag in ("think", "tool_call", "tool_response"):
        if not balanced_tags(joined, tag):
            return False, f"unbalanced_{tag}"
    return True, "ok"


def render_chatml(row: dict) -> str:
    chunks = ["<|startoftext|>"]
    for msg in row["conversations"]:
        role = ROLE_MAP[msg.get("from") or msg.get("role")]
        value = msg.get("value") if "value" in msg else msg.get("content")
        chunks.append(f"<|im_start|>{role}\n{value}<|im_end|>\n")
    return "".join(chunks)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="DJLougen/hermes-agent-traces-filtered")
    parser.add_argument("--config", default="default")
    parser.add_argument("--split", default="train")
    parser.add_argument("--model", default="LiquidAI/LFM2.5-8B-A1B-MLX-bf16")
    parser.add_argument("--out", default="lfm25_hermes_ft/datasets/hermes_filtered_text")
    parser.add_argument("--max-tokens", type=int, default=16_000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ds = load_dataset(args.dataset, args.config, split=args.split)

    splits = {"train": [], "valid": [], "test": []}
    reject_reasons = Counter()
    token_stats = Counter()
    examples = []

    for row in ds:
        row_id = str(row.get("id") or hashlib.sha256(str(row).encode("utf-8")).hexdigest())
        ok, reason = validate_row(row)
        if not ok:
            reject_reasons[reason] += 1
            continue
        text = render_chatml(row)
        token_count = len(tokenizer.encode(text))
        token_stats["total_tokens"] += token_count
        token_stats["accepted_rows_before_cap"] += 1
        if token_count > args.max_tokens:
            reject_reasons["over_token_cap"] += 1
            continue
        split = split_name(row_id)
        item = {
            "text": text,
            "id": row_id,
            "source_dataset": args.dataset,
            "token_count": token_count,
            "category": row.get("category"),
            "subcategory": row.get("subcategory"),
            "task": row.get("task"),
        }
        splits[split].append(item)
        if len(examples) < 3:
            examples.append({k: item[k] for k in item if k != "text"} | {"text_prefix": text[:1200]})

    write_jsonl(out_dir / "train.jsonl", splits["train"])
    write_jsonl(out_dir / "valid.jsonl", splits["valid"])
    write_jsonl(out_dir / "test.jsonl", splits["test"])

    accepted = sum(len(v) for v in splits.values())
    manifest = {
        "dataset": args.dataset,
        "config": args.config,
        "split": args.split,
        "model_tokenizer": args.model,
        "max_tokens": args.max_tokens,
        "input_rows": len(ds),
        "accepted_rows": accepted,
        "split_counts": {k: len(v) for k, v in splits.items()},
        "reject_reasons": dict(reject_reasons),
        "token_stats": dict(token_stats),
        "examples": examples,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
