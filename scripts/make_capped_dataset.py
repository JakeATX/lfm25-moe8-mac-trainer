#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-tokens", required=True, type=int)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary = {
        "source": str(args.input),
        "max_tokens": args.max_tokens,
        "splits": {},
    }
    for split in ("train", "valid", "test"):
        kept = 0
        dropped = 0
        max_seen = 0
        in_path = args.input / f"{split}.jsonl"
        out_path = args.output / f"{split}.jsonl"
        with in_path.open() as src, out_path.open("w") as dst:
            for line in src:
                row = json.loads(line)
                tokens = int(row.get("token_count", 0))
                max_seen = max(max_seen, tokens)
                if tokens <= args.max_tokens:
                    dst.write(json.dumps(row, ensure_ascii=False) + "\n")
                    kept += 1
                else:
                    dropped += 1
        summary["splits"][split] = {
            "kept": kept,
            "dropped_from_16k_artifact": dropped,
            "max_seen_before_cap": max_seen,
        }
    with (args.output / "manifest.json").open("w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
