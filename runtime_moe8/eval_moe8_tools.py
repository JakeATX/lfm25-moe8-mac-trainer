#!/usr/bin/env python3
"""Evaluate a direct MLX checkpoint on the fixed Hermes tool mini-suite."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from mlx_lm import generate, load

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lfm25_hermes_ft.scripts.eval_hermes_tools import CASES, chatml, score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load(args.model)
    results = []
    for case in CASES:
        prompt = chatml(case["messages"])
        start = time.time()
        output = generate(
            model,
            tokenizer,
            prompt=prompt,
            max_tokens=args.max_tokens,
            verbose=False,
        )
        results.append(
            {
                "case_id": case["id"],
                "output": output,
                "elapsed_s": time.time() - start,
                "score": score(case, output),
            }
        )

    summary = {
        "passed": sum(1 for r in results if r["score"].get("passed")),
        "total": len(results),
        "tool_parse_passed": sum(1 for r in results if r["score"].get("parse_ok")),
        "tool_call_cases_passed": sum(
            1
            for r, c in zip(results, CASES)
            if c["kind"] == "tool_call" and r["score"].get("passed")
        ),
        "no_tool_or_final_cases_passed": sum(
            1
            for r, c in zip(results, CASES)
            if c["kind"] != "tool_call" and r["score"].get("passed")
        ),
    }
    report = {
        "model": args.model,
        "cases": [{k: v for k, v in c.items() if k != "messages"} for c in CASES],
        "summary": summary,
        "results": results,
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
