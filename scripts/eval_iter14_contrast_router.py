#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import live_hermes_eval  # noqa: E402


def google(query: str) -> str:
    return "https://www.google.com/search?q=" + query.replace(" ", "+")


def build_cases(tool_names: set[str]) -> list[live_hermes_eval.Case]:
    cases: list[live_hermes_eval.Case] = []

    def add(case: live_hermes_eval.Case) -> None:
        if case.expect_tool is None or case.expect_tool in tool_names:
            cases.append(case)

    browser_prompts = [
        ("browser_lfm_plain", "search for LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("browser_lfm_lookup", "look up LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("browser_lfm_web", "search the web for LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("browser_weather_austin", "what's the weather in Austin today", "Austin weather today"),
        ("browser_weather_current", "check the current weather in Austin TX", "current weather Austin TX"),
        ("browser_weather_plain", "weather in Austin right now", "Austin weather right now"),
        ("browser_noaa", "find the latest NOAA forecast for Austin", "NOAA Austin forecast today"),
        ("browser_online_docs", "check online for current mlx-lm server tool calling docs", "current mlx-lm server tool calling docs"),
        ("browser_latest_llamacpp", "find the latest llama.cpp release notes", "latest llama.cpp release notes"),
        ("browser_unsloth", "search online for Unsloth LFM2.5 GGUF quants", "Unsloth LFM2.5 GGUF quants"),
        ("browser_time_tokyo", "what's the current time in Tokyo", "current time Tokyo"),
        ("browser_hf_model", "look online for sjakek LFM-2.5 Hermes tuned model", "sjakek LFM-2.5 Hermes tuned model"),
        ("browser_current_liquid", "find current Liquid AI model information", "Liquid AI current model information"),
        ("browser_nba", "search the web for today's NBA news", "today NBA news"),
        ("browser_python_release", "search for the latest Python release", "latest Python release"),
        ("browser_weather_radar", "open a browser search for weather radar Austin", "weather radar Austin"),
        ("browser_satx", "check Google for current weather in San Antonio", "current weather San Antonio"),
        ("browser_mlx_release", "can you check online for the latest MLX release", "latest MLX release"),
        ("browser_open_google", "navigate the browser to google.com", "https://www.google.com"),
        ("browser_open_hf", "open Hugging Face", "https://huggingface.co"),
    ]
    for case_id, prompt, query_or_url in browser_prompts:
        url = query_or_url if query_or_url.startswith("https://") else google(query_or_url)
        add(live_hermes_eval.Case(case_id, "iter14_browser_general_web", prompt, "browser_navigate", {"url": url}, ("url",)))

    x_prompts = [
        ("x_lfm_plain", "search X for LFM2.5 GGUF", "LFM2.5 GGUF"),
        ("x_lfm_recent", "look on X for recent posts about LFM2.5", "recent posts about LFM2.5"),
        ("x_liquid", "check X for recent Liquid AI posts", "Liquid AI"),
        ("x_mlx", "search Twitter/X for MLX LFM2.5", "MLX LFM2.5"),
        ("x_llamacpp", "find current X reactions to llama.cpp", "llama.cpp reactions"),
        ("x_hermes", "look up what people on X are saying about Hermes agent", "Hermes agent"),
        ("x_sjakek", "search X for sjakek model posts", "sjakek model"),
        ("x_apple_mlx", "check Twitter for current Apple MLX discussion", "Apple MLX discussion"),
        ("x_unsloth", "find X threads about Unsloth dynamic GGUF quants", "Unsloth dynamic GGUF quants"),
        ("x_tool_calling", "look for recent X chatter about tool calling", "tool calling"),
        ("x_austin_weather", "find X reactions to today's Austin weather", "Austin weather"),
        ("x_release", "search X for recent llama.cpp release discussion", "llama.cpp release"),
        ("x_hf_gguf", "check X for posts about Hugging Face GGUF uploads", "Hugging Face GGUF uploads"),
        ("x_liquid_handle", "search X posts from @liquid_ai about LFM2.5", "LFM2.5"),
        ("x_hf_handle", "check X posts from @huggingface about GGUF", "GGUF"),
        ("x_twitter_word", "search Twitter for current discussion of Apollo 13", "Apollo 13"),
        ("x_social_current", "find social posts on X about current AI agent tooling", "AI agent tooling"),
        ("x_reactions", "what are people saying on X about LFM2.5", "LFM2.5"),
        ("x_thread", "look for X threads about local GGUF inference", "local GGUF inference"),
        ("x_posts", "search posts on X about mlx-lm tool calling", "mlx-lm tool calling"),
    ]
    for case_id, prompt, query in x_prompts:
        add(live_hermes_eval.Case(case_id, "iter14_x_social_search", prompt, "x_search", {"query": query}, ("query",)))

    file_prompts = [
        ("file_lfm", "search this repo for LFM2.5 GGUF", {"pattern": "LFM2.5 GGUF"}),
        ("file_env", "search this repo for HERMES_PARSE_TEXT_TOOL_CALLS", {"pattern": "HERMES_PARSE_TEXT_TOOL_CALLS"}),
        ("file_browser_nav", "find browser_navigate in local files", {"pattern": "browser_navigate"}),
        ("file_iter14", "search local files for iter14", {"pattern": "iter14"}),
        ("file_parser", "search my repo for tool_parser_type", {"pattern": "tool_parser_type"}),
        ("file_x_search", "find where x_search is mentioned in the project", {"pattern": "x_search"}),
        ("file_live_eval", "search the codebase for live_hermes_eval", {"pattern": "live_hermes_eval"}),
        ("file_live_eval_name", "find Python files named live_hermes_eval", {"pattern": "live_hermes_eval", "target": "files"}),
        ("file_contrast", "find files matching *contrast* in the repo", {"pattern": "*contrast*", "target": "files"}),
        ("file_local_not_web", "search local files, not the web, for browser_navigate", {"pattern": "browser_navigate"}),
        ("file_runtime", "look in this repo for model_runtime_step01746_pythonic", {"pattern": "model_runtime_step01746_pythonic"}),
        ("file_lora", "search source files for run_lora_repair", {"pattern": "run_lora_repair"}),
        ("file_q6", "find docs mentioning Q6KXL", {"pattern": "Q6KXL"}),
        ("file_llamacpp", "search my local files for llama.cpp", {"pattern": "llama.cpp"}),
        ("file_tool_surface", "search this workspace for live_hermes_cli_tools", {"pattern": "live_hermes_cli_tools"}),
    ]
    for case_id, prompt, args in file_prompts:
        add(live_hermes_eval.Case(case_id, "iter14_local_file_search", prompt, "search_files", args, tuple(args.keys())))

    no_tool = [
        ("nt_search_word", "search means look carefully; explain the word, do not use tools", "search"),
        ("nt_lfm_explain", "explain what LFM2.5 GGUF means; do not search", "GGUF"),
        ("nt_x_concept", "what does x_search do conceptually? Do not call it.", "X"),
        ("nt_browser", "what is a browser in one sentence?", "website"),
        ("nt_terminal", "what is a terminal in one sentence?", "command"),
        ("nt_local_search", "what is local file search?", "file"),
        ("nt_weather_concept", "do not use tools; what's the weather as a concept?", "atmosphere"),
        ("nt_pwd", "do not use tools; what command prints the current folder?", "pwd"),
        ("nt_hello", "hello", "hello"),
        ("nt_are_you_there", "are you there?", "here"),
        ("nt_apollo13", "what happened on Apollo 13?", "Apollo"),
        ("nt_moon_disaster", "which moon landing was nearly a disaster?", "Apollo 13"),
        ("nt_moon_detail", "tell me about the moon landing in detail", "1969"),
        ("nt_json", "return a tiny JSON object with status ok", "ok"),
        ("nt_browser_word", "write a sentence using the word browser", "browser"),
        ("nt_search_sentence", "write a sentence using the word search", "search"),
        ("nt_math", "without tools, what is 12 times 13?", "156"),
        ("nt_tool_call", "explain what a tool call is without making one", "tool"),
        ("nt_no_open", "do not open anything; just say ready", "ready"),
        ("nt_quant", "define quantization in one paragraph", "precision"),
    ]
    for case_id, prompt, contains in no_tool:
        add(live_hermes_eval.Case(case_id, "iter14_no_tool_retention", prompt, content_contains=contains))

    corrections = [
        ("corr_web_not_x", "no, search the web, not X", "browser_navigate", {"url": google("LFM2.5 GGUF")}),
        ("corr_online_not_local", "that was local files, I meant search online", "browser_navigate", {"url": google("LFM2.5 GGUF")}),
        ("corr_weather_browser", "yes use browser tools to check the weather", "browser_navigate", {"url": google("Austin weather today")}),
        ("corr_x_specific", "no, search X specifically", "x_search", {"query": "LFM2.5 GGUF"}),
        ("corr_twitter_not_web", "I mean Twitter/X, not the web", "x_search", {"query": "LFM2.5 GGUF"}),
        ("corr_repo_instead", "no, search this repo instead", "search_files", {"pattern": "LFM2.5 GGUF"}),
        ("corr_local_not_online", "search local files, not online", "search_files", {"pattern": "browser_navigate"}),
        ("corr_browser_retry", "that didn't work, try a normal browser search", "browser_navigate", {"url": google("Austin weather today")}),
    ]
    for case_id, prompt, name, args in corrections[:5]:
        add(live_hermes_eval.Case(case_id, "iter14_correction_boundary", prompt, name, args, tuple(args.keys())))

    if len(cases) != 80:
        raise RuntimeError(f"Iter14 focused suite expected exactly 80 cases, got {len(cases)}")
    return cases


def summarize_with_confusion(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = live_hermes_eval.summarize(results)
    expected_tools = ["browser_navigate", "x_search", "search_files", None]
    confusion: dict[str, dict[str, int]] = {str(expected): {} for expected in expected_tools}
    for row in results:
        expected = row.get("expected_tool")
        actual = row.get("tool_name")
        if not expected and row.get("tool_calls"):
            actual = row.get("tool_calls", [{}])[0].get("function", {}).get("name")
        key = str(expected)
        actual_key = str(actual)
        confusion.setdefault(key, {})
        confusion[key][actual_key] = confusion[key].get(actual_key, 0) + 1
    cats = summary["category_metrics"]
    browser_rows = [r for r in results if r["category"] == "iter14_browser_general_web"]
    browser_to_x = sum(1 for r in browser_rows if r.get("tool_name") == "x_search")
    browser_to_files = sum(1 for r in browser_rows if r.get("tool_name") == "search_files")
    summary["iter14"] = {
        "confusion_matrix": confusion,
        "browser_to_x_search_rate": round(browser_to_x / len(browser_rows), 4) if browser_rows else 0,
        "browser_to_search_files_rate": round(browser_to_files / len(browser_rows), 4) if browser_rows else 0,
        "browser_pass_rate": cats.get("iter14_browser_general_web", {}).get("rate", 0),
        "x_search_pass_rate": cats.get("iter14_x_social_search", {}).get("rate", 0),
        "search_files_pass_rate": cats.get("iter14_local_file_search", {}).get("rate", 0),
        "no_tool_pass_rate": cats.get("iter14_no_tool_retention", {}).get("rate", 0),
    }
    summary["iter14_acceptance"] = {
        "browser_pass_gte_0_90": summary["iter14"]["browser_pass_rate"] >= 0.90,
        "x_search_pass_gte_0_90": summary["iter14"]["x_search_pass_rate"] >= 0.90,
        "search_files_pass_gte_0_90": summary["iter14"]["search_files_pass_rate"] >= 0.90,
        "no_tool_false_positive_lte_0_05": summary["no_tool_false_positive_rate"] <= 0.05,
        "browser_to_x_lte_0_05": summary["iter14"]["browser_to_x_search_rate"] <= 0.05,
        "browser_to_files_lte_0_05": summary["iter14"]["browser_to_search_files_rate"] <= 0.05,
        "zero_text_tool_leaks": summary["text_tool_leaks"] == 0,
        "zero_invented_tool_names": summary["invented_tool_names"] == 0,
    }
    return summary


def write_report(path: Path, name: str, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [f"# Iter14 Contrast Router Eval: {name}", "", "## Summary", ""]
    for key in [
        "passed",
        "total",
        "overall_rate",
        "valid_structured_tool_rate",
        "no_tool_false_positive_rate",
        "text_tool_leaks",
        "invented_tool_names",
    ]:
        lines.append(f"- `{key}`: `{summary.get(key)}`")
    lines.extend(["", "## Iter14 Metrics", ""])
    for key, value in summary["iter14"].items():
        if key != "confusion_matrix":
            lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Confusion Matrix", ""])
    for expected, actuals in sorted(summary["iter14"]["confusion_matrix"].items()):
        lines.append(f"- expected `{expected}`: `{actuals}`")
    lines.extend(["", "## Acceptance", ""])
    for key, value in summary["iter14_acceptance"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Failures", ""])
    failures = [r for r in results if not r["passed"]]
    if not failures:
        lines.append("No failures.")
    else:
        for row in failures:
            text = (row.get("assistant_visible_text") or "").replace("\n", " ")[:220]
            lines.append(
                f"- `{row['case_id']}` `{row['category']}` `{row.get('failure')}` expected `{row.get('expected_tool')}` got `{row.get('tool_name')}`: {text}"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Focused direct API eval for browser-vs-X-vs-local-file routing.")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tools-json", type=Path, required=True)
    parser.add_argument("--out-jsonl", type=Path, required=True)
    parser.add_argument("--out-report", type=Path, required=True)
    parser.add_argument("--adapter-path")
    parser.add_argument("--name", default="iter14_contrast_router")
    parser.add_argument("--allow-fail", action="store_true")
    args = parser.parse_args()

    tool_payload = json.loads(args.tools_json.read_text(encoding="utf-8"))
    tools = tool_payload["tools"]
    cases = build_cases({tool["function"]["name"] for tool in tools})
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with args.out_jsonl.open("w", encoding="utf-8") as handle:
        for idx, case in enumerate(cases, start=1):
            row = live_hermes_eval.evaluate_case(args.endpoint, args.model, tools, case, args.adapter_path)
            row["index"] = idx
            row["case"] = asdict(case)
            results.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"index": idx, "case_id": row["case_id"], "passed": row["passed"], "failure": row.get("failure")}))
    summary = summarize_with_confusion(results)
    args.out_jsonl.with_suffix(".summary.json").write_text(json.dumps({"summary": summary}, indent=2), encoding="utf-8")
    write_report(args.out_report, args.name, summary, results)
    print(json.dumps(summary, indent=2))
    accepted = all(summary["iter14_acceptance"].values())
    if not args.allow_fail and not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
