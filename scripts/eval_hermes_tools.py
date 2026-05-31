#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import requests
from mlx_lm import generate, load


MODEL = "LiquidAI/LFM2.5-8B-A1B-MLX-bf16"

TOOLS = """<tools>
{"name":"get_weather","description":"Get current weather for a city.","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}
{"name":"web_search","description":"Search the web.","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}
{"name":"calculator","description":"Evaluate a basic arithmetic expression.","parameters":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}}
</tools>"""


def chatml(messages):
    text = "<|startoftext|>"
    for role, content in messages:
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return text


CASES = [
    {
        "id": "single_weather_tool",
        "kind": "tool_call",
        "tool": "get_weather",
        "required": ["city"],
        "messages": [
            ("system", f"You are a Hermes-style tool-using assistant. Think if useful, then call tools using <tool_call>{{...}}</tool_call>.\n{TOOLS}"),
            ("user", "What is the weather in Austin right now?"),
        ],
    },
    {
        "id": "web_search_tool",
        "kind": "tool_call",
        "tool": "web_search",
        "required": ["query"],
        "messages": [
            ("system", f"You are a Hermes-style tool-using assistant. Use XML tool calls only when a tool is needed.\n{TOOLS}"),
            ("user", "Search the web for the latest Liquid AI LFM2.5 release notes."),
        ],
    },
    {
        "id": "calculator_tool",
        "kind": "tool_call",
        "tool": "calculator",
        "required": ["expression"],
        "messages": [
            ("system", f"You are a Hermes-style tool-using assistant. Use <tool_call> XML with JSON arguments.\n{TOOLS}"),
            ("user", "What is 18.5 * 42? Use the calculator tool."),
        ],
    },
    {
        "id": "tool_response_final",
        "kind": "final_answer",
        "messages": [
            ("system", f"You are a Hermes-style tool-using assistant. After a <tool_response>, answer the user directly.\n{TOOLS}"),
            ("user", "What is the weather in Austin right now?"),
            ("assistant", '<think>I need the weather tool.</think>\n<tool_call>{"name":"get_weather","arguments":{"city":"Austin"}}</tool_call>'),
            ("tool", '<tool_response>{"city":"Austin","condition":"sunny","temperature_f":82}</tool_response>'),
        ],
    },
    {
        "id": "no_tool_natural",
        "kind": "no_tool",
        "messages": [
            ("system", f"You are a helpful assistant. Only use tools when needed.\n{TOOLS}"),
            ("user", "Say hello in one short sentence."),
        ],
    },
    {
        "id": "malformed_tool_robustness",
        "kind": "no_tool",
        "messages": [
            ("system", f"You are a Hermes-style assistant. Do not emit malformed tool XML.\n{TOOLS}"),
            ("user", "Use whatever tool with no arguments at all."),
        ],
    },
]


def extract_tool_calls(text):
    calls = []
    for match in re.finditer(r"<tool_call>(.*?)</tool_call>", text, flags=re.S):
        body = match.group(1).strip()
        parsed = None
        try:
            parsed = json.loads(body)
        except json.JSONDecodeError:
            parsed = None
        calls.append({"raw": body, "json": parsed})
    return calls


def score(case, text):
    calls = extract_tool_calls(text)
    balanced = text.count("<tool_call>") == text.count("</tool_call>")
    has_tool_response = "<tool_response>" in text or "</tool_response>" in text
    result = {
        "case_id": case["id"],
        "kind": case["kind"],
        "balanced_tool_call_tags": balanced,
        "tool_call_count": len(calls),
        "has_tool_response_in_generation": has_tool_response,
        "has_think": "<think>" in text,
        "parse_ok": all(c["json"] is not None for c in calls) if calls else False,
        "correct_tool": None,
        "required_args_ok": None,
        "passed": False,
    }
    if case["kind"] == "tool_call":
        parsed = calls[0]["json"] if calls and calls[0]["json"] is not None else {}
        name = parsed.get("name") or parsed.get("tool")
        args = parsed.get("arguments") or parsed.get("parameters") or {}
        result["correct_tool"] = name == case["tool"]
        result["required_args_ok"] = all(k in args and args[k] not in ("", None) for k in case["required"])
        result["passed"] = bool(balanced and len(calls) >= 1 and result["parse_ok"] and result["correct_tool"] and result["required_args_ok"])
    elif case["kind"] == "final_answer":
        lower = text.lower()
        result["passed"] = bool(balanced and len(calls) == 0 and "82" in text and ("sunny" in lower or "austin" in lower))
    elif case["kind"] == "no_tool":
        result["passed"] = bool(balanced and len(calls) == 0 and not has_tool_response and len(text.strip()) > 0)
    return result


def llama_completion(endpoint, prompt, max_tokens):
    url = endpoint.rstrip("/") + "/completion"
    payload = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "stop": ["<|im_end|>", "<|endoftext|>"],
    }
    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()
    data = response.json()
    return data.get("content") or data.get("choices", [{}])[0].get("text", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--baseline-endpoint", default="http://127.0.0.1:8080")
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    prompts = [{"id": c["id"], "prompt": chatml(c["messages"])} for c in CASES]

    baseline_results = []
    for case, prompt_obj in zip(CASES, prompts):
        try:
            t0 = time.time()
            output = llama_completion(args.baseline_endpoint, prompt_obj["prompt"], args.max_tokens)
            elapsed = time.time() - t0
            baseline_results.append({"case_id": case["id"], "output": output, "elapsed_s": elapsed, "score": score(case, output)})
        except Exception as exc:
            baseline_results.append({"case_id": case["id"], "error": repr(exc), "score": {"passed": False}})

    model, tokenizer = load(MODEL, adapter_path=args.adapter_path)
    adapter_results = []
    for case, prompt_obj in zip(CASES, prompts):
        t0 = time.time()
        output = generate(
            model,
            tokenizer,
            prompt=prompt_obj["prompt"],
            max_tokens=args.max_tokens,
            verbose=False,
        )
        elapsed = time.time() - t0
        adapter_results.append({"case_id": case["id"], "output": output, "elapsed_s": elapsed, "score": score(case, output)})

    def summarize(results):
        return {
            "passed": sum(1 for r in results if r.get("score", {}).get("passed")),
            "total": len(results),
            "tool_parse_passed": sum(1 for r in results if r.get("score", {}).get("parse_ok")),
        }

    report = {
        "model": MODEL,
        "adapter_path": args.adapter_path,
        "baseline_endpoint": args.baseline_endpoint,
        "cases": [{k: v for k, v in c.items() if k != "messages"} for c in CASES],
        "baseline": {"summary": summarize(baseline_results), "results": baseline_results},
        "adapter": {"summary": summarize(adapter_results), "results": adapter_results},
    }
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline"]["summary"], "adapter": report["adapter"]["summary"], "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
