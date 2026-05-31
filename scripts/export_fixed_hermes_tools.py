#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_TOOLSETS = ["terminal_tools", "file_tools", "browser_tools"]


def git_commit(repo: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def export_tools(hermes_repo: Path, toolsets: list[str]) -> list[dict[str, Any]]:
    sys.path.insert(0, str(hermes_repo))
    from model_tools import get_tool_definitions  # type: ignore

    tools = get_tool_definitions(
        enabled_toolsets=toolsets,
        disabled_toolsets=[],
        quiet_mode=True,
    )
    return sorted(tools, key=lambda item: item["function"]["name"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hermes-repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--toolsets", nargs="+", default=DEFAULT_TOOLSETS)
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help="Optional exact tool-name allowlist after Hermes toolset resolution.",
    )
    args = parser.parse_args()

    hermes_repo = args.hermes_repo.resolve()
    tools = export_tools(hermes_repo, args.toolsets)
    if args.names:
        wanted = set(args.names)
        tools = [tool for tool in tools if tool["function"]["name"] in wanted]
        missing = wanted - {tool["function"]["name"] for tool in tools}
        if missing:
            raise SystemExit(f"Missing requested Hermes tools: {sorted(missing)}")

    payload = {
        "source": {
            "hermes_repo": str(hermes_repo),
            "git_commit": git_commit(hermes_repo),
            "toolsets": args.toolsets,
            "filtered_names": args.names,
        },
        "tool_count": len(tools),
        "tool_names": [tool["function"]["name"] for tool in tools],
        "tools": tools,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"tool_count": len(tools), "tool_names": payload["tool_names"]}, indent=2))


if __name__ == "__main__":
    main()
