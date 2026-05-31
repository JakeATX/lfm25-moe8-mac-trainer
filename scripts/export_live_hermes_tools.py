#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


DEFAULT_HERMES_REPO = Path("/Users/jkooker/.hermes/hermes-agent")
DEFAULT_PLATFORM = "cli"


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


def load_platform_toolsets(platform: str) -> list[str]:
    from hermes_cli.config import load_config  # type: ignore
    from hermes_cli.tools_config import _get_platform_tools  # type: ignore

    return sorted(_get_platform_tools(load_config(), platform))


def export_tools(hermes_repo: Path, platform: str, toolsets: list[str] | None) -> tuple[list[str], list[dict[str, Any]]]:
    sys.path.insert(0, str(hermes_repo))
    from model_tools import get_tool_definitions  # type: ignore

    resolved_toolsets = sorted(toolsets or load_platform_toolsets(platform))
    tools = get_tool_definitions(
        enabled_toolsets=resolved_toolsets,
        disabled_toolsets=[],
        quiet_mode=True,
    )

    # Some Hermes installs gate computer_use behind runtime requirements during
    # schema discovery, while the live CLI may still expose it after daemon
    # bootstrap. For training/eval we need the exact schema, so inject the
    # canonical Hermes schema when the toolset is active but discovery omits it.
    names = {tool["function"]["name"] for tool in tools}
    if "computer_use" in resolved_toolsets and "computer_use" not in names:
        try:
            from tools.computer_use.schema import get_computer_use_schema  # type: ignore

            tools.append({"type": "function", "function": get_computer_use_schema()})
        except Exception as exc:
            print(f"warning: could not inject computer_use schema: {exc}", file=sys.stderr)

    return resolved_toolsets, sorted(tools, key=lambda item: item["function"]["name"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the active installed Hermes tool surface.")
    parser.add_argument("--hermes-repo", type=Path, default=DEFAULT_HERMES_REPO)
    parser.add_argument("--platform", default=DEFAULT_PLATFORM)
    parser.add_argument("--toolsets", nargs="*", default=None)
    parser.add_argument("--names", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    hermes_repo = args.hermes_repo.resolve()
    toolsets, tools = export_tools(hermes_repo, args.platform, args.toolsets)
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
            "platform": args.platform,
            "toolsets": toolsets,
            "filtered_names": args.names,
        },
        "tool_count": len(tools),
        "tool_names": [tool["function"]["name"] for tool in tools],
        "tools": tools,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"tool_count": payload["tool_count"], "tool_names": payload["tool_names"]}, indent=2))


if __name__ == "__main__":
    main()
