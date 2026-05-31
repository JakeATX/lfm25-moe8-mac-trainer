#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--message", default="Upload artifacts")
    args = parser.parse_args()
    create_repo(args.repo_id, repo_type=args.repo_type, private=args.private, exist_ok=True)
    upload_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=str(args.local_dir),
        commit_message=args.message,
    )
    print(f"Uploaded {args.local_dir} to {args.repo_type}:{args.repo_id}")


if __name__ == "__main__":
    main()
