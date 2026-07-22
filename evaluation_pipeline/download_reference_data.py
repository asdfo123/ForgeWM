#!/usr/bin/env python3
"""Download the frozen ForgeWM reference inputs from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


REPO_ID = "ForgeWM/ForgeWM-GT-1000"
REVISION = "5a7071587e641ad130c7dbf6018cdb81631af9ac"


def main() -> None:
    pipeline_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--output", type=Path, default=pipeline_dir / "data/reference")
    parser.add_argument(
        "--token", default=None,
        help="HF token; normally omit this and run `huggingface-cli login` first.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.revision,
        local_dir=args.output,
        token=args.token,
    )
    print(f"Downloaded {args.repo_id}@{args.revision} to {result}")


if __name__ == "__main__":
    main()
