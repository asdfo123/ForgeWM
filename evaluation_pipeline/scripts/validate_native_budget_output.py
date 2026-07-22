#!/usr/bin/env python3
"""Fail unless a native-budget validation artifact proves full completion."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()
    if not args.validation.is_file():
        raise FileNotFoundError(args.validation)
    payload = json.loads(args.validation.read_text(encoding="utf-8"))
    if payload.get("status") != "valid" or payload.get("successful") != args.expected:
        raise RuntimeError(f"Invalid completion artifact: {payload}")
    if payload.get("frame_counts") != {"77": args.expected}:
        raise RuntimeError(f"Unexpected frame counts: {payload.get('frame_counts')}")
    if payload.get("resolutions") != {"640x352": args.expected}:
        raise RuntimeError(f"Unexpected resolutions: {payload.get('resolutions')}")
    if payload.get("fps") != {"12.0": args.expected}:
        raise RuntimeError(f"Unexpected FPS metadata: {payload.get('fps')}")
    print(f"valid: {args.validation} ({args.expected})")


if __name__ == "__main__":
    main()
