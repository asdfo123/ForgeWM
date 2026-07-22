#!/usr/bin/env python3
"""Merge single-dimension VBench runs for the aligned 77-frame suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


MODELS = ("forgewm", "mg2", "hyworld_aligned")
DIMENSIONS = ("aesthetic_quality", "imaging_quality")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    for model in MODELS:
        eval_dir = args.root / model / "eval"
        found: dict[str, object] = {}
        sources: dict[str, str] = {}
        for path in sorted(eval_dir.glob("*eval_results.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for dimension in DIMENSIONS:
                if dimension not in payload:
                    continue
                value = payload[dimension]
                if len(value) != 2 or len(value[1]) != 462:
                    raise RuntimeError(
                        f"Incomplete {model}/{dimension} in {path}: "
                        f"{len(value[1]) if len(value) == 2 else 'invalid'}"
                    )
                if dimension in found:
                    raise RuntimeError(f"Duplicate {model}/{dimension}: {path}")
                found[dimension] = value
                sources[dimension] = path.name
        missing = sorted(set(DIMENSIONS) - set(found))
        if missing:
            raise RuntimeError(f"Missing {model} dimensions: {missing}")

        output = eval_dir / "aligned77_eval_results.json"
        payload = {
            **found,
            "_protocol": {
                "frames": 77,
                "fps": 12,
                "resolution": "640x352",
                "videos": 462,
                "sources": sources,
            },
        }
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(
            f"{model}: aesthetic={found['aesthetic_quality'][0]:.10f} "
            f"imaging={found['imaging_quality'][0]:.10f}"
        )


if __name__ == "__main__":
    main()
