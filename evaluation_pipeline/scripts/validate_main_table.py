#!/usr/bin/env python3
"""Validate the complete three-model input bundle for the paper main table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PAIRED_MODELS = ("forgewm", "mg2", "hyworld")
CONSTANT_MODELS = ("forgewm", "mg2", "hyworld_aligned")


def tsv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def validate_generation(path: Path, expected: int) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "status": "valid",
        "successful": expected,
        "frame_counts": {"77": expected},
        "resolutions": {"640x352": expected},
        "fps": {"12.0": expected},
    }
    for key, value in required.items():
        if payload.get(key) != value:
            raise RuntimeError(f"Invalid {path}: {key}={payload.get(key)!r}, expected {value!r}")


def count_files(path: Path, pattern: str) -> int:
    return sum(1 for item in path.glob(pattern) if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--gt-manifest", type=Path, required=True)
    args = parser.parse_args()

    paired = args.main_root / "paired_full1000"
    constant = args.main_root / "constant_action"
    gt_rows = tsv_rows(args.gt_manifest)
    if len(gt_rows) != 1000:
        raise RuntimeError(f"GT manifest has {len(gt_rows)} rows, expected 1000")

    report: dict[str, object] = {"gt": len(gt_rows), "paired": {}, "constant": {}}
    for model in PAIRED_MODELS:
        model_root = paired / model
        validate_generation(model_root / "validation.json", 1000)
        rows = tsv_rows(model_root / "manifest.tsv")
        videos = count_files(model_root / "eval_videos", "*.mp4")
        images = count_files(model_root / "images", "*.png")
        if (len(rows), videos, images) != (1000, 1000, 1000):
            raise RuntimeError(
                f"Incomplete paired {model}: manifest={len(rows)} videos={videos} images={images}"
            )
        report["paired"][model] = {"manifest": len(rows), "videos": videos, "images": images}

    actions = {"forward", "back", "left", "right", "turn_left", "turn_right"}
    for model in CONSTANT_MODELS:
        model_root = constant / model
        rows = tsv_rows(model_root / "manifest.tsv")
        video_dir = model_root / ("eval_videos" if model == "hyworld_aligned" else "videos")
        videos = count_files(video_dir, "*.mp4")
        images = count_files(model_root / "images", "*.png")
        found_actions = {row["action"] for row in rows}
        scenes = {row["scene"] for row in rows}
        if (len(rows), videos, images, len(scenes)) != (462, 462, 462, 77):
            raise RuntimeError(
                f"Incomplete constant {model}: manifest={len(rows)} videos={videos} "
                f"images={images} scenes={len(scenes)}"
            )
        if found_actions != actions:
            raise RuntimeError(f"Unexpected actions for {model}: {sorted(found_actions)}")
        report["constant"][model] = {
            "manifest": len(rows), "videos": videos, "images": images, "scenes": len(scenes)
        }

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
