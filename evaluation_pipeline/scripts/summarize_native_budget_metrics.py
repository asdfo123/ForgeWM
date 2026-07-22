#!/usr/bin/env python3
"""Validate and summarize ForgeWM-1/2 with the paper main-table protocol."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = ("forgewm1", "forgewm2")
LABELS = {"forgewm1": "ForgeWM-1", "forgewm2": "ForgeWM-2"}
TRANSLATIONS = {"forward", "back", "left", "right"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_vbench(model_dir: Path) -> dict:
    paths = sorted(glob.glob(str(model_dir / "eval" / "*eval_results.json")))
    if not paths:
        raise RuntimeError(f"No VBench result under {model_dir}")
    payload = json.loads(Path(paths[-1]).read_text(encoding="utf-8"))
    for key in ("imaging_quality", "aesthetic_quality"):
        if key not in payload or len(payload[key]) < 2 or len(payload[key][1]) != 462:
            count = len(payload.get(key, [None, []])[1])
            raise RuntimeError(f"Incomplete VBench {model_dir.name}/{key}: {count}")
    return payload


def collect_shards(directory: Path) -> list[dict[str, str]]:
    output = []
    for path in sorted(directory.glob("rank*.tsv")):
        output.extend(rows(path))
    return output


def kctrl(trajectory_rows: list[dict]) -> float:
    index = {
        (row["scene"], int(row.get("seed", 0)), row["action"]): row
        for row in trajectory_rows
        if row["action"] in TRANSLATIONS
    }
    by_scene: dict[str, list[float]] = defaultdict(list)
    for scene, seed in sorted({(key[0], key[1]) for key in index}):
        for first, second in (("forward", "back"), ("left", "right")):
            a = index.get((scene, seed, first))
            b = index.get((scene, seed, second))
            if a is not None and b is not None:
                by_scene[scene].append(
                    float(float(a["net_direction_correct"]) > 0.5)
                    * float(float(b["net_direction_correct"]) > 0.5)
                )
    if len(by_scene) != 77 or any(len(value) != 2 for value in by_scene.values()):
        raise RuntimeError(f"Incomplete KCtrl pairs: scenes={len(by_scene)}")
    return float(np.mean([np.mean(value) for value in by_scene.values()]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("output/native_budget_eval")
    )
    args = parser.parse_args()
    root = args.root
    metric_root = root / "metrics"
    temporal_rows = rows(metric_root / "full1000/paired_temporal_action_metrics.tsv")

    summary = {
        "protocol": {
            "full1000": 1000,
            "constant_action_scenes": 77,
            "constant_action_videos": 462,
            "frames": 77,
            "resolution": "640x352",
            "fps": 12,
            "seed": 0,
            "first_chunk_nfe": 4,
            "steady_schedules": {
                "forgewm1": [1000],
                "forgewm2": [1000, 250],
            },
        },
        "models": {},
    }
    for model in MODELS:
        vbench = latest_vbench(root / "constant" / model)
        perceptual = collect_shards(metric_root / "full1000/perceptual" / model)
        if len(perceptual) != 1000:
            raise RuntimeError(f"Incomplete perceptual {model}: {len(perceptual)}")
        custom = rows(metric_root / "full1000/custom_metrics" / model / "per_video_metrics.tsv")
        if len(custom) != 1000:
            raise RuntimeError(f"Incomplete custom metrics {model}: {len(custom)}")
        temporal = [row for row in temporal_rows if row["model"] == model]
        if len(temporal) != 1000:
            raise RuntimeError(f"Incomplete temporal metrics {model}: {len(temporal)}")
        trajectory_payload = json.loads(
            (metric_root / "constant/action_trajectory" / model / "action_consistency.json")
            .read_text(encoding="utf-8")
        )
        trajectory = trajectory_payload["per_video"]
        if len(trajectory) != 462:
            raise RuntimeError(f"Incomplete trajectory {model}: {len(trajectory)}")
        idm = json.loads(
            (metric_root / "constant/action_idm" / model / "summary.json")
            .read_text(encoding="utf-8")
        )
        if idm["successful_videos"] != 462:
            raise RuntimeError(f"Incomplete IDM {model}: {idm['successful_videos']}")

        imaging = float(vbench["imaging_quality"][0])
        if imaging > 1.0:
            imaging /= 100.0
        summary["models"][model] = {
            "label": LABELS[model],
            "imaging_quality": imaging,
            "lpips_gt": float(np.mean([float(row["lpips_gt"]) for row in perceptual])),
            "aesthetic_quality": float(vbench["aesthetic_quality"][0]),
            "flow_profile_cosine": float(
                np.mean([float(row["flow_profile_cosine"]) for row in temporal])
            ),
            "depth_gt": float(np.mean([float(row["depth_gt"]) for row in custom])),
            "action_sign_accuracy": float(np.nanmean([
                float(row["action_signature_sign_accuracy"]) for row in temporal
            ])),
            "kctrl": kctrl(trajectory),
            "mouse_accuracy": float(idm["mouse_accuracy"]),
            "counts": {
                "vbench": 462,
                "perceptual": len(perceptual),
                "custom": len(custom),
                "temporal": len(temporal),
                "trajectory": len(trajectory),
                "idm": idm["successful_videos"],
            },
        }

    metric_root.mkdir(parents=True, exist_ok=True)
    (metric_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Native-budget metrics (paper main-table protocol)",
        "",
        "| Model | First/steady NFE | Imaging Quality ↑ | LPIPS ↓ | Aesthetic Quality ↑ | Flow Prof. ↑ | Depth ↑ | Sign Acc. ↑ | KCtrl ↑ | Mouse Acc. ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        value = summary["models"][model]
        lines.append(
            f"| {value['label']} | 4/{1 if model == 'forgewm1' else 2} | "
            f"{value['imaging_quality']:.4f} | {value['lpips_gt']:.4f} | "
            f"{value['aesthetic_quality']:.4f} | {value['flow_profile_cosine']:.4f} | "
            f"{value['depth_gt']:.4f} | {value['action_sign_accuracy']:.4f} | "
            f"{value['kctrl']:.4f} | {value['mouse_accuracy']:.4f} |"
        )
    (metric_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
