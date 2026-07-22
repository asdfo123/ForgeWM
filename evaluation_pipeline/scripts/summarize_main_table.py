#!/usr/bin/env python3
"""Reproduce the paper's three-model main table from frozen metric outputs."""

from __future__ import annotations

import argparse
import csv
import glob
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


MODELS = ("forgewm", "mg2", "hyworld")
LABELS = {
    "forgewm": "ForgeWM",
    "mg2": "Matrix-Game 2.0",
    "hyworld": "HY-WorldPlay",
}
CONSTANT_DIR = {"forgewm": "forgewm", "mg2": "mg2", "hyworld": "hyworld_aligned"}
TRANSLATIONS = {"forward", "back", "left", "right"}


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def latest_vbench(model_dir: Path) -> dict:
    candidates = []
    for name in glob.glob(str(model_dir / "eval" / "*eval_results.json")):
        payload = json.loads(Path(name).read_text(encoding="utf-8"))
        if all(len(payload.get(key, [None, []])[1]) == 462 for key in (
            "imaging_quality", "aesthetic_quality"
        )):
            candidates.append((Path(name).stat().st_mtime, payload))
    if not candidates:
        raise RuntimeError(f"No complete VBench result under {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def collect_shards(directory: Path) -> list[dict[str, str]]:
    result = []
    for path in sorted(directory.glob("rank*.tsv")):
        result.extend(rows(path))
    return result


def kctrl(trajectory: list[dict]) -> float:
    index = {
        (row["scene"], int(row.get("seed", 0)), row["action"]): row
        for row in trajectory if row["action"] in TRANSLATIONS
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
        raise RuntimeError(f"Incomplete KCtrl counterfactual pairs: {len(by_scene)} scenes")
    return float(np.mean([np.mean(value) for value in by_scene.values()]))


def choose_complete(primary: Path, fallback: Path, expected: int) -> list[dict[str, str]]:
    if primary.is_file():
        candidate = rows(primary)
        if len(candidate) == expected:
            return candidate
    candidate = rows(fallback)
    if len(candidate) != expected:
        raise RuntimeError(f"Incomplete metric rows: {fallback}={len(candidate)}, expected {expected}")
    return candidate


def choose_shards(primary: Path, fallback: Path, expected: int) -> list[dict[str, str]]:
    candidate = collect_shards(primary) if primary.is_dir() else []
    if len(candidate) != expected:
        candidate = collect_shards(fallback)
    if len(candidate) != expected:
        raise RuntimeError(f"Incomplete perceptual metric for {fallback.name}: {len(candidate)}")
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path)
    parser.add_argument("--metric-root", type=Path)
    args = parser.parse_args()
    root = args.main_root
    metric_root = args.metric_root or root / "metrics"
    paired = root / "paired_full1000"
    constant = root / "constant_action"
    visual = args.visual_root or constant

    temporal_path = metric_root / "paired_temporal_action_metrics.tsv"
    if not temporal_path.is_file():
        temporal_path = paired / "supplementary_metrics/paired_temporal_action_metrics.tsv"
    temporal_all = rows(temporal_path)
    if len(temporal_all) != 3000:
        raise RuntimeError(f"Incomplete temporal metrics: {len(temporal_all)}")

    summary = {
        "protocol": {
            "models": list(MODELS),
            "paired_gt_rollouts_per_model": 1000,
            "constant_action_scenes": 77,
            "constant_actions": 6,
            "constant_action_videos_per_model": 462,
            "frames": 77,
            "resolution": "640x352",
            "fps": 12,
            "seed": 0,
        },
        "models": {},
    }
    for model in MODELS:
        constant_model = CONSTANT_DIR[model]
        vbench = latest_vbench(visual / constant_model)
        perceptual = choose_shards(
            metric_root / "perceptual" / model,
            root / "perceptual" / model,
            1000,
        )
        depth = choose_complete(
            metric_root / "depth" / model / "depth.tsv",
            paired / "custom_metrics" / model / "per_video_metrics.tsv",
            1000,
        )
        temporal = [row for row in temporal_all if row["model"] == model]
        if len(temporal) != 1000:
            raise RuntimeError(f"Incomplete temporal metrics for {model}: {len(temporal)}")
        trajectory_payload = json.loads(
            (constant / "action_trajectory" / constant_model / "action_consistency.json")
            .read_text(encoding="utf-8")
        )
        trajectory = trajectory_payload["per_video"]
        idm = json.loads(
            (constant / "action_idm" / constant_model / "summary.json")
            .read_text(encoding="utf-8")
        )
        if len(trajectory) != 462 or idm["successful_videos"] != 462:
            raise RuntimeError(f"Incomplete control metrics for {model}")

        imaging = float(vbench["imaging_quality"][0])
        if imaging > 1:
            imaging /= 100.0
        summary["models"][model] = {
            "label": LABELS[model],
            "imaging_quality": imaging,
            "lpips_gt": float(np.mean([float(row["lpips_gt"]) for row in perceptual])),
            "aesthetic_quality": float(vbench["aesthetic_quality"][0]),
            "flow_profile_cosine": float(np.mean([
                float(row["flow_profile_cosine"]) for row in temporal
            ])),
            "depth_gt": float(np.mean([float(row["depth_gt"]) for row in depth])),
            "action_sign_accuracy": float(np.nanmean([
                float(row["action_signature_sign_accuracy"]) for row in temporal
            ])),
            "kctrl": kctrl(trajectory),
            "mouse_accuracy": float(idm["mouse_accuracy"]),
            "counts": {
                "vbench": 462, "perceptual": len(perceptual), "depth": len(depth),
                "temporal": len(temporal), "trajectory": len(trajectory),
                "idm": idm["successful_videos"],
            },
        }

    metric_root.mkdir(parents=True, exist_ok=True)
    (metric_root / "main_table_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Paper main-table evaluation",
        "",
        "| Model | Imaging Quality ↑ | LPIPS ↓ | Aesthetic Quality ↑ | Flow Prof. ↑ | Depth ↑ | Sign Acc. ↑ | KCtrl ↑ | Mouse Acc. ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("mg2", "hyworld", "forgewm"):
        value = summary["models"][model]
        lines.append(
            f"| {value['label']} | {value['imaging_quality']:.4f} | {value['lpips_gt']:.4f} | "
            f"{value['aesthetic_quality']:.4f} | {value['flow_profile_cosine']:.4f} | "
            f"{value['depth_gt']:.4f} | {value['action_sign_accuracy']:.4f} | "
            f"{value['kctrl']:.4f} | {value['mouse_accuracy']:.4f} |"
        )
    text = "\n".join(lines) + "\n"
    (metric_root / "main_table_summary.md").write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
