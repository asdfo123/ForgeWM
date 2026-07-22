#!/usr/bin/env python3
"""Compute only the paper's paired DA3 depth metric on Full-1000."""

import argparse
import csv
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(os.environ.get("FORGEWM_ROOT", Path(__file__).resolve().parents[2])).resolve()
sys.path.insert(0, str(Path(__file__).resolve().parent))
from custom_metrics_eval import DepthAnything
from path_utils import resolve_path

FIELDS = ("test_id", "clip_idx", "generated_video", "gt_video", "depth_gt")
INDICES = np.linspace(1, 76, 16).round().astype(int)
DEPTH_INDICES = set(
    INDICES[np.linspace(0, len(INDICES) - 1, 8).round().astype(int)].tolist()
)


def read_selected(path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    frames = []
    index = 0
    while index <= 76:
        ok, frame = capture.read()
        if not ok:
            break
        if index in DEPTH_INDICES:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        index += 1
    capture.release()
    if len(frames) != 8:
        raise RuntimeError(f"Expected 8 sampled frames from {path}, got {len(frames)}")
    return frames


def write_rows(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--model-manifest", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--da3-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    args = parser.parse_args()
    dataset_manifest = resolve_path(args.dataset_manifest, root=ROOT)
    model_manifest = resolve_path(args.model_manifest, root=ROOT)
    with dataset_manifest.open(encoding="utf-8") as handle:
        dataset = list(csv.DictReader(handle, delimiter="\t"))
    dataset = [
        row for index, row in enumerate(dataset)
        if index % args.world_size == args.rank
    ]
    with model_manifest.open(encoding="utf-8") as handle:
        generated = {
            int(row["test_id"]): resolve_path(
                row["generated_video_path"], root=ROOT, base=model_manifest.parent)
            for row in csv.DictReader(handle, delimiter="\t")
        }
    existing = []
    if args.output.is_file():
        with args.output.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames == list(FIELDS):
                existing = list(reader)
    completed = {int(row["test_id"]) for row in existing}
    rows = list(existing)
    device = torch.device(args.device)
    model = DepthAnything(args.da3_model, device)
    for position, row in enumerate(dataset, 1):
        test_id = int(row["test_id"])
        if test_id in completed:
            continue
        generated_frames = read_selected(generated[test_id])
        gt_path = resolve_path(row["video_path"], root=ROOT, base=dataset_manifest.parent)
        gt_frames = read_selected(gt_path)
        with torch.no_grad():
            generated_depth = model(generated_frames, device)
            gt_depth = model(gt_frames, device)
        score = float(F.cosine_similarity(
            generated_depth.flatten(1), gt_depth.flatten(1), dim=1).mean().item())
        rows.append({
            "test_id": test_id,
            "clip_idx": int(row["clip_idx"]),
            "generated_video": str(generated[test_id]),
            "gt_video": str(gt_path),
            "depth_gt": f"{score:.8f}",
        })
        write_rows(args.output, rows)
        print(
            f"sample={position}/{len(dataset)} test_id={test_id} depth={score:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
