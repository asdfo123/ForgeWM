#!/usr/bin/env python3
"""Evaluate GameWorld temporal consistency and paired, calibrated LPIPS."""

import argparse
import csv
import json
import os
from pathlib import Path

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
import clip

from path_utils import resolve_path


ROOT = Path(os.environ.get("FORGEWM_ROOT", Path(__file__).resolve().parents[2])).resolve()
FIELDS = ["test_id", "clip_idx", "generated_video", "gt_video",
          "temporal_consistency", "lpips_gt"]


def read_video(path, count=77):
    capture = cv2.VideoCapture(str(path))
    frames = []
    while len(frames) < count:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if len(frames) != count:
        raise RuntimeError(f"Expected {count} frames, got {len(frames)}: {path}")
    return np.stack(frames)


def clip_input(frames, device):
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).to(
        device=device, dtype=torch.float32) / 255.0
    height, width = tensor.shape[-2:]
    scale = 224.0 / min(height, width)
    resized = F.interpolate(
        tensor, size=(round(height * scale), round(width * scale)),
        mode="bicubic", align_corners=False, antialias=True)
    top = (resized.shape[-2] - 224) // 2
    left = (resized.shape[-1] - 224) // 2
    resized = resized[:, :, top:top + 224, left:left + 224]
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073],
                        device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711],
                       device=device).view(1, 3, 1, 1)
    return (resized - mean) / std


@torch.inference_mode()
def temporal_score(model, frames, device, batch_size=32):
    features = []
    for start in range(0, len(frames), batch_size):
        inputs = clip_input(frames[start:start + batch_size], device)
        features.append(F.normalize(model.encode_image(inputs), dim=-1).float())
    features = torch.cat(features)
    adjacent = (features[:-1] * features[1:]).sum(dim=-1).clamp_min(0)
    first = (features[:1] * features[1:]).sum(dim=-1).clamp_min(0)
    return float(((adjacent + first) * 0.5).mean().item())


@torch.inference_mode()
def paired_lpips(model, generated, gt, device):
    indices = np.linspace(1, 76, 16).round().astype(int)
    values = []
    for start in range(0, len(indices), 4):
        chosen = indices[start:start + 4]
        pred = torch.from_numpy(generated[chosen]).permute(0, 3, 1, 2)
        target = torch.from_numpy(gt[chosen]).permute(0, 3, 1, 2)
        pred = pred.to(device=device, dtype=torch.float32) / 255.0
        target = target.to(device=device, dtype=torch.float32) / 255.0
        values.append(model(pred, target, normalize=True).flatten().float().cpu())
    return float(torch.cat(values).mean().item())


def write_rows(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--model_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--clip_checkpoint", required=True)
    parser.add_argument("--max_samples", type=int)
    args = parser.parse_args()

    dataset_manifest = resolve_path(args.dataset_manifest, root=ROOT)
    model_manifest = resolve_path(args.model_manifest, root=ROOT)
    with dataset_manifest.open(encoding="utf-8") as handle:
        dataset = list(csv.DictReader(handle, delimiter="\t"))
    with model_manifest.open(encoding="utf-8") as handle:
        generated_rows = list(csv.DictReader(handle, delimiter="\t"))
    generated = {int(row["test_id"]): resolve_path(
                     row["generated_video_path"], root=ROOT, base=model_manifest.parent)
                 for row in generated_rows}
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    dataset = [row for index, row in enumerate(dataset)
               if index % args.world_size == args.rank]

    out_dir = resolve_path(args.out_dir, root=ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"rank{args.rank}.tsv"
    rows = []
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = {int(row["test_id"]) for row in rows}

    device = torch.device(args.device)
    checkpoint_path = resolve_path(args.clip_checkpoint, root=ROOT)
    clip_source = str(checkpoint_path) if checkpoint_path.is_file() else args.clip_checkpoint
    clip_model, _ = clip.load(clip_source, device=device)
    clip_model.eval()
    lpips_model = lpips.LPIPS(net="alex", pnet_rand=False).to(device).eval()
    for position, row in enumerate(dataset, 1):
        test_id = int(row["test_id"])
        if test_id in completed:
            continue
        generated_path = generated[test_id]
        gt_path = resolve_path(row["video_path"], root=ROOT, base=dataset_manifest.parent)
        pred_frames = read_video(generated_path)
        gt_frames = read_video(gt_path)
        result = {
            "test_id": test_id,
            "clip_idx": int(row["clip_idx"]),
            "generated_video": str(generated_path),
            "gt_video": str(gt_path),
            "temporal_consistency": f"{temporal_score(clip_model, pred_frames, device):.8f}",
            "lpips_gt": f"{paired_lpips(lpips_model, pred_frames, gt_frames, device):.8f}",
        }
        rows.append(result)
        write_rows(output, rows)
        print(f"rank={args.rank} sample={position}/{len(dataset)} test_id={test_id} "
              f"temporal={result['temporal_consistency']} lpips={result['lpips_gt']}",
              flush=True)

    summary = {
        "rank": args.rank,
        "samples": len(rows),
        "temporal_consistency": float(np.mean([float(r["temporal_consistency"]) for r in rows])),
        "lpips_gt": float(np.mean([float(r["lpips_gt"]) for r in rows])),
    }
    (out_dir / f"rank{args.rank}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
