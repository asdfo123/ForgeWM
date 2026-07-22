#!/usr/bin/env python3
"""Corrected custom metrics against the paired GT trajectory for each sample."""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity

ROOT = Path(os.environ.get("FORGEWM_ROOT", Path(__file__).resolve().parents[2])).resolve()
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from custom_metrics_eval import DepthAnything, VJEPA2, load_frames
from path_utils import resolve_path


FIELDS = [
    "test_id", "clip_idx", "generated_video", "gt_video", "psnr_gt",
    "ssim_gt", "dino_gt", "vjepa_gt", "depth_gt", "flow_gen", "flow_gt",
    "flow_abs_error", "flow_relative_error", "dynamic_degree_gen",
    "dynamic_degree_gt", "dynamic_degree_abs_error", "first_frame_psnr",
]


def sampled_indices(frame_count, count=16):
    # Frame zero is conditioning; score predicted frames only.
    return np.linspace(1, frame_count - 1, min(count, frame_count - 1)).round().astype(int)


def paired_pixel_metrics(generated, gt, indices):
    psnr_values = []
    ssim_values = []
    for index in indices:
        pred = generated[index]
        target = gt[index]
        mse = float(np.mean((pred.astype(np.float32) - target.astype(np.float32)) ** 2))
        psnr_values.append(100.0 if mse == 0 else 10.0 * np.log10(255.0 ** 2 / mse))
        ssim_values.append(structural_similarity(
            pred, target, channel_axis=2, data_range=255))
    return float(np.mean(psnr_values)), float(np.mean(ssim_values))


def first_frame_psnr(generated, gt):
    mse = float(np.mean(
        (generated[0].astype(np.float32) - gt[0].astype(np.float32)) ** 2))
    return 100.0 if mse == 0 else float(10.0 * np.log10(255.0 ** 2 / mse))


def flow_statistics(frames, threshold=0.5):
    gray = [
        cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (320, 176),
                   interpolation=cv2.INTER_AREA)
        for frame in frames
    ]
    means = []
    for first, second in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            first, second, None, pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        means.append(float(np.linalg.norm(flow, axis=2).mean()))
    values = np.asarray(means, dtype=np.float32)
    return float(values.mean()), float((values > threshold).mean())


class DINOFrameEncoder:
    def __init__(self, repo, checkpoint, device):
        self.device = device
        self.model = torch.hub.load(
            str(repo), "dino_vitb16", source="local", pretrained=True,
            path=str(checkpoint)).to(device).eval()
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(
            [0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def paired_similarity(self, generated, gt, indices):
        frames = [generated[index] for index in indices]
        frames.extend(gt[index] for index in indices)
        array = np.stack([
            cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
            for frame in frames
        ])
        tensor = torch.from_numpy(array).permute(0, 3, 1, 2)
        tensor = tensor.to(self.device, dtype=torch.float32) / 255.0
        features = F.normalize(self.model((tensor - self.mean) / self.std), dim=-1)
        count = len(indices)
        return float((features[:count] * features[count:]).sum(dim=1).mean().item())


def paired_depth_similarity(depth_model, generated, gt, indices, device):
    # Eight paired frames are enough to cover the trajectory while keeping DA3
    # runtime practical for 3,000 videos.
    depth_indices = indices[np.linspace(0, len(indices) - 1, min(8, len(indices))).round().astype(int)]
    generated_frames = [generated[index] for index in depth_indices]
    gt_frames = [gt[index] for index in depth_indices]
    with torch.no_grad():
        # DA3 performs joint multi-view inference, so generated and GT frames
        # must be processed separately to avoid cross-set information leakage.
        generated_depths = depth_model(generated_frames, device)
        gt_depths = depth_model(gt_frames, device)
    first = generated_depths.flatten(1)
    second = gt_depths.flatten(1)
    return float(F.cosine_similarity(first, second, dim=1).mean().item())


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
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--da3-model", default="depth-anything/DA3-GIANT-1.1")
    parser.add_argument("--vjepa-model", default="facebook/vjepa2-vitl-fpc64-256")
    parser.add_argument("--dino-repo", required=True)
    parser.add_argument("--dino-checkpoint", required=True)
    args = parser.parse_args()

    root = ROOT
    dataset_manifest = resolve_path(args.dataset_manifest, root=root)
    model_manifest = resolve_path(args.model_manifest, root=root)
    with dataset_manifest.open(encoding="utf-8") as handle:
        dataset = list(csv.DictReader(handle, delimiter="\t"))
    with model_manifest.open(encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle, delimiter="\t"))
    generated_by_test = {
        int(row["test_id"]): resolve_path(
            row["generated_video_path"], root=root, base=model_manifest.parent)
        for row in model_rows
    }
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]

    out_dir = resolve_path(args.out_dir, root=root)
    feature_dir = out_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "per_video_metrics.tsv"
    existing_rows = []
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as handle:
            existing_rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = {
        int(row["test_id"]) for row in existing_rows
        if (feature_dir / f"test_{int(row['test_id']):04d}.npz").is_file()
    }

    device = torch.device(args.device)
    vjepa = VJEPA2(args.vjepa_model, device)
    depth = DepthAnything(args.da3_model, device)
    dino = DINOFrameEncoder(
        resolve_path(args.dino_repo, root=root),
        resolve_path(args.dino_checkpoint, root=root),
        device)

    rows = list(existing_rows)
    for position, row in enumerate(dataset):
        test_id = int(row["test_id"])
        if test_id in completed:
            continue
        generated_path = generated_by_test.get(test_id)
        if generated_path is None:
            raise RuntimeError(f"No generated video for test_id={test_id}")
        gt_path = resolve_path(
            row["video_path"], root=root, base=dataset_manifest.parent)
        generated = load_frames(generated_path, max_frames=77)
        gt = load_frames(gt_path, max_frames=77)
        if len(generated) != 77 or len(gt) != 77:
            raise RuntimeError(
                f"Frame mismatch test_id={test_id}: generated={len(generated)} gt={len(gt)}")
        indices = sampled_indices(77, 16)
        psnr, ssim = paired_pixel_metrics(generated, gt, indices)
        dino_score = dino.paired_similarity(generated, gt, indices)
        generated_feature = vjepa(generated[1:])
        gt_feature = vjepa(gt[1:])
        jepa_score = float(np.dot(generated_feature, gt_feature) / (
            np.linalg.norm(generated_feature) * np.linalg.norm(gt_feature) + 1e-8))
        depth_score = paired_depth_similarity(depth, generated, gt, indices, device)
        flow_gen, dynamic_gen = flow_statistics(generated)
        flow_gt, dynamic_gt = flow_statistics(gt)
        flow_error = abs(flow_gen - flow_gt)
        result = {
            "test_id": test_id,
            "clip_idx": int(row["clip_idx"]),
            "generated_video": str(generated_path),
            "gt_video": str(gt_path),
            "psnr_gt": f"{psnr:.8f}",
            "ssim_gt": f"{ssim:.8f}",
            "dino_gt": f"{dino_score:.8f}",
            "vjepa_gt": f"{jepa_score:.8f}",
            "depth_gt": f"{depth_score:.8f}",
            "flow_gen": f"{flow_gen:.8f}",
            "flow_gt": f"{flow_gt:.8f}",
            "flow_abs_error": f"{flow_error:.8f}",
            "flow_relative_error": f"{flow_error / (flow_gt + 1e-8):.8f}",
            "dynamic_degree_gen": f"{dynamic_gen:.8f}",
            "dynamic_degree_gt": f"{dynamic_gt:.8f}",
            "dynamic_degree_abs_error": f"{abs(dynamic_gen - dynamic_gt):.8f}",
            "first_frame_psnr": f"{first_frame_psnr(generated, gt):.8f}",
        }
        rows.append(result)
        np.savez_compressed(
            feature_dir / f"test_{test_id:04d}.npz",
            test_id=np.asarray(test_id),
            generated_vjepa=generated_feature,
            gt_vjepa=gt_feature,
        )
        write_rows(metrics_path, rows)
        print(
            f"sample={position + 1}/{len(dataset)} test_id={test_id} "
            f"psnr={psnr:.3f} ssim={ssim:.4f} jepa={jepa_score:.4f}",
            flush=True)

    config = {
        "dataset_manifest": args.dataset_manifest,
        "model_manifest": args.model_manifest,
        "samples": len(dataset),
        "evaluated_frames": 77,
        "conditioning_frame_excluded_from_ranked_similarity_metrics": True,
        "paired_frame_samples": 16,
        "paired_depth_samples": 8,
        "depth_inference": "generated and GT processed separately",
        "flow_resolution": "320x176",
        "flow_threshold": 0.5,
        "lpips_excluded": (
            "Official ImageNet LPIPS backbone is unavailable locally; the old random-backbone "
            "number was invalid and is intentionally not reported."),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
