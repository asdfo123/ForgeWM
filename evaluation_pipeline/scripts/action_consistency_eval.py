#!/usr/bin/env python3
"""Evaluate camera-action consistency from poses estimated by Depth Anything 3.

The metrics intentionally avoid a test-time motion threshold:
  - direction_accuracy: fraction of relative poses with the expected sign.
  - dominant_axis_accuracy: expected sign is correct and the commanded axis
    dominates the other translation/rotation axes.
  - net_direction_correct: cumulative signed motion has the expected sign.
  - direction_cosine: expected-axis projection divided by total motion norm.

DA3 translation scale is monocular and arbitrary, so translation magnitudes are
reported for diagnostics only and are not compared directly across models.
Dense image flow is also reported for turns as an independent direction check.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation


ACTIONS = ("forward", "back", "left", "right", "turn_left", "turn_right")
ACTION_RE = re.compile(
    r"(?P<scene>scene\d+_[A-Za-z0-9_]+?)_"
    r"(?P<action>turn_left|turn_right|forward|back|left|right)_seed(?P<seed>\d+)"
    r"(?:-\d+)?(?:\.mp4)?$"
)
ACTION_SPEC = {
    "forward": ("translation", 2, 1.0),
    "back": ("translation", 2, -1.0),
    "left": ("translation", 0, -1.0),
    "right": ("translation", 0, 1.0),
    "turn_left": ("rotation", 1, -1.0),
    "turn_right": ("rotation", 1, 1.0),
}


def parse_identity(path: str) -> tuple[str, str, int]:
    match = ACTION_RE.search(os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot parse scene/action/seed from {path}")
    return match.group("scene"), match.group("action"), int(match.group("seed"))


def load_sampled_frames(path: str, stride: int = 4) -> list[Image.Image]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if index % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        index += 1
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"Too few sampled frames in {path}: {len(frames)}")
    return frames


def to_homogeneous(extrinsics: np.ndarray) -> np.ndarray:
    extrinsics = np.asarray(extrinsics, dtype=np.float64)
    if extrinsics.ndim == 4 and extrinsics.shape[0] == 1:
        extrinsics = extrinsics[0]
    if extrinsics.shape[-2:] == (3, 4):
        bottom = np.zeros((extrinsics.shape[0], 1, 4), dtype=extrinsics.dtype)
        bottom[:, 0, 3] = 1.0
        extrinsics = np.concatenate([extrinsics, bottom], axis=1)
    if extrinsics.ndim != 3 or extrinsics.shape[-2:] != (4, 4):
        raise ValueError(f"Unexpected extrinsics shape: {extrinsics.shape}")
    return extrinsics


def relative_c2w_from_w2c(extrinsics: np.ndarray) -> np.ndarray:
    w2c = to_homogeneous(extrinsics)
    c2w = np.linalg.inv(w2c)
    return np.linalg.inv(c2w[:-1]) @ c2w[1:]


def median_dense_flow(frames: list[Image.Image]) -> np.ndarray:
    """Return robust (x, y) image motion for each sampled-frame transition."""
    gray = [cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2GRAY) for frame in frames]
    vectors = []
    for previous, current in zip(gray[:-1], gray[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            previous, current, None, 0.5, 5, 21, 3, 7, 1.5, 0
        )
        height, width = previous.shape
        # Exclude borders and the lower HUD-heavy part of Minecraft frames.
        roi = flow[
            int(0.10 * height):int(0.82 * height),
            int(0.10 * width):int(0.90 * width),
        ]
        vectors.append(np.median(roi.reshape(-1, 2), axis=0))
    return np.asarray(vectors, dtype=np.float64)


def score_relative_poses(relative_c2w: np.ndarray, action: str) -> dict[str, float]:
    family, axis, sign = ACTION_SPEC[action]
    translation = relative_c2w[:, :3, 3]
    euler = Rotation.from_matrix(relative_c2w[:, :3, :3]).as_euler("xyz", degrees=False)
    values = translation if family == "translation" else euler
    signed = sign * values[:, axis]
    norms = np.linalg.norm(values, axis=1)
    abs_values = np.abs(values)
    other_axes = [i for i in range(3) if i != axis]
    dominant = abs_values[:, axis] >= np.max(abs_values[:, other_axes], axis=1)
    finite = np.isfinite(signed) & np.isfinite(norms)
    if not finite.any():
        raise ValueError("No finite relative-pose estimates")
    signed = signed[finite]
    norms = norms[finite]
    dominant = dominant[finite]
    cosine = signed / np.maximum(norms, 1e-12)
    correct = signed > 0
    magnitude = np.abs(values[finite, axis])

    if family == "translation":
        # Classify within the four translation controls, independent of scale.
        t = translation[finite]
        class_scores = np.stack([t[:, 2], -t[:, 2], -t[:, 0], t[:, 0]], axis=1)
        expected_class = ("forward", "back", "left", "right").index(action)
    else:
        y = euler[finite, 1]
        class_scores = np.stack([-y, y], axis=1)
        expected_class = ("turn_left", "turn_right").index(action)
    predicted = np.argmax(class_scores, axis=1)

    return {
        "direction_accuracy": float(correct.mean()),
        "dominant_axis_accuracy": float((correct & dominant).mean()),
        "family_action_accuracy": float((predicted == expected_class).mean()),
        "net_direction_correct": float(signed.sum() > 0),
        "direction_cosine": float(cosine.mean()),
        "median_axis_magnitude": float(np.median(magnitude)),
        "cumulative_signed_motion": float(signed.sum()),
        "num_transitions": int(len(signed)),
    }


def score_turn_flow(flow: np.ndarray, action: str) -> dict[str, float]:
    if action not in ("turn_left", "turn_right"):
        raise ValueError(f"Flow turn scoring does not support {action}")
    # Camera left -> world pixels move right; camera right -> pixels move left.
    sign = 1.0 if action == "turn_left" else -1.0
    finite = np.isfinite(flow).all(axis=1)
    values = flow[finite]
    if not len(values):
        raise ValueError("No finite optical-flow estimates")
    signed = sign * values[:, 0]
    norms = np.linalg.norm(values, axis=1)
    correct = signed > 0
    dominant = np.abs(values[:, 0]) >= np.abs(values[:, 1])
    cosine = signed / np.maximum(norms, 1e-12)
    predicted_left = values[:, 0] > 0
    expected_left = action == "turn_left"
    return {
        "direction_accuracy": float(correct.mean()),
        "dominant_axis_accuracy": float((correct & dominant).mean()),
        "family_action_accuracy": float((predicted_left == expected_left).mean()),
        "net_direction_correct": float(signed.sum() > 0),
        "direction_cosine": float(cosine.mean()),
        "median_axis_magnitude": float(np.median(np.abs(values[:, 0]))),
        "cumulative_signed_motion": float(signed.sum()),
        "num_transitions": int(len(signed)),
    }


def bootstrap_scene_ci(rows: list[dict], key: str, seed: int = 0) -> tuple[float, float]:
    by_scene = defaultdict(list)
    for row in rows:
        by_scene[row["scene"]].append(float(row[key]))
    scene_values = np.array([np.mean(v) for v in by_scene.values()], dtype=np.float64)
    if len(scene_values) < 2:
        value = float(scene_values.mean()) if len(scene_values) else float("nan")
        return value, value
    rng = np.random.default_rng(seed)
    sampled = scene_values[rng.integers(0, len(scene_values), size=(20000, len(scene_values)))]
    means = sampled.mean(axis=1)
    return tuple(float(v) for v in np.quantile(means, [0.025, 0.975]))


def opposite_pair_metrics(rows: list[dict]) -> dict[str, float]:
    index = {(r["scene"], r["action"]): r for r in rows}
    pairs = {
        "forward_back": ("forward", "back"),
        "left_right": ("left", "right"),
        "turn_left_right": ("turn_left", "turn_right"),
    }
    out = {}
    for label, (a, b) in pairs.items():
        values = []
        for scene in sorted({r["scene"] for r in rows}):
            if (scene, a) not in index or (scene, b) not in index:
                continue
            values.append(
                float(index[(scene, a)]["net_direction_correct"] > 0.5)
                * float(index[(scene, b)]["net_direction_correct"] > 0.5)
            )
        out[label] = float(np.mean(values)) if values else float("nan")
    valid = [v for v in out.values() if np.isfinite(v)]
    out["opposite_pair_accuracy"] = float(np.mean(valid)) if valid else float("nan")
    return out


def write_outputs(output_dir: Path, model_name: str, rows: list[dict], config: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = (
        "direction_accuracy",
        "dominant_axis_accuracy",
        "family_action_accuracy",
        "net_direction_correct",
        "direction_cosine",
        "median_axis_magnitude",
        "cumulative_signed_motion",
    )
    overall = {key: float(np.mean([r[key] for r in rows])) for key in metrics}
    ci = {key: bootstrap_scene_ci(rows, key) for key in metrics[:5]}
    by_action = {}
    for action in ACTIONS:
        subset = [r for r in rows if r["action"] == action]
        by_action[action] = {
            key: float(np.mean([r[key] for r in subset])) for key in metrics
        }
    pair_metrics = opposite_pair_metrics(rows)
    payload = {
        "model": model_name,
        "config": config,
        "overall": overall,
        "overall_95ci_by_scene": ci,
        "by_action": by_action,
        "opposite_pairs": pair_metrics,
        "per_video": rows,
    }
    (output_dir / "action_consistency.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    diagnostic_fields = [
        "estimator",
        "flow_direction_accuracy",
        "flow_net_direction_correct",
        "flow_cumulative_signed_motion",
    ]
    fields = (
        ["model", "scene", "action", "seed", "video"]
        + list(metrics)
        + ["num_transitions"]
        + diagnostic_fields
    )
    with (output_dir / "per_video_action_consistency.tsv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)

    action_zh = {
        "forward": "前进", "back": "后退", "left": "左移", "right": "右移",
        "turn_left": "左转", "turn_right": "右转",
    }
    md = [
        f"# {model_name} 动作一致性评测",
        "",
        "评估器：六类动作统一使用 Depth Anything 3 相对相机位姿。",
        f"每隔 {config['stride']} 个视频帧采样一次；77 帧视频对应 20 个采样帧。",
        "平移尺度来自单目估计，仅用于模型内部诊断，不跨模型比较绝对幅度。",
        "左右转另用相邻帧稠密光流做独立方向交叉检查，但不混入主指标。",
        "",
        "## 总体结果",
        "",
        "| 指标 | 均值 | 场景 bootstrap 95% CI |",
        "| :--- | ---: | :---: |",
    ]
    labels = {
        "direction_accuracy": "方向准确率",
        "dominant_axis_accuracy": "主轴方向准确率",
        "family_action_accuracy": "动作族分类准确率",
        "net_direction_correct": "净轨迹方向准确率",
        "direction_cosine": "方向余弦",
    }
    for key, label in labels.items():
        low, high = ci[key]
        md.append(f"| {label} | {overall[key]:.4f} | [{low:.4f}, {high:.4f}] |")
    md.extend([
        f"| 相反动作成对准确率 | {pair_metrics['opposite_pair_accuracy']:.4f} | — |",
        "",
        "## 分动作结果",
        "",
        "| 动作 | 方向准确率 | 主轴方向准确率 | 动作族分类准确率 | 净轨迹方向准确率 | 方向余弦 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for action in ACTIONS:
        values = by_action[action]
        md.append(
            f"| {action_zh[action]} | {values['direction_accuracy']:.4f} | "
            f"{values['dominant_axis_accuracy']:.4f} | {values['family_action_accuracy']:.4f} | "
            f"{values['net_direction_correct']:.4f} | {values['direction_cosine']:.4f} |"
        )
    md.extend([
        "",
        "## 相反动作配对",
        "",
        "| 动作对 | 两个方向均正确的场景比例 |",
        "| :--- | ---: |",
        f"| 前进/后退 | {pair_metrics['forward_back']:.4f} |",
        f"| 左移/右移 | {pair_metrics['left_right']:.4f} |",
        f"| 左转/右转 | {pair_metrics['turn_left_right']:.4f} |",
        "",
        "方向准确率只判断符号；主轴方向准确率还要求目标轴是平移或旋转中的主导轴。",
        "动作族分类准确率在四个平移动作或两个旋转动作内部分类，不在平移与旋转之间比较单目尺度。",
        "左右转的光流交叉检查保存在 TSV/JSON 中；相机左转预期场景像素向右，右转反之。",
    ])
    (output_dir / "summary_action_consistency.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--da3-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-videos", type=int, default=0)
    args = parser.parse_args()

    from depth_anything_3.api import DepthAnything3

    device = torch.device(args.device)
    estimator = DepthAnything3.from_pretrained(args.da3_model).to(device).eval()
    videos = sorted(glob.glob(os.path.join(args.videos_dir, "*.mp4")))
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    if not videos:
        raise RuntimeError(f"No MP4 videos found in {args.videos_dir}")

    rows = []
    for index, video in enumerate(videos, start=1):
        scene, action, seed = parse_identity(video)
        if action in ("turn_left", "turn_right"):
            motion_frames = load_sampled_frames(video, 1)
            frames = motion_frames[::args.stride]
        else:
            motion_frames = None
            frames = load_sampled_frames(video, args.stride)
        with torch.no_grad():
            prediction = estimator.inference(frames, export_format="mini_npz")
        relative = relative_c2w_from_w2c(prediction.extrinsics)
        score = score_relative_poses(relative, action)
        if action in ("turn_left", "turn_right"):
            flow_score = score_turn_flow(median_dense_flow(motion_frames), action)
        else:
            flow_score = None
        row = {
            "model": args.model_name,
            "scene": scene,
            "action": action,
            "seed": seed,
            "video": video,
            **score,
            "estimator": "da3_relative_c2w",
            "flow_direction_accuracy": (
                flow_score["direction_accuracy"] if flow_score else None
            ),
            "flow_net_direction_correct": (
                flow_score["net_direction_correct"] if flow_score else None
            ),
            "flow_cumulative_signed_motion": (
                flow_score["cumulative_signed_motion"] if flow_score else None
            ),
        }
        rows.append(row)
        print(
            f"[{index}/{len(videos)}] {os.path.basename(video)} "
            f"dir={score['direction_accuracy']:.3f} "
            f"axis={score['dominant_axis_accuracy']:.3f} "
            f"net={score['net_direction_correct']:.0f}",
            flush=True,
        )

    config = {
        "videos_dir": os.path.abspath(args.videos_dir),
        "da3_model": os.path.abspath(args.da3_model),
        "stride": args.stride,
        "num_videos": len(rows),
        "pose_estimator": "DA3 extrinsics treated as w2c; relative motion computed in c2w",
        "rotation_cross_check": "Farneback median dense flow in central ROI; left turn expects +x image motion",
        "threshold_policy": "No test-time motion threshold",
    }
    write_outputs(Path(args.output_dir), args.model_name, rows, config)


if __name__ == "__main__":
    main()
