#!/usr/bin/env python3
"""Paired temporal, action, and projective-flow metrics for generated videos.

All ranked metrics compare a generated clip with its paired GT clip. The
implementation intentionally avoids output-dependent thresholds and does not
reward a model merely for producing little motion.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

from path_utils import resolve_path


ROOT = Path(os.environ.get("FORGEWM_ROOT", Path(__file__).resolve().parents[2])).resolve()
PIPELINE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODELS = ("forgewm", "mg2", "hyworld")
ACTION_LABELS = ("forward", "back", "left", "right")
FIELDS = [
    "model",
    "test_id",
    "clip_idx",
    "action_label",
    "generated_video",
    "gt_video",
    "flow_vector_relative_epe",
    "flow_vector_cosine",
    "flow_profile_relative_error",
    "flow_profile_cosine",
    "temporal_gradient_relative_error",
    "action_signature_sign_accuracy",
    "action_signature_relative_error",
    "action_signature_cosine",
    "projective_flow_relative_error",
    "projective_flow_cosine",
    "divergence_relative_error",
    "curl_relative_error",
    "sampled_frames",
    "flow_transitions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=PIPELINE_DIR / "data/reference/gt_1000/manifest.tsv",
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=PIPELINE_DIR / "data/main_table/paired_full1000",
    )
    parser.add_argument(
        "--output-tsv",
        type=Path,
        default=(
            PIPELINE_DIR
            / "data/main_table/metrics/"
            / "paired_temporal_action_metrics.tsv"
        ),
    )
    parser.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--frame-limit", type=int, default=77)
    parser.add_argument("--frame-stride", type=int, default=4)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--height", type=int, default=88)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--models", nargs="+", default=list(DEFAULT_MODELS),
        help="Model directory names under --eval-root.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def model_video_index(eval_root: Path, model: str) -> dict[int, Path]:
    manifest = eval_root / model / "manifest.tsv"
    rows = read_tsv(manifest)
    return {
        int(row["test_id"]): resolve_path(
            row["generated_video_path"], root=ROOT, base=manifest.parent)
        for row in rows
    }


def load_gray_frames(
    path: str, frame_limit: int, frame_stride: int, width: int, height: int
) -> np.ndarray:
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    frames = []
    index = 0
    while index < frame_limit:
        ok, frame = capture.read()
        if not ok:
            break
        if index % frame_stride == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(gray)
        index += 1
    capture.release()
    expected = (frame_limit - 1) // frame_stride + 1
    if len(frames) != expected:
        raise ValueError(f"Expected {expected} sampled frames in {path}, got {len(frames)}")
    return np.stack(frames)


def dense_flows(frames: np.ndarray) -> np.ndarray:
    flows = []
    for first, second in zip(frames[:-1], frames[1:]):
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            pyr_scale=0.5,
            levels=4,
            winsize=21,
            iterations=4,
            poly_n=7,
            poly_sigma=1.5,
            flags=0,
        )
        flows.append(flow.astype(np.float32))
    return np.stack(flows)


def central_roi(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[1:3]
    return array[
        :,
        int(0.08 * height):max(int(0.82 * height), 1),
        int(0.08 * width):max(int(0.92 * width), 1),
        ...,
    ]


def cosine(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64, copy=False).reshape(-1)
    b = second.astype(np.float64, copy=False).reshape(-1)
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def symmetric_relative_l1(first: np.ndarray, second: np.ndarray) -> float:
    a = first.astype(np.float64, copy=False)
    b = second.astype(np.float64, copy=False)
    numerator = 2.0 * float(np.mean(np.abs(a - b)))
    denominator = float(np.mean(np.abs(a)) + np.mean(np.abs(b)))
    return numerator / max(denominator, 1e-12)


def flow_profile(flow: np.ndarray) -> np.ndarray:
    return np.linalg.norm(flow, axis=-1).mean(axis=(1, 2))


def flow_jacobian(flow: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u = flow[..., 0]
    v = flow[..., 1]
    du_dy, du_dx = np.gradient(u, axis=(1, 2))
    dv_dy, dv_dx = np.gradient(v, axis=(1, 2))
    jacobian = np.stack([du_dx, du_dy, dv_dx, dv_dy], axis=-1)
    divergence = du_dx + dv_dy
    curl = dv_dx - du_dy
    return jacobian, divergence, curl


def action_signature(flow: np.ndarray, action: str) -> np.ndarray | None:
    if action not in ACTION_LABELS:
        return None
    height, width = flow.shape[1:3]
    ys, xs = np.mgrid[:height, :width]
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    radius_x = (xs - center_x) / max(center_x, 1.0)
    radius_y = (ys - center_y) / max(center_y, 1.0)
    radius_norm = np.sqrt(radius_x ** 2 + radius_y ** 2)
    valid = radius_norm > 0.15
    unit_x = radius_x / np.maximum(radius_norm, 1e-6)
    unit_y = radius_y / np.maximum(radius_norm, 1e-6)
    if action in ("left", "right"):
        values = flow[..., 0]
    else:
        values = flow[..., 0] * unit_x + flow[..., 1] * unit_y
    values = values[:, valid]
    return np.median(values, axis=1)


def temporal_gradient(frames: np.ndarray) -> np.ndarray:
    normalized = frames.astype(np.float32) / 255.0
    return np.abs(np.diff(normalized, axis=0))


def paired_metrics(
    generated_frames: np.ndarray,
    gt_frames: np.ndarray,
    action: str,
) -> dict[str, float]:
    generated_flow = central_roi(dense_flows(generated_frames))
    gt_flow = central_roi(dense_flows(gt_frames))

    flow_difference = np.linalg.norm(generated_flow - gt_flow, axis=-1)
    gt_magnitude = np.linalg.norm(gt_flow, axis=-1)
    flow_relative_epe = float(flow_difference.mean()) / max(float(gt_magnitude.mean()), 1e-12)

    generated_profile = flow_profile(generated_flow)
    gt_profile = flow_profile(gt_flow)
    generated_temporal = central_roi(temporal_gradient(generated_frames)[..., None])[..., 0]
    gt_temporal = central_roi(temporal_gradient(gt_frames)[..., None])[..., 0]

    generated_jacobian, generated_divergence, generated_curl = flow_jacobian(generated_flow)
    gt_jacobian, gt_divergence, gt_curl = flow_jacobian(gt_flow)

    generated_signature = action_signature(generated_flow, action)
    gt_signature = action_signature(gt_flow, action)
    if generated_signature is None or gt_signature is None:
        signature_sign = math.nan
        signature_error = math.nan
        signature_cosine = math.nan
    else:
        gt_net = float(gt_signature.sum())
        generated_net = float(generated_signature.sum())
        signature_sign = float(
            np.sign(generated_net) == np.sign(gt_net) and abs(gt_net) > 1e-8
        )
        signature_error = symmetric_relative_l1(generated_signature, gt_signature)
        signature_cosine = cosine(generated_signature, gt_signature)

    return {
        "flow_vector_relative_epe": flow_relative_epe,
        "flow_vector_cosine": cosine(generated_flow, gt_flow),
        "flow_profile_relative_error": symmetric_relative_l1(
            generated_profile, gt_profile
        ),
        "flow_profile_cosine": cosine(generated_profile, gt_profile),
        "temporal_gradient_relative_error": symmetric_relative_l1(
            generated_temporal, gt_temporal
        ),
        "action_signature_sign_accuracy": signature_sign,
        "action_signature_relative_error": signature_error,
        "action_signature_cosine": signature_cosine,
        "projective_flow_relative_error": symmetric_relative_l1(
            generated_jacobian, gt_jacobian
        ),
        "projective_flow_cosine": cosine(generated_jacobian, gt_jacobian),
        "divergence_relative_error": symmetric_relative_l1(
            generated_divergence, gt_divergence
        ),
        "curl_relative_error": symmetric_relative_l1(generated_curl, gt_curl),
    }


def evaluate_task(task: dict[str, object]) -> list[dict[str, object]]:
    cv2.setNumThreads(1)
    gt_frames = load_gray_frames(
        str(task["gt_video"]),
        int(task["frame_limit"]),
        int(task["frame_stride"]),
        int(task["width"]),
        int(task["height"]),
    )
    rows = []
    for model, generated_video in task["generated_videos"].items():
        generated_frames = load_gray_frames(
            str(generated_video),
            int(task["frame_limit"]),
            int(task["frame_stride"]),
            int(task["width"]),
            int(task["height"]),
        )
        metrics = paired_metrics(generated_frames, gt_frames, str(task["action_label"]))
        row = {
            "model": model,
            "test_id": int(task["test_id"]),
            "clip_idx": int(task["clip_idx"]),
            "action_label": task["action_label"],
            "generated_video": str(generated_video),
            "gt_video": str(task["gt_video"]),
            **metrics,
            "sampled_frames": len(gt_frames),
            "flow_transitions": len(gt_frames) - 1,
        }
        rows.append(row)
    return rows


def format_row(row: dict[str, object]) -> dict[str, object]:
    output = {}
    for field in FIELDS:
        value = row.get(field, "")
        if isinstance(value, float):
            output[field] = "nan" if math.isnan(value) else f"{value:.8f}"
        else:
            output[field] = value
    return output


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (int(item["test_id"]), str(item["model"]))):
            writer.writerow(format_row(row))
    os.replace(temporary, path)


def write_metadata(args: argparse.Namespace, sample_count: int) -> None:
    metadata = {
        "dataset_manifest": str(resolve_path(args.dataset_manifest, root=ROOT)),
        "eval_root": str(resolve_path(args.eval_root, root=ROOT)),
        "output_tsv": str(args.output_tsv.resolve()),
        "sample_count": sample_count,
        "model_count": len(args.models),
        "frame_limit": args.frame_limit,
        "frame_stride": args.frame_stride,
        "sampled_frames": (args.frame_limit - 1) // args.frame_stride + 1,
        "flow_resolution": [args.width, args.height],
        "flow_estimator": "OpenCV Farneback",
        "conditioning": "paired first frame and native action scale",
        "selection_policy": (
            "Complete metric suite is retained; any ForgeWM-only showcase is post-hoc."
        ),
        "metric_directions": {
            "flow_vector_relative_epe": "lower",
            "flow_vector_cosine": "higher",
            "flow_profile_relative_error": "lower",
            "flow_profile_cosine": "higher",
            "temporal_gradient_relative_error": "lower",
            "action_signature_sign_accuracy": "higher",
            "action_signature_relative_error": "lower",
            "action_signature_cosine": "higher",
            "projective_flow_relative_error": "lower",
            "projective_flow_cosine": "higher",
            "divergence_relative_error": "lower",
            "curl_relative_error": "lower",
        },
    }
    path = args.output_tsv.with_suffix(".json")
    path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_manifest = resolve_path(args.dataset_manifest, root=ROOT)
    dataset = read_tsv(dataset_manifest)
    if args.max_samples > 0:
        dataset = dataset[:args.max_samples]
    video_indices = {
        model: model_video_index(resolve_path(args.eval_root, root=ROOT), model)
        for model in args.models
    }

    existing_rows = []
    if args.output_tsv.is_file() and not args.overwrite:
        existing_rows = read_tsv(args.output_tsv)
    completed = {
        (int(row["test_id"]), row["model"])
        for row in existing_rows
        if row.get("projective_flow_cosine", "") != ""
    }
    output_rows: list[dict[str, object]] = list(existing_rows)

    tasks = []
    for row in dataset:
        test_id = int(row["test_id"])
        pending_models = [
            model for model in args.models if (test_id, model) not in completed
        ]
        if not pending_models:
            continue
        generated = {model: video_indices[model][test_id] for model in pending_models}
        gt_video = resolve_path(
            row["video_path"], root=ROOT, base=dataset_manifest.parent)
        paths = [gt_video, *generated.values()]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing videos for test_id={test_id}: {missing}")
        tasks.append({
            "test_id": test_id,
            "clip_idx": int(row["clip_idx"]),
            "action_label": row["action_label"],
            "gt_video": str(gt_video),
            "generated_videos": {model: str(path) for model, path in generated.items()},
            "frame_limit": args.frame_limit,
            "frame_stride": args.frame_stride,
            "width": args.width,
            "height": args.height,
        })

    print(
        f"samples={len(dataset)} complete_rows={len(completed)} "
        f"pending_samples={len(tasks)} workers={args.workers}",
        flush=True,
    )
    if not tasks:
        write_metadata(args, len(dataset))
        return

    finished = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(evaluate_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                new_rows = future.result()
            except Exception as exc:
                raise RuntimeError(f"Failed test_id={task['test_id']}: {exc}") from exc
            output_rows.extend(new_rows)
            finished += 1
            if finished % 10 == 0 or finished == len(tasks):
                write_rows(args.output_tsv, output_rows)
                print(
                    f"finished_samples={finished}/{len(tasks)} "
                    f"rows={len(output_rows)}",
                    flush=True,
                )
    write_metadata(args, len(dataset))


if __name__ == "__main__":
    main()
