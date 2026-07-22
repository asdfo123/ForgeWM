#!/usr/bin/env python3
"""Evaluate generated videos with the GameWorld/VPT inverse dynamics model."""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(os.environ.get("FORGEWM_ROOT", Path(__file__).resolve().parents[2])).resolve()
GAMEWORLD = Path(os.environ.get(
    "GAMEWORLD_ROOT", ROOT / "Matrix-Game/Matrix-Game-1/GameWorldScore"
)).expanduser().resolve()
sys.path.insert(0, str(GAMEWORLD))
sys.path.insert(0, str(GAMEWORLD / "GameWorld/third_party/IDM"))

from GameWorld.third_party.IDM.IDM_bench import IDMAgent, camera_direction


FIELDS = ["test_id", "clip_idx", "generated_video", "keyboard_accuracy",
          "mouse_accuracy", "forward_back_accuracy", "left_right_accuracy"]


def resolve(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def read_frames(path, count=76):
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


def exclusive_labels(first, second):
    first = np.asarray(first).reshape(-1) > 0.5
    second = np.asarray(second).reshape(-1) > 0.5
    labels = np.zeros(len(first), dtype=np.int64)
    labels[first & ~second] = 1
    labels[second & ~first] = 2
    return labels


def flatten_action(actions, key, count=76):
    values = np.asarray(actions[key]).reshape(-1)
    if len(values) < count:
        raise RuntimeError(f"IDM returned {len(values)} values for {key}, need {count}")
    return values[:count]


def alignment_slices(offset, prediction_count, action_count):
    """Align prediction at video frame i with action i + offset."""
    pred_start = max(0, -offset)
    pred_stop = min(prediction_count, action_count - offset)
    if pred_stop <= pred_start:
        raise ValueError(
            f"Offset {offset} leaves no overlap: predictions={prediction_count} "
            f"actions={action_count}")
    return (
        slice(pred_start, pred_stop),
        slice(pred_start + offset, pred_stop + offset),
    )


def write_rows(path, rows):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def load_agent(model_path, weights_path, device):
    with open(model_path, "rb") as handle:
        parameters = pickle.load(handle)
    net_kwargs = parameters["model"]["args"]["net"]["args"]
    head_kwargs = parameters["model"]["args"]["pi_head_opts"]
    head_kwargs["temperature"] = float(head_kwargs["temperature"])
    agent = IDMAgent(net_kwargs, head_kwargs, device=device)
    agent.load_weights(weights_path)
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_manifest", required=True)
    parser.add_argument("--model_manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world_size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_samples", type=int)
    parser.add_argument(
        "--keyboard_offset", type=int, default=1,
        help="Compare keyboard prediction at frame i with action i + offset")
    parser.add_argument(
        "--mouse_offset", type=int, default=1,
        help="Compare mouse prediction at frame i with action i + offset")
    args = parser.parse_args()

    with resolve(args.dataset_manifest).open(encoding="utf-8") as handle:
        dataset = list(csv.DictReader(handle, delimiter="\t"))
    with resolve(args.model_manifest).open(encoding="utf-8") as handle:
        model_rows = list(csv.DictReader(handle, delimiter="\t"))
    videos = {int(row["test_id"]): resolve(row["generated_video_path"])
              for row in model_rows}
    if args.max_samples is not None:
        dataset = dataset[:args.max_samples]
    dataset = [row for index, row in enumerate(dataset)
               if index % args.world_size == args.rank]

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / f"rank{args.rank}.tsv"
    rows = []
    if output.exists():
        with output.open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
    completed = {int(row["test_id"]) for row in rows}

    agent = load_agent(args.model, args.weights, args.device)
    for position, row in enumerate(dataset, 1):
        test_id = int(row["test_id"])
        if test_id in completed:
            continue
        video_path = videos[test_id]
        frames = read_frames(video_path)
        agent.reset()
        with torch.inference_mode():
            predicted = agent.predict_actions(frames)

        with np.load(resolve(row["actions_path"])) as action_file:
            keyboard = np.asarray(
                action_file["keyboard"], dtype=np.float32)[:, :4]
            mouse = np.asarray(action_file["mouse"], dtype=np.float32)[:, :2]

        pred_fb = exclusive_labels(
            flatten_action(predicted, "back"), flatten_action(predicted, "forward"))
        pred_lr = exclusive_labels(
            flatten_action(predicted, "left"), flatten_action(predicted, "right"))
        gt_fb = exclusive_labels(keyboard[:, 1], keyboard[:, 0])
        gt_lr = exclusive_labels(keyboard[:, 2], keyboard[:, 3])
        pred_keyboard_slice, action_keyboard_slice = alignment_slices(
            args.keyboard_offset, len(pred_fb), len(keyboard))
        fb_accuracy = float(np.mean(
            pred_fb[pred_keyboard_slice] == gt_fb[action_keyboard_slice]))
        lr_accuracy = float(np.mean(
            pred_lr[pred_keyboard_slice] == gt_lr[action_keyboard_slice]))

        pred_camera = np.asarray(predicted["camera"]).reshape(-1, 2)[:76]
        pred_direction = np.asarray([camera_direction(value) for value in pred_camera])
        gt_direction = np.asarray([camera_direction(value) for value in mouse])
        pred_mouse_slice, action_mouse_slice = alignment_slices(
            args.mouse_offset, len(pred_direction), len(mouse))
        mouse_accuracy = float(np.mean(
            pred_direction[pred_mouse_slice] == gt_direction[action_mouse_slice]))
        result = {
            "test_id": test_id,
            "clip_idx": int(row["clip_idx"]),
            "generated_video": str(video_path),
            "keyboard_accuracy": f"{(fb_accuracy + lr_accuracy) * 0.5:.8f}",
            "mouse_accuracy": f"{mouse_accuracy:.8f}",
            "forward_back_accuracy": f"{fb_accuracy:.8f}",
            "left_right_accuracy": f"{lr_accuracy:.8f}",
        }
        rows.append(result)
        write_rows(output, rows)
        print(f"rank={args.rank} sample={position}/{len(dataset)} test_id={test_id} "
              f"keyboard={result['keyboard_accuracy']} mouse={result['mouse_accuracy']}",
              flush=True)

    summary = {
        "rank": args.rank,
        "samples": len(rows),
        "keyboard_offset": args.keyboard_offset,
        "mouse_offset": args.mouse_offset,
    }
    for field in FIELDS[3:]:
        summary[field] = float(np.mean([float(row[field]) for row in rows]))
    (out_dir / f"rank{args.rank}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
