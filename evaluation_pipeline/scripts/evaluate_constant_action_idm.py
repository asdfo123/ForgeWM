#!/usr/bin/env python3
"""Evaluate constant-action Minecraft videos with the GameWorld/VPT IDM.

The benchmark contains four translation controls (forward/back/left/right) and
two yaw controls (turn_left/turn_right).  Keyboard accuracy follows the
GameWorld convention of averaging the forward/back and left/right three-way
classification tasks.  Mouse accuracy uses GameWorld's nine-way camera
direction labels.  Command-group and yaw-sign accuracy are retained as more
direct diagnostics of the commanded action.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from action_consistency_eval import ACTIONS, parse_identity
from evaluate_missing_idm import (
    camera_direction,
    exclusive_labels,
    flatten_action,
    load_agent,
    read_frames,
)


TRANSLATION_ACTIONS = ("forward", "back", "left", "right")
TURN_ACTIONS = ("turn_left", "turn_right")
EXPECTED_KEY_LABELS = {
    "forward": (2, 0),
    "back": (1, 0),
    "left": (0, 1),
    "right": (0, 2),
}
EXPECTED_MOUSE_LABELS = {"turn_left": 3, "turn_right": 4}
EXPECTED_YAW_SIGNS = {"turn_left": -1.0, "turn_right": 1.0}
FIELDS = [
    "model",
    "scene",
    "action",
    "seed",
    "video",
    "status",
    "error",
    "frames",
    "keyboard_accuracy",
    "forward_back_accuracy",
    "left_right_accuracy",
    "command_group_accuracy",
    "mouse_accuracy",
    "yaw_sign_accuracy",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--frame-count", type=int, default=76)
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_existing(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["video"]: row for row in csv.DictReader(handle, delimiter="\t")}


def write_rows(path: Path, videos: list[Path], rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, delimiter="\t")
        writer.writeheader()
        for video in videos:
            key = str(video.resolve())
            if key in rows:
                writer.writerow({field: rows[key].get(field, "") for field in FIELDS})
    os.replace(temporary, path)


def score_prediction(predicted: dict, action: str, frame_count: int) -> dict[str, float | str]:
    pred_fb = exclusive_labels(
        flatten_action(predicted, "back", frame_count),
        flatten_action(predicted, "forward", frame_count),
    )
    pred_lr = exclusive_labels(
        flatten_action(predicted, "left", frame_count),
        flatten_action(predicted, "right", frame_count),
    )
    result: dict[str, float | str] = {
        "keyboard_accuracy": "",
        "forward_back_accuracy": "",
        "left_right_accuracy": "",
        "command_group_accuracy": "",
        "mouse_accuracy": "",
        "yaw_sign_accuracy": "",
    }
    if action in TRANSLATION_ACTIONS:
        expected_fb, expected_lr = EXPECTED_KEY_LABELS[action]
        fb = float(np.mean(pred_fb == expected_fb))
        lr = float(np.mean(pred_lr == expected_lr))
        result.update(
            keyboard_accuracy=(fb + lr) * 0.5,
            forward_back_accuracy=fb,
            left_right_accuracy=lr,
            command_group_accuracy=fb if action in ("forward", "back") else lr,
        )
    else:
        camera = np.asarray(predicted["camera"]).reshape(-1, 2)[:frame_count]
        directions = np.asarray([camera_direction(value) for value in camera])
        horizontal = camera[:, 1]
        result.update(
            mouse_accuracy=float(np.mean(directions == EXPECTED_MOUSE_LABELS[action])),
            yaw_sign_accuracy=float(
                np.mean(EXPECTED_YAW_SIGNS[action] * horizontal > 1e-2)
            ),
        )
    return result


def mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key, "") not in ("", None)]
    return float(np.mean(values)) if values else None


def write_summary(output_dir: Path, model_name: str, rows: list[dict], frame_count: int) -> None:
    successful = [row for row in rows if row["status"] == "ok"]
    translations = [row for row in successful if row["action"] in TRANSLATION_ACTIONS]
    turns = [row for row in successful if row["action"] in TURN_ACTIONS]
    by_action = {}
    for action in ACTIONS:
        subset = [row for row in successful if row["action"] == action]
        by_action[action] = {
            "videos": len(subset),
            "keyboard_accuracy": mean(subset, "keyboard_accuracy"),
            "command_group_accuracy": mean(subset, "command_group_accuracy"),
            "mouse_accuracy": mean(subset, "mouse_accuracy"),
            "yaw_sign_accuracy": mean(subset, "yaw_sign_accuracy"),
        }
    payload = {
        "model": model_name,
        "requested_videos": len(rows),
        "successful_videos": len(successful),
        "track_success_rate": len(successful) / len(rows) if rows else 0.0,
        "frame_count": frame_count,
        "keyboard_accuracy": mean(translations, "keyboard_accuracy"),
        "command_group_accuracy": mean(translations, "command_group_accuracy"),
        "mouse_accuracy": mean(turns, "mouse_accuracy"),
        "yaw_sign_accuracy": mean(turns, "yaw_sign_accuracy"),
        "by_action": by_action,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        f"# {model_name} constant-action IDM evaluation",
        "",
        f"Videos: {len(successful)}/{len(rows)} successful; first {frame_count} frames per video.",
        "",
        "| Keyboard Acc. | Command-group Acc. | Mouse Acc. | Yaw-sign Acc. |",
        "| ---: | ---: | ---: | ---: |",
        f"| {payload['keyboard_accuracy']:.4f} | {payload['command_group_accuracy']:.4f} | "
        f"{payload['mouse_accuracy']:.4f} | {payload['yaw_sign_accuracy']:.4f} |",
        "",
        "`Keyboard Acc.` averages the forward/back and left/right three-way tasks, following GameWorld.",
        "`Mouse Acc.` uses GameWorld's exact nine-way camera-direction label; yaw-sign accuracy only checks left/right sign.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    videos = sorted(args.videos_dir.resolve().glob("*.mp4"))
    if args.max_videos > 0:
        videos = videos[: args.max_videos]
    if not videos:
        raise RuntimeError(f"No MP4 videos found in {args.videos_dir}")

    output_tsv = args.output_dir / "per_video.tsv"
    rows = {} if args.overwrite else read_existing(output_tsv)
    pending = [video for video in videos if rows.get(str(video.resolve()), {}).get("status") != "ok"]
    print(f"inputs={len(videos)} complete={len(videos) - len(pending)} pending={len(pending)}", flush=True)
    if pending:
        agent = load_agent(args.model, args.weights, args.device)
        for index, video in enumerate(pending, start=1):
            scene, action, seed = parse_identity(str(video))
            row: dict[str, object] = {
                "model": args.model_name,
                "scene": scene,
                "action": action,
                "seed": seed,
                "video": str(video.resolve()),
                "status": "failed",
                "error": "",
                "frames": args.frame_count,
            }
            try:
                frames = read_frames(video, count=args.frame_count)
                agent.reset()
                with torch.inference_mode():
                    predicted = agent.predict_actions(frames)
                row.update(score_prediction(predicted, action, args.frame_count))
                row["status"] = "ok"
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}".replace("\t", " ")
            rows[str(video.resolve())] = row
            write_rows(output_tsv, videos, rows)
            print(
                f"[{index}/{len(pending)}] {video.name} status={row['status']} "
                f"key={row.get('keyboard_accuracy', '')} mouse={row.get('mouse_accuracy', '')}",
                flush=True,
            )
    ordered = [rows[str(video.resolve())] for video in videos]
    write_summary(args.output_dir, args.model_name, ordered, args.frame_count)
    if any(row["status"] != "ok" for row in ordered):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
