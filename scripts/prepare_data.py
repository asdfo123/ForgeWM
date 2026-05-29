#!/usr/bin/env python
"""
Prepare GF-Minecraft data_2003 as *sharded* LMDB at MG2's NATIVE 352×640
resolution (latent 44×80), with the GF pitch sign flipped so it matches
MG2's mouse convention.

Differences vs prepare_action_lmdb_shard.py (the 480p version):

  * target_h/target_w defaults: 480×832 -> 352×640
  * aspect-preserving resize + center-crop (mirrors MG2's _resizecrop in
    inference_base_single_action.py) instead of pure bilinear stretch.
    640/352 = 1.818  vs  832/480 = 1.733; bilinear stretch would squash
    the aspect ratio slightly.
  * mouse_cond[:, 0] = -pitch_delta  (GF convention: +pitch = look-down;
    MG2 convention: mouse[0] > 0 = camera-up / look-up).
  * Output shape check: latent should be (21, 16, 44, 80) instead of
    (21, 16, 60, 104).  Printed once per shard.
  * IncrementalShardWriter resumes from existing `latents_shape[0]` so you
    can re-run the same shard and append new clips instead of starting at
    index 0.

Usage (same shard slicing as the 480p script):

    CUDA_VISIBLE_DEVICES=0 python prepare_action_lmdb_shard_360p.py \\
        --data_dir    ... \\
        --output_dir  ... \\
        --vae_path    ckpts/MG2-base/Wan2.1_VAE.pth \\
        --video_start 0   --video_end 501 \\
        --num_clips_per_video 20 \\
        --flush_every 20
"""
import argparse
import csv as csvmod
import glob
import json
import os
import sys

import lmdb
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.lmdb_ import store_arrays_to_lmdb


# ---------------------------------------------------------------------------
# Frame loading + action parsing (360p variant)


def _resize_crop_tensor(frames, target_h, target_w):
    """Aspect-preserving resize + center-crop, matching MG2's _resizecrop.

    frames: [T, C, H, W] float in [0, 1].  Returns [T, C, target_h, target_w].
    """
    _, _, h, w = frames.shape
    # Choose scale so that the shorter side matches target and the longer
    # side is >= target (then center crop).
    if h / w > target_h / target_w:
        new_w = target_w
        new_h = int(round(h * target_w / w))
    else:
        new_h = target_h
        new_w = int(round(w * target_h / h))
    frames = torch.nn.functional.interpolate(
        frames, size=(new_h, new_w), mode='bilinear', align_corners=False,
    )
    top = (new_h - target_h) // 2
    left = (new_w - target_w) // 2
    return frames[:, :, top:top + target_h, left:left + target_w]


def load_video_frames(video_path, start_frame, num_frames, target_h=352, target_w=640):
    import decord
    decord.bridge.set_bridge('torch')
    vr = decord.VideoReader(video_path)
    end_frame = min(start_frame + num_frames, len(vr))
    if end_frame - start_frame < num_frames:
        return None
    indices = list(range(start_frame, end_frame))
    frames = vr.get_batch(indices)                          # [T, H, W, C] uint8
    frames = frames.permute(0, 3, 1, 2).float() / 255.0     # [T, C, H, W]
    frames = _resize_crop_tensor(frames, target_h, target_w)
    frames = frames * 2.0 - 1.0                              # [-1, 1]
    return frames


def parse_actions(metadata, start_frame, num_frames):
    """Return (mouse [F,2] pitch/yaw deltas, keyboard [F,4] WSAD one-hots).

    360p variant: pitch sign is flipped because GF-Minecraft uses
    pitch_delta > 0 for look-down but MG2 uses mouse[0] > 0 for look-up.
    yaw is left unchanged (both conventions agree: +yaw = turn right).
    """
    actions = metadata['actions']
    mouse_cond = np.zeros((num_frames, 2), dtype=np.float32)
    keyboard_cond = np.zeros((num_frames, 4), dtype=np.float32)
    for i in range(num_frames):
        key = str(start_frame + i)
        act = actions.get(key)
        if act is None:
            continue
        # *** pitch sign flip: GF (+ = down) -> MG2 (+ = up) ***
        mouse_cond[i, 0] = -float(act.get('pitch_delta', 0.0))
        mouse_cond[i, 1] = float(act.get('yaw_delta', 0.0))
        ws = int(act.get('ws', 0))
        ad = int(act.get('ad', 0))
        keyboard_cond[i, 0] = 1.0 if ws == 1 else 0.0
        keyboard_cond[i, 1] = 1.0 if ws == 2 else 0.0
        keyboard_cond[i, 2] = 1.0 if ad == 1 else 0.0
        keyboard_cond[i, 3] = 1.0 if ad == 2 else 0.0
    return mouse_cond, keyboard_cond


def get_prompt_from_annotation(annotation_csv, video_name):
    if annotation_csv is None:
        return "A Minecraft gameplay video in first person perspective."
    for row in annotation_csv:
        if video_name in row[0]:
            return row[3] if len(row) > 3 else "A Minecraft gameplay video."
    return "A Minecraft gameplay video."


# ---------------------------------------------------------------------------
# Incremental LMDB writer (appends to existing shard if one exists).

class IncrementalShardWriter:
    def __init__(self, env, flush_every):
        self.env = env
        self.flush_every = flush_every
        self.start_index = self._read_existing_count()
        self._reset_buffers()

    def _read_existing_count(self):
        """Return how many clips are already in this LMDB so we append."""
        with self.env.begin(write=False) as txn:
            raw = txn.get(b'latents_shape')
        if raw is None:
            return 0
        try:
            return int(raw.decode().split()[0])
        except Exception:
            return 0

    def _reset_buffers(self):
        self.lat_buf, self.mouse_buf, self.kbd_buf, self.prompt_buf = [], [], [], []

    def add(self, latent_np, mouse_np, kbd_np, prompt):
        self.lat_buf.append(latent_np)
        self.mouse_buf.append(mouse_np)
        self.kbd_buf.append(kbd_np)
        self.prompt_buf.append(prompt)
        if len(self.lat_buf) >= self.flush_every:
            self.flush()

    def flush(self):
        if not self.lat_buf:
            return
        lat = np.stack(self.lat_buf, axis=0)
        mouse = np.stack(self.mouse_buf, axis=0)
        kbd = np.stack(self.kbd_buf, axis=0)
        prompts = np.array(self.prompt_buf, dtype=object)
        store_arrays_to_lmdb(
            self.env,
            {
                'latents': lat,
                'prompts': prompts,
                'mouse_conditions': mouse,
                'keyboard_conditions': kbd,
            },
            start_index=self.start_index,
        )
        self.start_index += len(self.lat_buf)
        self._reset_buffers()

    def finalize_shapes(self, latent_shape, mouse_shape, kbd_shape):
        with self.env.begin(write=True) as txn:
            txn.put(
                b'latents_shape',
                ' '.join(map(str, (self.start_index, *latent_shape))).encode(),
            )
            txn.put(b'prompts_shape', str(self.start_index).encode())
            txn.put(
                b'mouse_conditions_shape',
                ' '.join(map(str, (self.start_index, *mouse_shape))).encode(),
            )
            txn.put(
                b'keyboard_conditions_shape',
                ' '.join(map(str, (self.start_index, *kbd_shape))).encode(),
            )


# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', required=True,
                   help='Root with video/ and metadata/ subdirs, plus annotation.csv.')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--vae_path', default='ckpts/MG2-base/Wan2.1_VAE.pth')
    p.add_argument('--num_clips_per_video', type=int, default=20)
    p.add_argument('--target_h', type=int, default=352)   # MG2 native
    p.add_argument('--target_w', type=int, default=640)   # MG2 native
    p.add_argument('--raw_frames_per_clip', type=int, default=81)
    p.add_argument('--device', type=str, default='cuda:0')
    p.add_argument('--video_start', type=int, default=0)
    p.add_argument('--video_end',   type=int, default=-1)
    p.add_argument('--flush_every', type=int, default=50)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    np.random.seed(args.seed)
    device = torch.device(args.device)
    num_raw_frames = args.raw_frames_per_clip

    print(f"Loading VAE from {args.vae_path}...")
    from wan.modules.vae import _video_vae
    vae = _video_vae(pretrained_path=args.vae_path, z_dim=16).eval().to(device)
    vae.requires_grad_(False)

    mean = torch.tensor([
        -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
        0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
    ], dtype=torch.float32, device=device)
    std = torch.tensor([
        2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
        3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
    ], dtype=torch.float32, device=device)
    scale = [mean, 1.0 / std]

    video_subdir = os.path.join(args.data_dir, 'video')
    meta_subdir = os.path.join(args.data_dir, 'metadata')
    if os.path.isdir(video_subdir) and os.path.isdir(meta_subdir):
        video_files = sorted(glob.glob(os.path.join(video_subdir, '*.mp4')))
        meta_dir = meta_subdir
    else:
        video_files = sorted(glob.glob(os.path.join(args.data_dir, '*.mp4')))
        meta_dir = args.data_dir

    if args.video_end < 0 or args.video_end > len(video_files):
        args.video_end = len(video_files)
    video_files = video_files[args.video_start:args.video_end]
    print(f"Processing {len(video_files)} videos (slice [{args.video_start}:{args.video_end}])"
          f"  meta_dir={meta_dir}")
    print(f"Target resolution: {args.target_h}x{args.target_w} (MG2 native 352x640)")
    print("Pitch sign: FLIPPED (GF +pitch = look-down -> MG2 mouse[0] < 0)")

    annotation_csv = None
    for cand in (os.path.join(args.data_dir, 'annotation.csv'),
                 os.path.join(os.path.dirname(args.data_dir), 'annotation.csv')):
        if os.path.exists(cand):
            with open(cand) as f:
                annotation_csv = list(csvmod.reader(f))
            print(f"Loaded annotation.csv  rows={len(annotation_csv)}")
            break

    os.makedirs(args.output_dir, exist_ok=True)
    env = lmdb.open(args.output_dir, map_size=int(1e12), writemap=False)
    writer = IncrementalShardWriter(env, flush_every=args.flush_every)
    if writer.start_index > 0:
        print(f"[append] existing shard has {writer.start_index} clips; "
              f"new writes will start at that index.")

    latent_row_shape, mouse_row_shape, kbd_row_shape = None, None, None
    skipped_short = skipped_missing = clip_fail = 0
    first_shape_printed = False

    for video_path in tqdm(video_files, desc="videos"):
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        json_path = os.path.join(meta_dir, f"{video_name}.json")
        if not os.path.exists(json_path):
            skipped_missing += 1
            continue

        with open(json_path) as f:
            metadata = json.load(f)
        total_frames = len(metadata['actions'])
        if total_frames < num_raw_frames + 10:
            skipped_short += 1
            continue

        prompt = get_prompt_from_annotation(annotation_csv, video_name)

        max_start = total_frames - num_raw_frames
        if args.num_clips_per_video >= max_start:
            starts = list(range(1, max_start, max(1, max_start // args.num_clips_per_video)))
        else:
            starts = np.random.choice(range(1, max_start),
                                      size=args.num_clips_per_video, replace=False)
            starts = sorted(int(s) for s in starts)

        for start_frame in starts:
            frames = load_video_frames(
                video_path, start_frame, num_raw_frames,
                target_h=args.target_h, target_w=args.target_w,
            )
            if frames is None:
                clip_fail += 1
                continue

            mouse_np, kbd_np = parse_actions(metadata, start_frame, num_raw_frames)

            frames_5d = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device, dtype=torch.float32)
            with torch.no_grad():
                latent = vae.encode(frames_5d, scale).float().squeeze(0)
                latent = latent.permute(1, 0, 2, 3).contiguous()   # [T_lat, C, h, w]
            latent_np = latent.cpu().half().numpy()

            if latent_row_shape is None:
                latent_row_shape = latent_np.shape
                mouse_row_shape = mouse_np.shape
                kbd_row_shape = kbd_np.shape

            if not first_shape_printed:
                expected_latent = (21, 16, args.target_h // 8, args.target_w // 8)
                print(f"[shape] first clip: frames={tuple(frames.shape)} "
                      f"latent={latent_np.shape} (expected {expected_latent}) "
                      f"mouse={mouse_np.shape} kbd={kbd_np.shape}")
                if tuple(latent_np.shape) != expected_latent:
                    print(f"[shape][WARN] latent shape {latent_np.shape} != expected {expected_latent}")
                first_shape_printed = True

            writer.add(latent_np, mouse_np, kbd_np, prompt)

    writer.flush()
    if latent_row_shape is not None:
        writer.finalize_shapes(latent_row_shape, mouse_row_shape, kbd_row_shape)
    env.sync(); env.close()

    print(f"\nDone.  total_in_shard={writer.start_index} clips  "
          f"missing_json={skipped_missing}  too_short={skipped_short}  frame_fail={clip_fail}")
    print(f"LMDB at: {args.output_dir}")


if __name__ == '__main__':
    main()
