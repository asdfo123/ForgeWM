#!/usr/bin/env python
"""Custom metrics for the ForgeWM / MG2 / HY-WorldPlay bakeoff.

Per-video metrics (written per video, averaged into a model score):
  - LPIPS : mean pairwise LPIPS between consecutive frames (temporal change)
  - Flow  : mean optical-flow magnitude (Farneback, weight-free) — motion intensity
  - Photo : mean per-frame CLIP-IQA-style photorealism (CLIP ViT-L/14, laion mirror)
  - Depth : mean adjacent-frame depth-map cosine similarity (Depth-Anything-V2/3)

Distribution-level metrics (one value per model, computed over all its videos):
  - FVD   : Fréchet distance between V-JEPA2 feature distributions of
            generated vs. real reference clips
  - JEPA  : distribution compactness — mean intra-set feature cosine
            similarity of the generated set vs. that of the real set
            (lower = more diverse / closer to real spread)

All videos are center-cropped + resized to a common (H,W) before feature
extraction so models of different native resolution compare fairly.

Models (all on gpfs2 mirror, worker-visible):
  V-JEPA2  : facebook/vjepa2-vitl-fpc64-256
  DA3      : depth-anything/DA3-GIANT-1.1
  CLIP     : laion/CLIP-ViT-L-14-laion2B-s32B-b82K  (for CLIP-IQA "Photo")
  LPIPS    : pip lpips, AlexNet backbone random-init (pnet_rand=True — ImageNet
             weights are network-blocked; linear calibration weights bundled)
"""
import argparse
import json
import os
import sys
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import cv2
import imageio.v2 as imageio
from PIL import Image


# ─── video loading ──────────────────────────────────────────────────────────
def load_frames(path, max_frames=None):
    """Load video → list of (H,W,3) uint8 RGB frames."""
    r = imageio.get_reader(path)
    frames = []
    for i, fr in enumerate(r):
        if fr.ndim == 2:
            fr = np.stack([fr] * 3, axis=-1)
        elif fr.shape[2] == 4:
            fr = fr[..., :3]
        frames.append(fr)
        if max_frames and i + 1 >= max_frames:
            break
    r.close()
    return frames


def to_tensor_chw(frames_uint8, device, dtype=torch.float32):
    """(N,H,W,3) uint8 → (N,3,H,W) float in [0,1]."""
    arr = np.stack(frames_uint8)
    t = torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous().to(device=device, dtype=dtype)
    return t / 255.0


def center_crop_resize(frames_uint8, out_h, out_w):
    """Center-crop to common aspect then resize each frame to (out_h,out_w)."""
    h, w = frames_uint8[0].shape[:2]
    target_ar = out_w / out_h
    ar = w / h
    if ar > target_ar:
        new_w = int(h * target_ar)
        x0 = (w - new_w) // 2
        crop = [0, x0, h, x0 + new_w]
    else:
        new_h = int(w / target_ar)
        y0 = (h - new_h) // 2
        crop = [y0, 0, y0 + new_h, w]
    out = []
    for fr in frames_uint8:
        c = fr[crop[0]:crop[2], crop[1]:crop[3]]
        c = cv2.resize(c, (out_w, out_h), interpolation=cv2.INTER_AREA)
        out.append(c)
    return out


# ─── per-video metrics ──────────────────────────────────────────────────────
def metric_lpips(frames_t, lpips_fn):
    """frames_t: (N,3,H,W) float [0,1]. Returns mean consecutive-frame LPIPS."""
    if frames_t.shape[0] < 2:
        return 0.0
    with torch.no_grad():
        a = frames_t[:-1]
        b = frames_t[1:]
        d = lpips_fn(a, b).flatten()
    return float(d.mean().item())


def compute_flow_mags(frames_uint8):
    """Per-pair mean Farneback optical-flow magnitudes."""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames_uint8]
    mags = []
    for i in range(len(grays) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            grays[i], grays[i + 1], None,
            pyr_scale=0.5, levels=3, winsize=15, iterations=3,
            poly_n=5, poly_sigma=1.2, flags=0)
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        mags.append(float(mag.mean()))
    return mags


def metric_flow(frames_uint8):
    """Mean Farneback optical-flow magnitude between consecutive frames
    (motion intensity, continuous)."""
    mags = compute_flow_mags(frames_uint8)
    return float(np.mean(mags)) if mags else 0.0


def metric_dynamic_degree(frames_uint8, thresh=0.5):
    """VBench-style dynamic degree (Farneback substitute for RAFT).
    Fraction of frame-pairs whose mean flow magnitude exceeds ``thresh``,
    reported as a [0,1] score (higher = more dynamic motion)."""
    mags = compute_flow_mags(frames_uint8)
    if not mags:
        return 0.0
    arr = np.array(mags)
    return float((arr > thresh).mean())


def metric_photo(frames_t, clipiqa):
    """Mean per-frame CLIP-IQA photorealism score."""
    with torch.no_grad():
        scores = clipiqa(frames_t).flatten()
    return float(scores.mean().item())


# ─── CLIP-IQA (photorealism, no pyiqa download needed) ──────────────────────
class CLIPIQA:
    """CLIP-IQA-style no-reference image quality via prompt-pair probability.

    Uses a CLIP ViT-L/14 from the gpfs2 laion mirror. For each frame, computes
    P(good photo) / (P(good photo) + P(bad photo)) over a small prompt ensemble
    — higher = more photorealistic. Mirrors the clipiqa (non-plus) recipe.
    """

    GOOD = ["good photo", "high quality photo", "sharp photo", "realistic photo"]
    BAD = ["bad photo", "low quality photo", "blurry photo", "unrealistic photo"]

    def __init__(self, model_dir, device):
        from transformers import CLIPModel, CLIPProcessor
        self.model = CLIPModel.from_pretrained(model_dir, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model_dir)
        self.device = device
        with torch.no_grad():
            gt = self.proc(text=self.GOOD, return_tensors="pt", padding=True).to(device)
            bt = self.proc(text=self.BAD, return_tensors="pt", padding=True).to(device)
            self.good_emb = F.normalize(self.model.get_text_features(**gt), dim=-1)  # (G,D)
            self.bad_emb = F.normalize(self.model.get_text_features(**bt), dim=-1)   # (B,D)

    @torch.no_grad()
    def __call__(self, frames_t):
        """frames_t: (N,3,H,W) float [0,1] → (N,) scores in [0,1]."""
        # CLIP expects pixel values normalized by its mean/std; processor does that
        # from uint8/PIL. Convert back to uint8 PIL for the processor.
        N = frames_t.shape[0]
        scores = []
        for i in range(N):
            arr = (frames_t[i].permute(1, 2, 0).float().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            inp = self.proc(images=Image.fromarray(arr), return_tensors="pt").to(self.device, torch.float16)
            img_emb = F.normalize(self.model.get_image_features(**inp), dim=-1)  # (1,D)
            g = (img_emb @ self.good_emb.T).mean(dim=1)  # mean sim to good prompts
            b = (img_emb @ self.bad_emb.T).mean(dim=1)
            p = g.softmax(0)[0] / (g.softmax(0)[0] + b.softmax(0)[0] + 1e-8)  # bounded
            # simpler: sigmoid of (g-b)
            s = torch.sigmoid(g - b)
            scores.append(float(s.item()))
        return torch.tensor(scores, device=self.device)


def metric_depth(frames_uint8, depth_fn, device):
    """Mean adjacent-frame depth-map cosine similarity (temporal depth consistency).
    depth_fn(frames_uint8) → (N,1,Hd,Wd) float depth maps."""
    with torch.no_grad():
        depths = depth_fn(frames_uint8, device)  # (N,1,Hd,Wd)
    if depths.shape[0] < 2:
        return 1.0
    a = depths[:-1].flatten(1)
    b = depths[1:].flatten(1)
    cos = F.cosine_similarity(a, b, dim=1)
    return float(cos.mean().item())


# ─── depth-anything wrapper ─────────────────────────────────────────────────
class DepthAnything:
    """Wraps Depth-Anything-3 (DA3-GIANT-1.1) via the depth_anything_3 library.
    Returns per-frame depth maps for temporal-consistency scoring."""

    def __init__(self, model_dir, device):
        # DA3's api.py imports several 3D-export deps (pycolmap, trimesh, evo,
        # plyfile, moviepy) at module load; they're installed but only the
        # depth-estimation path is used here.
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained(model_dir).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, frames_uint8, device):
        imgs = [Image.fromarray(f) for f in frames_uint8]
        pred = self.model.inference(imgs, export_format="mini_npz")
        depths = pred.depth  # (N, H, W) numpy
        if isinstance(depths, np.ndarray):
            depths = torch.from_numpy(depths)
        depths = depths.to(device).float()
        if depths.ndim == 2:
            depths = depths.unsqueeze(0)
        # resize to the original frame size for fair pairwise comparison
        depths = F.interpolate(
            depths.unsqueeze(1), size=(frames_uint8[0].shape[:2]),
            mode="bilinear", align_corners=False)
        return depths  # (N,1,H,W)


# ─── V-JEPA2 feature extractor ──────────────────────────────────────────────
class VJEPA2:
    """V-JEPA2 video feature extractor. Returns one (D,) feature vector per
    video clip (mean-pooled token embedding). Manual preprocessing (no
    AutoVideoProcessor — it fails to infer channel format from tensors)."""

    # ImageNet normalization (matches video_preprocessor_config.json)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self, model_dir, device, n_frames=16, size=256):
        from transformers import VJEPA2Model
        self.model = VJEPA2Model.from_pretrained(model_dir).to(device).eval()
        self.device = device
        # tubelet_size=2 → temporal length must be divisible by 2
        self.n_frames = n_frames if n_frames % 2 == 0 else n_frames + 1
        self.size = size
        mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1, 1)
        std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1, 1)
        self.mean = mean
        self.std = std

    @torch.no_grad()
    def __call__(self, frames_uint8):
        """frames_uint8: list of (H,W,3) uint8. Returns (D,) numpy vector."""
        n = len(frames_uint8)
        idx = np.linspace(0, n - 1, self.n_frames).astype(int)
        sampled = [frames_uint8[i] for i in idx]
        # resize each to (size,size)
        sampled = [cv2.resize(f, (self.size, self.size),
                              interpolation=cv2.INTER_AREA) for f in sampled]
        clip = np.stack(sampled)  # (T,H,W,3) uint8
        # VJEPA2 expects (B, T, C, H, W); it permutes to (B,C,T,H,W) internally
        clip_t = torch.from_numpy(clip).permute(0, 3, 1, 2).unsqueeze(0)
        clip_t = clip_t.float().to(self.device) / 255.0
        # normalize: mean/std shaped for (B,T,C,H,W) → broadcast over (B,T,C,1,1)
        mean = self.mean.view(1, 1, 3, 1, 1)
        std = self.std.view(1, 1, 3, 1, 1)
        clip_t = (clip_t - mean) / std
        out = self.model(clip_t)
        # last_hidden_state: (B, T', D) → mean pool over tokens
        h = out.last_hidden_state
        feat = h.mean(dim=1).flatten()  # (D,)
        return feat.float().cpu().numpy()


def frechet_distance(mu1, sig1, mu2, sig2, eps=1e-6):
    """Standard FVD = ||mu1-mu2||^2 + Tr(sig1+sig2-2*sqrt(sig1*sig2))."""
    diff = mu1 - mu2
    covmean, _ = scipy_sqrt_product(sig1, sig2, eps)
    return float(diff @ diff + np.trace(sig1) + np.trace(sig2) - 2 * np.trace(covmean))


def scipy_sqrt_product(sig1, sig2, eps):
    from scipy.linalg import sqrtm
    prod = sig1 @ sig2
    if np.iscomplexobj(prod):
        prod = prod.real
    result, _ = sqrtm(prod, disp=False)
    if np.iscomplexobj(result):
        result = result.real
    return result, None


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bakeoff_root", default="output/eval_bakeoff")
    ap.add_argument("--real_clips_dir", default="output/eval_bakeoff/real_clips")
    ap.add_argument("--models", nargs="+",
                    default=["forgewm", "mg2", "hyworld"])
    ap.add_argument("--vjepa_dir", default="models--facebook--vjepa2-vitl-fpc64-256")
    ap.add_argument("--depth_dir", default="models--depth-anything--DA3-GIANT-1.1")
    ap.add_argument("--clip_dir", default="models--laion--CLIP-ViT-L-14-laion2B-s32B-b82K")
    ap.add_argument("--mirror", default=os.environ.get(
        "MODEL_CACHE_ROOT", str(Path.home() / ".cache/huggingface/hub")))
    ap.add_argument("--out_h", type=int, default=352)
    ap.add_argument("--out_w", type=int, default=640)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max_videos_per_model", type=int, default=200)
    ap.add_argument("--max_real", type=int, default=300)
    args = ap.parse_args()

    device = torch.device(args.device)
    os.makedirs(os.path.join(args.bakeoff_root, "custom_metrics"), exist_ok=True)

    # Resolve model dirs (follow symlinks into gpfs2 mirror)
    vjepa_dir = args.vjepa_dir if os.path.isdir(args.vjepa_dir) else os.path.join(args.mirror, args.vjepa_dir)
    depth_dir = args.depth_dir if os.path.isdir(args.depth_dir) else os.path.join(args.mirror, args.depth_dir)
    clip_dir = args.clip_dir if os.path.isdir(args.clip_dir) else os.path.join(args.mirror, args.clip_dir)
    print(f"V-JEPA2: {vjepa_dir}\nDepth:   {depth_dir}\nCLIP:    {clip_dir}")

    # ── Load models ─────────────────────────────────────────────────────────
    import lpips
    print("Loading LPIPS / CLIP-IQA / V-JEPA2 / Depth-Anything ...")
    # pnet_rand=True: random-init AlexNet backbone (no ImageNet-weight download,
    # which is blocked by the network proxy). LPIPS linear calibration weights
    # (v0.1/alex.pth) are bundled with the lpips package, so this needs no network.
    lpips_fn = lpips.LPIPS(net="alex", pnet_rand=True).to(device).eval()
    iqa_fn = CLIPIQA(clip_dir, device)
    vjepa = VJEPA2(vjepa_dir, device)
    depth_fn = DepthAnything(depth_dir, device)

    # ── Collect video lists per model ───────────────────────────────────────
    def collect(model_dir):
        video_dir = os.path.join(model_dir, "eval_videos")
        if not glob.glob(os.path.join(video_dir, "*.mp4")):
            video_dir = os.path.join(model_dir, "videos")
        vids = []
        for ext in ("*.mp4", "*/*.mp4", "*-0.mp4"):
            vids = glob.glob(os.path.join(video_dir, ext))
            if vids:
                break
        return sorted(vids)

    model_videos = {}
    for m in args.models:
        mdir = os.path.join(args.bakeoff_root, m)
        vids = collect(mdir)[:args.max_videos_per_model]
        model_videos[m] = vids
        print(f"  {m}: {len(vids)} videos")

    real_videos = sorted(glob.glob(os.path.join(args.real_clips_dir, "*.mp4")))[:args.max_real]
    print(f"  real: {len(real_videos)} reference clips")

    # ── Per-video metrics + collect V-JEPA2 features ────────────────────────
    per_video = {m: [] for m in args.models}
    feats = {m: [] for m in args.models}
    real_feats = []

    def process_one(path, store_feats, label):
        frames = load_frames(path)
        if len(frames) < 2:
            print(f"  skip (too short): {path}")
            return None
        frames_cr = center_crop_resize(frames, args.out_h, args.out_w)
        ft = to_tensor_chw(frames_cr, device)
        row = {
            "video": path,
            "lpips": metric_lpips(ft, lpips_fn),
            "flow": metric_flow(frames_cr),
            "dynamic_degree": metric_dynamic_degree(frames_cr),
            "photo": metric_photo(ft, iqa_fn),
            "depth": metric_depth(frames_cr, depth_fn, device),
        }
        feat = vjepa(frames_cr)
        if store_feats is not None:
            store_feats.append(feat)
        return row

    for m in args.models:
        print(f"\n=== Per-video metrics: {m} ({len(model_videos[m])}) ===")
        for i, v in enumerate(model_videos[m]):
            row = process_one(v, feats[m], m)
            if row is not None:
                per_video[m].append(row)
            if (i + 1) % 10 == 0:
                print(f"  {m} {i+1}/{len(model_videos[m])}")

    real_cache = os.path.join(
        args.real_clips_dir, f".vjepa2_features_{len(real_videos)}.npy"
    )
    if os.path.isfile(real_cache):
        real_feats = list(np.load(real_cache))
        print(f"\nLoaded {len(real_feats)} cached real V-JEPA2 features: {real_cache}")
    else:
        print(f"\n=== V-JEPA2 features: real ({len(real_videos)}) ===")
        # Distribution metrics only need V-JEPA2 embeddings for real clips.
        # Avoid spending most of the run on unused LPIPS/flow/photo/depth scores.
        for i, v in enumerate(real_videos):
            frames = load_frames(v)
            if len(frames) < 2:
                continue
            frames_cr = center_crop_resize(frames, args.out_h, args.out_w)
            real_feats.append(vjepa(frames_cr))
            if (i + 1) % 25 == 0:
                print(f"  real {i+1}/{len(real_videos)}")
        np.save(real_cache, np.stack(real_feats))
        print(f"Saved real V-JEPA2 feature cache: {real_cache}")

    # ── Distribution-level: FVD + JEPA ─────────────────────────────────────
    print("\n=== Distribution metrics (FVD, JEPA) ===")
    real_feats = np.stack(real_feats) if real_feats else np.zeros((0, 1))
    real_mu = real_feats.mean(0) if len(real_feats) else np.zeros(1)
    real_sig = np.cov(real_feats, rowvar=False) if len(real_feats) > 1 else np.zeros((1, 1))
    # JEPA: real intra-set mean cosine similarity (compactness reference)
    def mean_cos(mat):
        if len(mat) < 2:
            return 0.0
        n = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
        sim = n @ n.T
        iu = np.triu_indices(n.shape[0], k=1)
        return float(sim[iu].mean())
    real_cos = mean_cos(real_feats)

    dist_metrics = {}
    for m in args.models:
        f = feats[m]
        if len(f) < 2:
            dist_metrics[m] = {"fvd": float("nan"), "jepa": float("nan")}
            continue
        f = np.stack(f)
        mu = f.mean(0)
        sig = np.cov(f, rowvar=False)
        fvd = frechet_distance(mu, sig, real_mu, real_sig)
        gen_cos = mean_cos(f)
        # JEPA score: how close the generated compactness is to the real one.
        # |gen_cos - real_cos| → 0 is better; report 1 - |diff| so higher=better.
        jepa = 1.0 - abs(gen_cos - real_cos)
        dist_metrics[m] = {"fvd": fvd, "jepa": jepa,
                           "gen_cos": gen_cos, "real_cos": real_cos}
        print(f"  {m}: FVD={fvd:.4f} JEPA={jepa:.4f} "
              f"(gen_cos={gen_cos:.4f} real_cos={real_cos:.4f})")

    # ── Save ────────────────────────────────────────────────────────────────
    out = {
        "per_video": {m: per_video[m] for m in args.models},
        "distribution": dist_metrics,
        "config": {
            "out_h": args.out_h, "out_w": args.out_w,
            "vjepa_dir": vjepa_dir, "depth_dir": depth_dir, "clip_dir": clip_dir,
            "max_videos_per_model": args.max_videos_per_model,
            "num_real": len(real_videos),
        },
    }
    out_path = os.path.join(args.bakeoff_root, "custom_metrics", "custom_metrics.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    # per-video TSV
    tsv_path = os.path.join(args.bakeoff_root, "custom_metrics", "per_video_metrics.tsv")
    with open(tsv_path, "w") as f:
        f.write("model\tvideo\tlpips\tflow\tdynamic_degree\tphoto\tdepth\n")
        for m in args.models:
            for row in per_video[m]:
                f.write(f"{m}\t{row['video']}\t{row['lpips']:.6f}\t"
                        f"{row['flow']:.6f}\t{row['dynamic_degree']:.6f}\t"
                        f"{row['photo']:.6f}\t{row['depth']:.6f}\n")

    print(f"\nSaved: {out_path}")
    print(f"Saved: {tsv_path}")


if __name__ == "__main__":
    main()
