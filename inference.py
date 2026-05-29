"""
ForgeWM Inference — Action-conditioned video generation from a reference frame.

Supports all training stages:
  - Stage 0 (bidirectional): one-shot denoising, ancestral sampling
  - Stage 1 (causal AR): chunked causal denoising with KV cache
  - Stage 3 (DMD): 4-step fast inference with KV cache

Usage:
    python inference.py \
        --checkpoint_path ckpts/stage3/model.pt \
        --image_path demo_images/forest.png \
        --action_type forward \
        --output_path output/demo.mp4
"""
import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
from tqdm import tqdm
import imageio.v2 as imageio

from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


# ─── Action Palette ───────────────────────────────────────────────────────────
CAM_VALUE = 0.10


def make_action(action_type, num_raw_frames, mouse_dim=2, keyboard_dim=6):
    """Build mouse/keyboard action tensors. Shape: [1, T, dim]."""
    mouse = torch.zeros(1, num_raw_frames, mouse_dim)
    keyboard = torch.zeros(1, num_raw_frames, keyboard_dim)

    if action_type == "forward":
        keyboard[:, :, 0] = 1.0
    elif action_type == "back":
        keyboard[:, :, 1] = 1.0
    elif action_type == "left":
        keyboard[:, :, 2] = 1.0
    elif action_type == "right":
        keyboard[:, :, 3] = 1.0
    elif action_type == "turn_right":
        mouse[:, :, 1] = CAM_VALUE
    elif action_type == "turn_left":
        mouse[:, :, 1] = -CAM_VALUE
    elif action_type == "look_up":
        mouse[:, :, 0] = CAM_VALUE
    elif action_type == "look_down":
        mouse[:, :, 0] = -CAM_VALUE
    elif action_type == "forward_turn_right":
        keyboard[:, :, 0] = 1.0
        mouse[:, :, 1] = CAM_VALUE
    elif action_type == "random":
        torch.manual_seed(42)
        mouse = (torch.rand(1, num_raw_frames, mouse_dim) - 0.5) * (2 * CAM_VALUE)
        keyboard[:, :, :4] = (torch.rand(1, num_raw_frames, 4) > 0.5).float()
    elif action_type == "no_action":
        pass
    else:
        raise ValueError(f"Unknown action: {action_type}")
    return mouse, keyboard


# ─── Condition Building ───────────────────────────────────────────────────────

def load_reference_frame(image_path, device, dtype, height=352, width=640):
    """Load and preprocess a reference image. Returns [1, 1, 3, H, W]."""
    transform = Compose([
        Resize((height, width), interpolation=InterpolationMode.BILINEAR),
        ToTensor(),
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0).unsqueeze(0).to(device=device, dtype=dtype)


def build_conditional_dict(pipeline, pixel, num_frames, mouse_cond, keyboard_cond,
                           dtype, device, vae_tcr=4):
    """Build the full conditional dict for inference (training-aligned)."""
    num_pixel_frames = (num_frames - 1) * vae_tcr + 1

    with torch.no_grad():
        visual_context = pipeline.vae.encode_visual_context_from_pixels(pixel).to(dtype)

        # Pixel-space zero-padding then VAE encode
        first_frame = pixel[:, 0:1]
        pad_pix = torch.zeros(1, num_pixel_frames - 1, 3, pixel.shape[3], pixel.shape[4],
                              device=device, dtype=dtype)
        padded = torch.cat([first_frame, pad_pix], dim=1).permute(0, 2, 1, 3, 4)
        img_cond = pipeline.vae.encode_to_latent(padded).to(dtype)

    _, F_lat, C_lat, H_lat, W_lat = img_cond.shape
    mask = torch.zeros(1, num_frames, 4, H_lat, W_lat, device=device, dtype=dtype)
    mask[:, 0:1] = 1
    cond_concat = torch.cat([mask, img_cond], dim=2)

    return {
        "visual_context": visual_context,
        "cond_concat": cond_concat,
        "mouse_condition": mouse_cond.to(device=device, dtype=dtype),
        "keyboard_condition": keyboard_cond.to(device=device, dtype=dtype),
    }


# ─── Inference Modes ──────────────────────────────────────────────────────────

def infer_causal(pipeline, conditional_dict, num_frames, height, width,
                 device, dtype, seed=0):
    """Causal inference with KV cache (Stage 1 multi-step or Stage 3 DMD)."""
    torch.manual_seed(seed)
    noise = torch.randn([1, num_frames, 16, height // 8, width // 8],
                        device=device, dtype=dtype)

    if hasattr(pipeline, 'inference'):
        video, _ = pipeline.inference(
            noise=noise,
            conditional_dict=conditional_dict,
            return_latents=True,
            return_video=True,
        )
    else:
        video = pipeline.inference(
            noise=noise,
            conditional_dict=conditional_dict,
            return_latents=False,
        )
    return video


def infer_bidirectional(generator, vae, conditional_dict, num_frames, height, width,
                        device, dtype, seed=0, sampling_steps=50, shift=5.0,
                        num_train_timesteps=1000):
    """Bidirectional one-shot inference (Stage 0). No KV cache."""
    torch.manual_seed(seed)
    noise = torch.randn([1, num_frames, 16, height // 8, width // 8],
                        device=device, dtype=dtype)

    generator.model.num_frame_per_block = num_frames
    for attr in ("block_mask", "block_mask_keyboard", "block_mask_mouse"):
        if hasattr(generator.model, attr):
            setattr(generator.model, attr, None)

    scheduler = FlowUniPCMultistepScheduler(
        num_train_timesteps=num_train_timesteps, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(sampling_steps, device=device, shift=shift)

    latents = noise
    for t in tqdm(scheduler.timesteps, desc="Denoising"):
        timestep = t * torch.ones([1, num_frames], device=device, dtype=torch.float32)
        flow_pred, _ = generator(
            noisy_image_or_video=latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )
        latents = scheduler.step(flow_pred, t, latents, return_dict=False)[0]

    video = vae.decode_to_pixel(latents)
    video = (video * 0.5 + 0.5).clamp(0, 1)
    return video


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ForgeWM Inference")
    parser.add_argument("--config_path", type=str, default="configs/stage3_dmd.yaml")
    parser.add_argument("--checkpoint_path", type=str, default="",
                        help="Model checkpoint (.pt). Empty = use base weights.")
    parser.add_argument("--image_path", type=str, required=True,
                        help="Reference frame image.")
    parser.add_argument("--action_type", type=str, default="forward",
                        choices=["forward", "back", "left", "right",
                                 "turn_right", "turn_left", "look_up", "look_down",
                                 "forward_turn_right", "random", "no_action"])
    parser.add_argument("--output_path", type=str, default="output/demo.mp4")
    parser.add_argument("--num_frames", type=int, default=21,
                        help="Number of latent frames to generate.")
    parser.add_argument("--height", type=int, default=352)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.set_grad_enabled(False)

    # Load config
    config = OmegaConf.load(args.config_path)
    if os.path.exists("configs/default.yaml"):
        default_config = OmegaConf.load("configs/default.yaml")
        config = OmegaConf.merge(default_config, config)

    is_causal = config.get("causal", True)
    is_dmd = hasattr(config, "denoising_step_list")

    # Build pipeline
    print(f"Mode: {'DMD' if is_dmd else 'causal' if is_causal else 'bidirectional'}")

    if is_causal:
        if is_dmd:
            pipeline = CausalInferencePipeline(config, device=device)
        else:
            pipeline = CausalDiffusionInferencePipeline(config, device=device)

        # Load checkpoint
        if args.checkpoint_path:
            print(f"Loading checkpoint: {args.checkpoint_path}")
            state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            if isinstance(state_dict, dict):
                gen_sd = state_dict.get("generator",
                         state_dict.get("generator_ema", state_dict))
            else:
                gen_sd = state_dict
            fixed = {k.replace("._fsdp_wrapped_module.", ".")
                      .replace("._checkpoint_wrapped_module.", "."): v
                     for k, v in gen_sd.items()}
            missing, unexpected = pipeline.generator.load_state_dict(fixed, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")

        pipeline.generator = pipeline.generator.to(device=device, dtype=dtype)
        pipeline.vae = pipeline.vae.to(device=device, dtype=dtype)
        vae = pipeline.vae
        generator = pipeline.generator
    else:
        # Bidirectional (Stage 0)
        model_kwargs = dict(getattr(config, "model_kwargs", {}))
        action_config = getattr(config, "action_config", None)
        if action_config:
            ac = dict(action_config)
            ac.pop("local_attn_size", None)
            model_kwargs["action_config"] = ac
        generator = WanDiffusionWrapper(**model_kwargs, is_causal=True)

        if args.checkpoint_path:
            print(f"Loading checkpoint: {args.checkpoint_path}")
            state_dict = torch.load(args.checkpoint_path, map_location="cpu")
            gen_sd = state_dict.get("generator",
                     state_dict.get("generator_ema", state_dict))
            fixed = {k.replace("._fsdp_wrapped_module.", ".")
                      .replace("._checkpoint_wrapped_module.", "."): v
                     for k, v in gen_sd.items()}
            generator.load_state_dict(fixed, strict=False)

        generator = generator.to(device=device, dtype=dtype).eval()
        vae = WanVAEWrapper().to(device=device, dtype=dtype).eval()

        # Shim for build_conditional_dict
        class _Shim:
            pass
        pipeline = _Shim()
        pipeline.vae = vae
        pipeline.generator = generator

    # Build conditions
    pixel = load_reference_frame(args.image_path, device, dtype, args.height, args.width)

    action_config = config.get("action_config", {})
    vae_tcr = action_config.get("vae_time_compression_ratio", 4) if action_config else 4
    num_raw_frames = (args.num_frames - 1) * vae_tcr + 1

    mouse_cond, keyboard_cond = make_action(
        args.action_type, num_raw_frames,
        mouse_dim=action_config.get("mouse_dim_in", 2) if action_config else 2,
        keyboard_dim=action_config.get("keyboard_dim_in", 6) if action_config else 6,
    )

    conditional_dict = build_conditional_dict(
        pipeline, pixel, args.num_frames, mouse_cond, keyboard_cond,
        dtype, device, vae_tcr)

    # Generate
    print(f"Generating: {args.action_type}, {args.num_frames} frames, seed={args.seed}")

    if is_causal:
        video = infer_causal(pipeline, conditional_dict, args.num_frames,
                             args.height, args.width, device, dtype, args.seed)
    else:
        video = infer_bidirectional(
            generator, vae, conditional_dict, args.num_frames,
            args.height, args.width, device, dtype, args.seed,
            shift=float(getattr(config, "timestep_shift", 5.0)))

    # Save
    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    video_np = (video[0].permute(0, 2, 3, 1).cpu().float().numpy() * 255
                ).clip(0, 255).astype(np.uint8)
    writer = imageio.get_writer(args.output_path, fps=12, codec='libx264',
                                quality=8, macro_block_size=None)
    for frame in video_np:
        writer.append_data(frame)
    writer.close()
    print(f"Saved: {args.output_path} ({len(video_np)} frames)")


if __name__ == "__main__":
    main()
