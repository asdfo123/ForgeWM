"""Bidirectional Diffusion (Stage 0) — Bid SFT for MG2 with ActionModule.

This is the equivalent of minWM Phase 1 / Stage 0 (Wan21/model/camera_bidirectional_diffusion.py),
adapted for MG2 (mouse + keyboard ActionModule instead of PRoPE camera).

What it does:
  * Loads MG2 base weights (with ActionModule already trained) and runs
    BIDIRECTIONAL flow-matching SFT on Minecraft 360p latent + mouse/keyboard
    + cond_concat + visual_context.
  * The model class is `WanModel` (NOT `CausalWanModel`) — full attention,
    no chunked causal mask, no KV cache, no teacher forcing.
  * Goal: domain-adapt MG2 base to the user's specific Minecraft data
    distribution while preserving the bidirectional generative prior.
    This is the analogue of running a Wan2.1-T2V → bidirectional/model.pt
    SFT stage in the minWM pipeline.

Why bidirectional and NOT causal:
  * MG2 base was trained bidirectional; the cleanest way to adapt the
    style/content prior is to keep the same training regime.
  * Stage 1 TF AR (existing `model/diffusion.py`) is the place where
    bidirectional → causal flips. Stage 0 should NOT do that flip.
  * Once Stage 0 produces a good Bid ckpt, Stage 1 can start from it
    instead of MG2 base.
"""
from typing import Tuple, Optional
import torch

from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper


class BidirectionalDiffusion(BaseModel):
    """Bidirectional flow-matching SFT model.

    Mirror of `model.diffusion.CausalDiffusion` with three differences:
      1. `is_causal=False` (uses `WanModel`, full attention)
      2. NO `num_frame_per_block` — bidirectional means full attention,
         no block-causal mask
      3. NO `teacher_forcing` path — clean_x is never passed; the model
         denoises pure noisy_latents from scratch (standard FM SFT).
    """

    def __init__(self, args, device):
        super().__init__(args, device)

        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Hyperparameters (same as CausalDiffusion).
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.guidance_scale = args.guidance_scale
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        # Bidirectional has no chunked structure; the timestep sampler still
        # needs `num_frame_per_block` for compatibility with BaseModel._get_timestep
        # but we set it = num_training_frames so the "block" degenerates to
        # the full sequence (i.e. all frames share the same timestep — uniform_t).
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        # Defensive: never enable TF on the bid model, even if the yaml leaks
        # the flag in.  Bid SFT is pure flow matching from noise.
        self.teacher_forcing = False

    def _initialize_models(self, args, device):
        """Build the bid model with action_config + I2V conditioning intact.

        action_config layout:
          - top-level `local_attn_size` is a CausalWanAttentionBlock kwarg, NOT
            an ActionModule kwarg → pop it before forwarding (same dance as
            CausalDiffusion._initialize_models).
          - the rest of action_config goes into ActionModule via model_kwargs.
        """
        self.action_config = getattr(args, "action_config", None)
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        if self.action_config is not None:
            action_config = dict(self.action_config)
            action_config.pop("local_attn_size", None)
            model_kwargs["action_config"] = action_config

        # Bidirectional → is_causal=False → model class is `WanModel`.
        # local_attn_size / sink_size are causal-only kwargs and are dropped
        # by WanDiffusionWrapper when is_causal=False (see utils/wan_wrapper.py
        # _load_local_model L181).
        self.generator = WanDiffusionWrapper(
            **model_kwargs,
            is_causal=False,
        )
        self.generator.model.requires_grad_(True)

        self.text_encoder = None  # MG2 path uses CLIP visual_context, not T5

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(
            device=device,
            dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
        )

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,  # unused; kept for trainer-API parity
        clean_latent: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,  # unused
    ) -> Tuple[torch.Tensor, dict]:
        """Standard bidirectional flow-matching loss.

        Identical to minWM `Wan21/model/camera_bidirectional_diffusion.py::generator_loss`,
        with viewmats/Ks replaced by our action conditioning (mouse/keyboard
        already inside conditional_dict, plumbed via WanDiffusionWrapper).
        """
        del unconditional_dict, initial_latent  # not used in bid SFT

        noise = torch.randn_like(clean_latent)
        batch_size, num_frame = image_or_video_shape[:2]

        # Sample timestep (uniform across all frames — bid has no chunk
        # structure, just one global t per sample).
        index = self._get_timestep(
            0,
            self.scheduler.num_train_timesteps,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True,  # bid: same t across all frames per sample
        )
        timestep = self.scheduler.timesteps[index].to(
            dtype=self.dtype, device=self.device
        )
        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))

        training_target = self.scheduler.training_target(
            clean_latent, noise, timestep
        )

        # Bid forward: NO clean_x, NO kv_cache.  Pure flow prediction
        # from (noisy_latents, conditional_dict, t).
        flow_pred, x0_pred = self.generator(
            noisy_image_or_video=noisy_latents,
            conditional_dict=conditional_dict,
            timestep=timestep,
        )

        # Per-element MSE → flow-matching weight → mean.
        loss = torch.nn.functional.mse_loss(
            flow_pred.float(), training_target.float(), reduction="none"
        ).mean(dim=(2, 3, 4))
        loss = loss * self.scheduler.training_weight(timestep).unflatten(
            0, (batch_size, num_frame)
        )
        loss = loss.mean()

        log_dict = {
            "x0": clean_latent.detach(),
            "x0_pred": x0_pred.detach(),
        }
        return loss, log_dict
