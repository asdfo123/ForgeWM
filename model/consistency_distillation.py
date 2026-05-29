import torch.nn.functional as F
from typing import Tuple
import torch
import random
from model.base import BaseModel
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.scheduler import FlowMatchScheduler
from pipeline import CausalDiffusionInferencePipeline


class NaiveConsistency(BaseModel):
    """Causal Consistency Distillation — MG2-adapted.

    This is the port of the upstream 2026.4.16 `naive_consistency.py` 3x
    speedup.  The core change is in :meth:`generator_loss`: we replace the
    per-chunk / per-frame pipeline rollout (which called
    ``CausalDiffusionInferencePipeline.inference_for_genuine_cd`` up to 21
    times per training step) with a **single full-frame teacher forward**
    that predicts the Euler step ``latent_t_next = latent_t - dt * v_pred``
    in one go.

    The upstream version does this because the teacher runs in
    *teacher-forcing* mode (``clean_x=clean_latent``), which means every
    noisy frame can attend to the full clean history in one shot — there
    is no autoregressive dependency to unroll.  So the per-chunk loop was
    pure waste: 21 identical algorithmic ops per training step, each
    rebuilding KV caches and block masks.

    MG2-specific additions preserved from the previous revision:
      * ``action_config`` plumbing (pop ``local_attn_size`` from the dict
        before forwarding to :class:`ActionModule`, forward the rest as a
        sub-dict into ``WanDiffusionWrapper``).
      * ``strict=False`` state_dict load when ``action_config`` is set
        (the upstream base Wan checkpoint has no action-module weights,
        but MG2 base ckpts do — we tolerate either).
      * I2V ``cond_concat`` + ``visual_context`` flowing through
        ``conditional_dict`` / ``unconditional_dict`` unchanged.
      * VAE is initialised in ``_initialize_models`` so that the trainer
        (trainer/naive_cd.py) can build ``cond_concat`` outside this
        class and pass it in via ``conditional_dict``.
    """

    def __init__(self, args, device):
        super().__init__(args, device)

        # ── Step 1: build generator / generator_ema / teacher ────────────
        # ``action_config`` plumbing: the MG2 base model expects
        # ``action_config`` to be a sub-dict of ``model_kwargs`` (consumed
        # by CausalWanAttentionBlock when it instantiates ActionModule).
        # ``local_attn_size`` lives at the block level (NOT inside
        # ActionModule), so we peel it off before forwarding.
        self.action_config = getattr(args, "action_config", None)
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        if self.action_config is not None:
            action_kwargs = dict(self.action_config)
            action_kwargs.pop("local_attn_size", None)
            model_kwargs["action_config"] = action_kwargs

        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=args.is_causal)
        self.generator.model.requires_grad_(True)

        self.generator_ema = WanDiffusionWrapper(
            **model_kwargs, is_causal=args.is_causal)
        self.generator_ema.model.requires_grad_(False)

        # Teacher is always causal (matches Stage-1 TF training mode).
        self.teacher = WanDiffusionWrapper(**model_kwargs, is_causal=True)
        self.teacher.model.requires_grad_(False)

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block
            self.generator_ema.model.num_frame_per_block = self.num_frame_per_block
            self.teacher.model.num_frame_per_block = self.num_frame_per_block

        # ── Step 2: optionally warm-start from a Stage-1 checkpoint ──────
        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")
            # Accept a variety of ckpt layouts:
            #   {'generator': sd}, {'generator_ema': sd}, {'model': sd}, or sd
            if "generator" in state_dict:
                sd = state_dict["generator"]
            elif "generator_ema" in state_dict:
                sd = state_dict["generator_ema"]
            elif "model" in state_dict:
                sd = state_dict["model"]
            else:
                sd = state_dict
            # Strip FSDP / checkpoint wrappers if present.
            fixed = {}
            for k, v in sd.items():
                if k.startswith("model._fsdp_wrapped_module."):
                    k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                if "._checkpoint_wrapped_module." in k:
                    k = k.replace("._checkpoint_wrapped_module.", ".", 1)
                fixed[k] = v
            sd = fixed

            # Strict load when no action_config (vanilla Wan); non-strict
            # for MG2 + action since the base weights may not carry the
            # action_model sub-tree.
            _strict = not bool(self.action_config)
            self.generator.load_state_dict(sd, strict=_strict)
            self.teacher.load_state_dict(sd, strict=_strict)
            self.generator_ema.load_state_dict(sd, strict=_strict)

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # ── Step 3: hyperparameters ──────────────────────────────────────
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.guidance_scale = args.guidance_scale

        self.discrete_cd_N = getattr(args, "discrete_cd_N", 48)
        self.scheduler = FlowMatchScheduler(
            shift=5.0, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(
            num_inference_steps=self.discrete_cd_N,
            denoising_strength=1.0,
        )
        self.scheduler.sigmas = self.scheduler.sigmas.to(device)

        # ── Step 4: stash a no-VAE CausalDiffusionInferencePipeline ──────
        # Previously this pipeline was used per-chunk in ``generator_loss``.
        # After the 4.16 upstream refactor the pipeline is no longer
        # strictly required (full-frame teacher fwd handles it), but we
        # keep it here for backward compat — e.g. someone calls
        # ``model.pipeline.inference_for_genuine_cd`` from a debug script.
        self.pipeline = CausalDiffusionInferencePipeline(
            args, device=device, need_vae=False)
        self.pipeline.generator = self.teacher
        self.pipeline.text_encoder = self.text_encoder

    def _initialize_models(self, args, device):
        # Called from ``BaseModel.__init__`` before ``self.generator`` etc.
        # are created above — pre-fill teacher/generator/generator_ema with
        # the same WanDiffusionWrapper layout (they are overwritten in
        # :meth:`__init__`, but ``_initialize_models`` is where the text
        # encoder + VAE + scheduler go that ``__init__`` relies on).
        self.action_config = getattr(args, "action_config", None)
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        if self.action_config is not None:
            action_kwargs = dict(self.action_config)
            action_kwargs.pop("local_attn_size", None)
            model_kwargs["action_config"] = action_kwargs

        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=True)
        self.generator.model.requires_grad_(True)

        self.teacher = WanDiffusionWrapper(
            **model_kwargs, is_causal=True)
        self.teacher.model.requires_grad_(False)

        self.generator_ema = WanDiffusionWrapper(
            **model_kwargs, is_causal=args.is_causal)
        self.generator_ema.model.requires_grad_(False)

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        # VAE is required by the I2V cond_concat construction inside the
        # CD trainer (trainer/naive_cd.py Step 2.6).  Without this the
        # ``self.model.vae`` reference would fail and decode_to_pixel /
        # encode_to_latent / encode_visual_context_from_pixels would error.
        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)
        self.vae = self.vae.to(
            device=device,
            dtype=torch.bfloat16 if args.mixed_precision else torch.float32,
        )

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    # ────────────────────────────────────────────────────────────────────
    # Core training step — ported from upstream 2026.4.16 (3x speedup)
    # ────────────────────────────────────────────────────────────────────
    def generator_loss(
            self,
            conditional_dict,
            unconditional_dict,
            clean_latent,
            ema_model,
    ) -> Tuple[torch.Tensor, dict]:
        """Causal Consistency Distillation loss (full-frame, MG2-adapted).

        Algorithm:
          1. Sample a random discrete timestep index ``i`` in [0, N-1).
          2. ``t``      = scheduler.timesteps[i]
             ``t_next`` = scheduler.timesteps[i+1]   (i.e. one ODE step closer to clean)
          3. Noise ``x_t`` = add_noise(clean, t).
          4. **Teacher Euler step** (no grad):
                v_cond   = teacher(x_t, cond,    t, clean_x=clean)
                v_uncond = teacher(x_t, uncond,  t, clean_x=clean)
                v_pred   = v_uncond + guidance_scale * (v_cond - v_uncond)
                dt       = (t - t_next) / 1000
                x_t_next = x_t - dt * v_pred      ← the CFG-guided Euler step
          5. Generator forward at both t and t_next:
                cm_t      = generator    (x_t,     cond, t,      clean_x=clean)
                cm_t_next = generator_ema(x_t_next, cond, t_next, clean_x=clean)  (no grad)
          6. Loss = MSE(cm_t, cm_t_next).

        The upstream 3x speedup comes from Step 4: the old implementation
        did a per-chunk ``pipeline.inference_for_genuine_cd`` rollout
        (21 calls framewise) because it mis-modelled the teacher as
        needing autoregressive state.  But since the teacher is in
        teacher-forcing mode (``clean_x=clean_latent``), every noisy
        frame can attend to the full clean history in a single forward
        pass — no rollout needed.
        """
        clean_latent = clean_latent.to(self.device).to(torch.bfloat16)
        B, num_frames = clean_latent.shape[:2]
        timestep_idx = random.randrange(self.discrete_cd_N - 1)

        t = self.scheduler.timesteps[timestep_idx]
        t_next = self.scheduler.timesteps[timestep_idx + 1]
        timestep = t * torch.ones(
            [B, num_frames], device=self.device, dtype=torch.bfloat16)
        timestep_next = t_next * torch.ones(
            [B, num_frames], device=self.device, dtype=torch.bfloat16)

        noise = torch.randn_like(clean_latent)
        latent_t = self.scheduler.add_noise(
            clean_latent, noise=noise,
            timestep=t * torch.ones([1], device=self.device),
        ).to(torch.bfloat16)

        # ── Step 4: full-frame teacher Euler step (no grad) ──────────────
        with torch.no_grad():
            v_cond, _ = self.teacher(
                latent_t, conditional_dict, timestep, clean_x=clean_latent,
            )
            v_uncond, _ = self.teacher(
                latent_t, unconditional_dict, timestep, clean_x=clean_latent,
            )
            v_pred = v_uncond + self.guidance_scale * (v_cond - v_uncond)

            # ``v_pred`` is shape [B, F, C, H, W] (wrapper standardises it).
            # dt broadcast shape is [B, F, 1, 1, 1] so each frame gets its
            # own (t - t_next); in practice t is the same across frames,
            # but this keeps the math explicit.
            dt = (timestep - timestep_next).reshape(B, num_frames, 1, 1, 1)
            dt = dt / 1000.0
            latent_t_next = latent_t - dt.to(v_pred.dtype) * v_pred

        # Share the TF block_mask between generator / ema / teacher if
        # the teacher has built one and the generator hasn't — this
        # prevents redundant 30-layer mask allocation.  Safe because the
        # three models share num_frame_per_block / local_attn_size /
        # num_frames / frame_seqlen (all derived from the same config).
        # NOTE: we also need to share teacher_forcing_block_mask because
        # in CD training we always pass clean_x (TF mode).  Upstream only
        # shares ``block_mask``; we share both to cover the MG2 TF path.
        for attr in ("block_mask", "teacher_forcing_block_mask",
                     "block_mask_mouse", "block_mask_keyboard"):
            src = getattr(self.teacher.model, attr, None)
            if src is None:
                continue
            if getattr(self.generator.model, attr, None) is None:
                setattr(self.generator.model, attr, src)
            if getattr(self.generator_ema.model, attr, None) is None:
                setattr(self.generator_ema.model, attr, src)

        # ── Step 5: generator at t (grad) + EMA at t_next (no grad) ──────
        _, cm_pred_t = self.generator(
            latent_t, conditional_dict, timestep, clean_x=clean_latent,
        )

        with torch.no_grad():
            ema_model.copy_to(self.generator_ema)
            _, cm_pred_t_next = self.generator_ema(
                latent_t_next, conditional_dict, timestep_next,
                clean_x=clean_latent,
            )

        with torch.enable_grad():
            loss = F.mse_loss(cm_pred_t, cm_pred_t_next, reduction="mean")

        log_dict = {
            "unnormalized_loss": F.mse_loss(
                cm_pred_t, cm_pred_t_next, reduction="none",
            ).mean(dim=[1, 2, 3, 4]).detach(),
            # Useful for training-health monitoring; these are scalars.
            "cd_t": float(t.item()),
            "cd_t_next": float(t_next.item()),
        }

        return loss, log_dict
