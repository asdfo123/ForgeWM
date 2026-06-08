from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


def _slice_cond_for_block(conditional_dict: dict, current_start_frame: int,
                          num_frames_in_block: int,
                          vae_time_compression_ratio: int = 4) -> dict:
    """Return a shallow copy of conditional_dict with cond_concat /
    mouse_condition / keyboard_condition sliced for the current block.

    MG2's CausalWanModel concatenates cond_concat to x along the channel
    dim, so cond_concat must have the SAME F dim as the noisy video
    being fed in (= num_frames_in_block during autoregressive rollout).

    visual_context stays as-is (it's the first-frame CLIP embedding,
    not per-frame).

    mouse/keyboard are sampled at pixel-frame rate; slice up to the
    pixel frame corresponding to the end of the current block.
    """
    if "cond_concat" not in conditional_dict:
        return conditional_dict
    current = dict(conditional_dict)
    current["cond_concat"] = conditional_dict["cond_concat"][
        :, current_start_frame:current_start_frame + num_frames_in_block]
    # Slice mouse/keyboard up to the corresponding pixel frame boundary.
    raw_end = 1 + vae_time_compression_ratio * (
        current_start_frame + num_frames_in_block - 1)
    for k in ("mouse_condition", "keyboard_condition", "mouse_cond", "keyboard_cond"):
        if k in conditional_dict and conditional_dict[k] is not None:
            current[k] = conditional_dict[k][:, :raw_end]
    return current


class SelfForcingTrainingPipeline:
    def __init__(self,
                 denoising_step_list: List[int],
                 scheduler: SchedulerInterface,
                 generator: WanDiffusionWrapper,
                 num_frame_per_block=3,
                 independent_first_frame: bool = False,
                 same_step_across_blocks: bool = False,
                 last_step_only: bool = False,
                 num_max_frames: int = 21,
                 context_noise: int = 0,
                 frame_seq_length: Optional[int] = None,
                 vae_time_compression_ratio: int = 4,
                 use_action: bool = False,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters.
        # num_transformer_blocks: MG2 base has num_layers=30.
        self.num_transformer_blocks = 30
        # frame_seq_length is (latent_H / patch_H) * (latent_W / patch_W)
        # with patch_size=(1,2,2):
        #   - 480p (60x104 latent) -> 30 * 52 = 1560
        #   - 360p (44x80  latent) -> 22 * 40 =  880  ← MG2 native
        # Prefer to pass this explicitly from config; otherwise fall back
        # to the legacy 1560 default for back-compat with T2V Wan-1.3B configs.
        self.frame_seq_length = frame_seq_length if frame_seq_length is not None else 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False
        self.vae_time_compression_ratio = vae_time_compression_ratio

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.kv_cache_mouse = None
        self.kv_cache_keyboard = None
        self.use_action = use_action
        # When the generator is MG2 (action-conditioned), the cross-attention
        # context is CLIP visual_context of length 257, not legacy T5 text
        # embeddings of length 512.
        self._use_mg2_ctx = use_action
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        # IMPORTANT: train-time KV cache is sized to the FULL rollout
        # (num_max_frames frames), regardless of local_attn_size.  This
        # mirrors SF original (Self-Forcing/pipeline/self_forcing_training.py:39).
        #
        # Why: the sliding window during training is enforced by the
        # _prepare_blockwise_causal_attn_mask FUNCTION (causal_model.py:535-540),
        # NOT by physically truncating the KV cache.  Training cache stays
        # full-size so the eviction-shift code path in causal_model.py L228
        # never triggers — that path mutates kv_cache["k"]/["v"] in-place,
        # which breaks gradient_checkpointing's recompute (the recomputed
        # forward sees a different cache state than the original forward,
        # producing a CheckpointError "saved metadata != recomputed metadata").
        #
        # Inference (pipeline/causal_inference.py:339) sizes the cache to
        # `local_attn_size * frame_seq_length` and DOES trigger eviction —
        # because at inference the cache is single-pass and has no
        # backward to recompute.  Training and inference behaviors differ;
        # the mask function is what aligns them semantically.
        self.num_max_frames = num_max_frames
        self.kv_cache_size = num_max_frames * self.frame_seq_length

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(num_blocks,),
                device=device
            )
            # In our training, self.last_step_only is False
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
            self,
            noise: torch.Tensor,
            clean_image_or_video: torch.Tensor = None, # same shape as noise
            initial_latent: Optional[torch.Tensor] = None,
            return_sim_step: bool = False,
            **conditional_dict
    ) -> torch.Tensor:
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        self._initialize_crossattn_cache(
            batch_size=batch_size, dtype=noise.dtype, device=noise.device
        )
        if self.use_action:
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size, dtype=noise.dtype, device=noise.device
            )


        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None: # Never met
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            # Cast initial_latent to the same dtype as noise so the ref-frame
            # DiT forward doesn't hit a float/bfloat16 mismatch in
            # patch_embedding (FSDP keeps model params bf16 under
            # mixed_precision).
            initial_latent = initial_latent.to(dtype=noise.dtype)
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            output[:, :1] = initial_latent
            # The ref-frame init is a 1-frame forward.  If the model's
            # num_frame_per_block > 1, temporarily flip it to 1 so that the
            # ActionModule's internal assertions (which assume one forward
            # call processes exactly num_frame_per_block latent frames) hold.
            # NOTE: the generator is FSDP-wrapped, so writing to
            # `self.generator.model.num_frame_per_block` actually mutates
            # the outer FSDP shell.  To reach the inner CausalWanModel
            # (which `kwargs["num_frame_per_block"] = self.num_frame_per_block`
            # reads at forward time), we walk through `_fsdp_wrapped_module`
            # if present.
            def _inner_model(m):
                # Unwrap FSDP / checkpoint wrappers to get the real CausalWanModel
                while hasattr(m, "_fsdp_wrapped_module"):
                    m = m._fsdp_wrapped_module
                if hasattr(m, "_checkpoint_wrapped_module"):
                    m = m._checkpoint_wrapped_module
                return m
            _inner = _inner_model(self.generator.model)
            _orig_nfpb = getattr(_inner, "num_frame_per_block", 1)
            if _orig_nfpb != 1:
                _inner.num_frame_per_block = 1
            try:
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=initial_latent,
                        conditional_dict=_slice_cond_for_block(
                            conditional_dict, current_start_frame, 1,
                            self.vae_time_compression_ratio),
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache1,
                        crossattn_cache=self.crossattn_cache,
                        kv_cache_mouse=self.kv_cache_mouse,
                        kv_cache_keyboard=self.kv_cache_keyboard,
                        current_start=current_start_frame * self.frame_seq_length
                    )
            finally:
                if _orig_nfpb != 1:
                    _inner.num_frame_per_block = _orig_nfpb
            current_start_frame += 1

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # In out training, self.independent_first_frame is False
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21

        # for block_index in range(num_blocks):
        for block_index, current_num_frames in enumerate(all_num_frames):
            # Slice the per-frame conditioning (cond_concat / mouse / kbd)
            # to cover only the current block's latent frames.  visual_context
            # is first-frame CLIP, stays as-is.
            block_cond = _slice_cond_for_block(
                conditional_dict, current_start_frame, current_num_frames,
                self.vae_time_compression_ratio,
            )

            if True:
                noisy_input = noise[
                    :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

                # Step 3.1: Spatial denoising loop
                for index, current_timestep in enumerate(self.denoising_step_list):
                    if self.same_step_across_blocks:
                        exit_flag = (index == exit_flags[0])
                    else:
                        exit_flag = (index == exit_flags[block_index])
                    timestep = torch.ones(
                        [batch_size, current_num_frames],
                        device=noise.device,
                        dtype=torch.int64) * current_timestep

                    if not exit_flag:
                        with torch.no_grad():
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=block_cond,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                kv_cache_mouse=self.kv_cache_mouse,
                                kv_cache_keyboard=self.kv_cache_keyboard,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                            next_timestep = self.denoising_step_list[index + 1]
                            noisy_input = self.scheduler.add_noise(
                                denoised_pred.flatten(0, 1),
                                torch.randn_like(denoised_pred.flatten(0, 1)),
                                next_timestep * torch.ones(
                                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
                            ).unflatten(0, denoised_pred.shape[:2])
                    else:
                        if current_start_frame < start_gradient_frame_index:
                            with torch.no_grad():
                                _, denoised_pred = self.generator(
                                    noisy_image_or_video=noisy_input,
                                    conditional_dict=block_cond,
                                    timestep=timestep,
                                    kv_cache=self.kv_cache1,
                                    crossattn_cache=self.crossattn_cache,
                                    kv_cache_mouse=self.kv_cache_mouse,
                                    kv_cache_keyboard=self.kv_cache_keyboard,
                                    current_start=current_start_frame * self.frame_seq_length
                                )
                        else: # enable grad
                            _, denoised_pred = self.generator(
                                noisy_image_or_video=noisy_input,
                                conditional_dict=block_cond,
                                timestep=timestep,
                                kv_cache=self.kv_cache1,
                                crossattn_cache=self.crossattn_cache,
                                kv_cache_mouse=self.kv_cache_mouse,
                                kv_cache_keyboard=self.kv_cache_keyboard,
                                current_start=current_start_frame * self.frame_seq_length
                            )
                        break

            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred

            # Step 3.3: rerun with timestep zero to update the cache
            #
            # ─── Cache-refresh fix ───────────────────────────────────────────
            # Previously we skipped this rerun for grad-enabled blocks (i.e.
            # all chunks 1..6 when num_training_frames=22 since
            # start_gradient_frame_index = num_output_frames - 21 = 1).
            # That created a train/inference distribution shift on the KV
            # cache: at inference each chunk goes through the full 4-step
            # denoising chain so its cache is near-clean (≈ t=0); at training
            # we used to leave cache at the exit_flags step (often noisy).
            #
            # CF orig / minWM / SF orig all run this refresh unconditionally.
            # Restoring the original behavior closes the gap and empirically
            # fixes HUD shrinkage / OOD drift on long-video rollouts.
            #
            # The previous "skip" branch is kept commented below for
            # reference; do not re-enable without re-analyzing.
            #
            # was_grad_block = (current_start_frame >= start_gradient_frame_index)
            # if was_grad_block:
            #     current_start_frame += current_num_frames
            #     continue
            # ─────────────────────────────────────────────────────────────────


            context_timestep = torch.ones_like(timestep) * self.context_noise
            # add context noise
            denoised_pred = self.scheduler.add_noise(
                denoised_pred.flatten(0, 1),
                torch.randn_like(denoised_pred.flatten(0, 1)),
                context_timestep * torch.ones(
                    [batch_size * current_num_frames], device=noise.device, dtype=torch.long)
            ).unflatten(0, denoised_pred.shape[:2])
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=denoised_pred,
                    conditional_dict=block_cond,
                    timestep=context_timestep,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    kv_cache_mouse=self.kv_cache_mouse,
                    kv_cache_keyboard=self.kv_cache_keyboard,
                    current_start=current_start_frame * self.frame_seq_length
                )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks: # Useless, never met
            denoised_timestep_from, denoised_timestep_to = None, None
        # T -> \tau_1 -> \tau_2 ->...-> \tau —— enable grad ——> 0
        # denoised_timestep_from = \tau
        # denoised_timestep_to = next timestep smaller than \tau
        # These are just engineering tricks
        # to align DMD timestep sampling with the actual denoising range used by the generator
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            # corner case when \tau is the smallest non-zero timestep
            denoised_timestep_to = 0
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()
        else:
            denoised_timestep_to = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0).item()
            denoised_timestep_from = 1000 - torch.argmin(
                (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0).item()

        if return_sim_step: # False
            return output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1

        return output, denoised_timestep_from, denoised_timestep_to

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        MG2 base: num_heads=12, head_dim=128.
        """
        kv_cache1 = []

        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, self.kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device):
        """Initialize per-block KV caches for the MG2 ActionModule's
        mouse / keyboard attention.  Only needed when the generator is
        an action-conditioned MG2 model.

        Shapes follow pipeline/causal_diffusion_inference.py:
          - keyboard: [B, cache_size, 16, 64]          (heads_num=16, head_dim=64)
          - mouse:    [B * frame_seq, cache_size, 16, 64]  (per spatial token)
        """
        kv_cache_mouse = []
        kv_cache_keyboard = []
        # Mouse/keyboard kv_cache is sized per-LATENT-FRAME (not per-token).
        #
        # IMPORTANT: at TRAIN time we always size to num_max_frames (the full
        # rollout), regardless of local_attn_size.  Same reason as the main
        # KV cache (see __init__ comment): if we sized to local_attn_size
        # here, the eviction code path in ActionModule would mutate
        # k/v in-place between forward and gradient_checkpointing's
        # recompute, producing
        #   CheckpointError: Recomputed values have different metadata
        #   saved [880, 6, 16, 64]  vs  recomputed [880, 3, 16, 64]
        # The sliding window is enforced semantically by the attention
        # mask, NOT by physically truncating the cache.  Inference uses
        # local_attn_size sizing (single-pass, no backward) — see
        # pipeline/causal_inference.py for the inference path.
        kv_cache_size = self.num_max_frames
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
            kv_cache_mouse.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            })
        self.kv_cache_mouse = kv_cache_mouse
        self.kv_cache_keyboard = kv_cache_keyboard

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.

        For MG2 (I2V, CLIP visual_context) the context length is 257
        (16x16 ViT tokens + cls token).  For legacy T2V (T5 text) it
        was 512.  We pick based on whether action_config is present —
        a proxy for "this is MG2".
        """
        ctx_len = 257 if getattr(self, "_use_mg2_ctx", True) else 512
        crossattn_cache = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, ctx_len, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, ctx_len, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache
