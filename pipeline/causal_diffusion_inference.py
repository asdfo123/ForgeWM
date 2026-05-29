from tqdm import tqdm
from typing import List, Optional
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


def cond_current(conditional_dict, current_start_frame, num_frame_per_block, vae_time_compression_ratio=4):
    raw_end = 1 + vae_time_compression_ratio * (current_start_frame + num_frame_per_block - 1)
    current = {
        "visual_context": conditional_dict["visual_context"],
        "cond_concat": conditional_dict["cond_concat"][:, current_start_frame:current_start_frame + num_frame_per_block],
    }
    mouse_condition = conditional_dict.get("mouse_condition", conditional_dict.get("mouse_cond"))
    keyboard_condition = conditional_dict.get("keyboard_condition", conditional_dict.get("keyboard_cond"))
    if mouse_condition is not None:
        current["mouse_condition"] = mouse_condition[:, :raw_end]
    if keyboard_condition is not None:
        current["keyboard_condition"] = keyboard_condition[:, :raw_end]
    return current


class CausalDiffusionInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            need_vae = True
    ):
        super().__init__()
        # Step 1: Initialize all models
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        action_config = getattr(args, "action_config", None)
        if action_config is not None:
            action_config = dict(action_config)
            action_config.pop("local_attn_size", None)
            model_kwargs["action_config"] = action_config
        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=True) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        if need_vae:
            self.vae = WanVAEWrapper() if vae is None else vae

        # Step 2: Initialize scheduler
        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = args.timestep_shift

        self.num_transformer_blocks = 30
        self.frame_seq_length = None

        self.kv_cache_pos = None
        self.kv_cache_neg = None
        self.kv_cache_mouse_pos = None
        self.kv_cache_keyboard_pos = None
        self.crossattn_cache_pos = None
        self.crossattn_cache_neg = None
        self.args = args
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.independent_first_frame = args.independent_first_frame
        self.local_attn_size = self.generator.model.local_attn_size

        print(f"KV inference with {self.num_frame_per_block} frames per block")

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

    def inference(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        start_frame_index: Optional[int] = 0,
        return_video=True
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and Stage1 conditional inputs.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            conditional_dict (dict): MG2-style conditioning with visual_context, cond_concat,
                and optional mouse/keyboard action sequences.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
            start_frame_index (int): In long video generation, where does the current window start?
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_frames, num_channels, height, width). It is normalized to be in the range [0, 1].
        """
        batch_size, num_frames, num_channels, height, width = noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        elif self.independent_first_frame and initial_latent is None:
            # Using a [1, 4, 4, 4, 4, 4] model to generate a video without image conditioning
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        self.frame_seq_length = (height // self.generator.model.patch_size[1]) * (width // self.generator.model.patch_size[2])
        vae_time_compression_ratio = getattr(self.args.action_config, 'vae_time_compression_ratio', 4) if getattr(self.args, 'action_config', None) else 4

        output = torch.zeros(
            [batch_size, num_output_frames, num_channels, height, width],
            device=noise.device,
            dtype=noise.dtype
        )

        # Step 1: Initialize KV cache to all zeros
        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_output_frames=num_output_frames,
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device,
                num_output_frames=num_output_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)

        # Step 2: Cache context feature
        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=cond_current(conditional_dict, current_start_frame, 1, vae_time_compression_ratio),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for block_index in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                output[:, cache_start_frame:cache_start_frame + self.num_frame_per_block] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, vae_time_compression_ratio),
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        for current_num_frames in all_num_frames:
            noisy_input = noise[
                :, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            latents = noisy_input

            # Step 3.1: Spatial denoising loop
            sample_scheduler = self._initialize_sample_scheduler(noise)
            for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
                latent_model_input = latents
                timestep = t * torch.ones(
                    [batch_size, current_num_frames], device=noise.device, dtype=torch.float32
                )

                flow_pred, _ = self.generator(
                    noisy_image_or_video=latent_model_input,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, current_num_frames, vae_time_compression_ratio),
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length
                )

                temp_x0 = sample_scheduler.step(
                    flow_pred,
                    t,
                    latents,
                    return_dict=False)[0]
                latents = temp_x0

            # Step 3.2: record the model's output
            output[:, cache_start_frame:cache_start_frame + current_num_frames] = latents

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=cond_current(conditional_dict, current_start_frame, current_num_frames, vae_time_compression_ratio),
                timestep=timestep * 0,
                kv_cache=self.kv_cache_pos,
                kv_cache_mouse=self.kv_cache_mouse_pos,
                kv_cache_keyboard=self.kv_cache_keyboard_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length
            )

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        # Step 4: Decode the output
        if return_video:
            video = self.vae.decode_to_pixel(output)
            video = (video * 0.5 + 0.5).clamp(0, 1)

            if return_latents:
                return video, output
            else:
                return video
        else:
            return output

    
    def inference_for_cd(
        self,
        noise: torch.Tensor,
        conditional_dict: dict,
        record_step_indices: List[int],
        initial_latent: Optional[torch.Tensor] = None,
        start_frame_index: int = 0
    ) -> torch.Tensor:
        """Run causal denoising and record selected latent states for consistency distillation."""
        self.sampling_steps = 48
        batch_size, num_frames, _, height, width = noise.shape

        if (not self.independent_first_frame) or (self.independent_first_frame and initial_latent is not None):
            assert num_frames % self.num_frame_per_block == 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        self.frame_seq_length = (height // self.generator.model.patch_size[1]) * (width // self.generator.model.patch_size[2])
        vae_time_compression_ratio = getattr(self.args.action_config, 'vae_time_compression_ratio', 4) if getattr(self.args, 'action_config', None) else 4

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device,
                                      num_output_frames=num_frames + num_input_frames)
            self._initialize_kv_cache_mouse_and_keyboard(batch_size=batch_size, dtype=noise.dtype, device=noise.device,
                                                          num_output_frames=num_frames + num_input_frames)
            self._initialize_crossattn_cache(batch_size=batch_size, dtype=noise.dtype, device=noise.device)
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_mouse_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard_pos[block_index]["global_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)
                self.kv_cache_keyboard_pos[block_index]["local_end_index"] = torch.tensor([0], dtype=torch.long, device=noise.device)

        sample_scheduler_probe = self._initialize_sample_scheduler(noise)
        total_steps = len(sample_scheduler_probe.timesteps)
        record_step_indices = sorted(set(int(i) for i in record_step_indices))
        if not record_step_indices:
            raise ValueError("record_step_indices must be non-empty")
        if record_step_indices[0] < 0 or record_step_indices[-1] >= total_steps:
            raise ValueError(f"record_step_indices out of range: valid=[0,{total_steps - 1}], got={record_step_indices}")
        record_set = set(record_step_indices)

        current_start_frame = start_frame_index
        cache_start_frame = 0
        if initial_latent is not None:
            timestep0 = torch.zeros([batch_size, 1], device=noise.device, dtype=torch.int64)
            if self.independent_first_frame:
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=cond_current(conditional_dict, current_start_frame, 1, vae_time_compression_ratio),
                    timestep=timestep0,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, cache_start_frame:cache_start_frame + self.num_frame_per_block]
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, self.num_frame_per_block, vae_time_compression_ratio),
                    timestep=timestep0,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                current_start_frame += self.num_frame_per_block
                cache_start_frame += self.num_frame_per_block

        all_num_frames = [self.num_frame_per_block] * num_blocks
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames

        full_chunk_record = []
        for current_num_frames in all_num_frames:
            latents = noise[:, cache_start_frame - num_input_frames:cache_start_frame + current_num_frames - num_input_frames]
            chunk_records = []
            sample_scheduler = self._initialize_sample_scheduler(noise)
            current_cond = cond_current(conditional_dict, current_start_frame, current_num_frames, vae_time_compression_ratio)

            for progress_id, t in enumerate(tqdm(sample_scheduler.timesteps)):
                if progress_id in record_set:
                    print(f"{progress_id}: {t} saved")
                    chunk_records.append(latents.detach().clone())

                timestep = t * torch.ones([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
                flow_pred, _ = self.generator(
                    noisy_image_or_video=latents,
                    conditional_dict=current_cond,
                    timestep=timestep,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                latents = sample_scheduler.step(flow_pred, t, latents, return_dict=False)[0]

            chunk_records.append(latents.detach().clone())
            full_chunk_record.append(torch.stack(chunk_records, dim=1))

            timestep0 = torch.zeros([batch_size, current_num_frames], device=noise.device, dtype=torch.float32)
            self.generator(
                noisy_image_or_video=latents,
                conditional_dict=current_cond,
                timestep=timestep0,
                kv_cache=self.kv_cache_pos,
                kv_cache_mouse=self.kv_cache_mouse_pos,
                kv_cache_keyboard=self.kv_cache_keyboard_pos,
                crossattn_cache=self.crossattn_cache_pos,
                current_start=current_start_frame * self.frame_seq_length,
                cache_start=cache_start_frame * self.frame_seq_length,
            )

            current_start_frame += current_num_frames
            cache_start_frame += current_num_frames

        return torch.cat(full_chunk_record, dim=2)
    
    
    def inference_for_genuine_cd(
        self,
        noisy_input: torch.Tensor,
        conditional_dict: dict,
        initial_latent: Optional[torch.Tensor] = None,
        timestep_idx=0,
        sampling_steps=48,
        chunksize=3
    ) -> torch.Tensor:
        batch_size, num_frames, _, height, width = noisy_input.shape
        assert num_frames == chunksize

        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        self.frame_seq_length = (height // self.generator.model.patch_size[1]) * (width // self.generator.model.patch_size[2])
        vae_time_compression_ratio = getattr(self.args.action_config, 'vae_time_compression_ratio', 4) if getattr(self.args, 'action_config', None) else 4

        if self.kv_cache_pos is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device,
                num_output_frames=num_frames + num_input_frames,
            )
            self._initialize_kv_cache_mouse_and_keyboard(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device,
                num_output_frames=num_frames + num_input_frames,
            )
            self._initialize_crossattn_cache(
                batch_size=batch_size,
                dtype=noisy_input.dtype,
                device=noisy_input.device
            )
        else:
            for block_index in range(self.num_transformer_blocks):
                self.crossattn_cache_pos[block_index]["is_init"] = False
            for block_index in range(len(self.kv_cache_pos)):
                self.kv_cache_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_mouse_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_mouse_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_keyboard_pos[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)
                self.kv_cache_keyboard_pos[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noisy_input.device)

        current_start_frame = 0
        cache_start_frame = 0
        timestep0 = torch.zeros([batch_size, 1], device=noisy_input.device, dtype=torch.int64)

        if initial_latent is not None:
            if self.independent_first_frame:
                assert (num_input_frames - 1) % chunksize == 0
                num_input_blocks = (num_input_frames - 1) // chunksize
                self.generator(
                    noisy_image_or_video=initial_latent[:, :1],
                    conditional_dict=cond_current(conditional_dict, current_start_frame, 1, vae_time_compression_ratio),
                    timestep=timestep0,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                current_start_frame += 1
                cache_start_frame += 1
            else:
                assert num_input_frames % chunksize == 0
                num_input_blocks = num_input_frames // chunksize

            for _ in range(num_input_blocks):
                current_ref_latents = initial_latent[:, cache_start_frame:cache_start_frame + chunksize]
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=cond_current(conditional_dict, current_start_frame, chunksize, vae_time_compression_ratio),
                    timestep=timestep0,
                    kv_cache=self.kv_cache_pos,
                    kv_cache_mouse=self.kv_cache_mouse_pos,
                    kv_cache_keyboard=self.kv_cache_keyboard_pos,
                    crossattn_cache=self.crossattn_cache_pos,
                    current_start=current_start_frame * self.frame_seq_length,
                    cache_start=cache_start_frame * self.frame_seq_length,
                )
                current_start_frame += chunksize
                cache_start_frame += chunksize

        sample_scheduler = self._initialize_sample_scheduler(noisy_input, sampling_steps=sampling_steps)
        t = sample_scheduler.timesteps[timestep_idx]
        timestep = t * torch.ones(
            [batch_size, chunksize], device=noisy_input.device, dtype=torch.float32
        )
        flow_pred, _ = self.generator(
            noisy_image_or_video=noisy_input,
            conditional_dict=cond_current(conditional_dict, current_start_frame, chunksize, vae_time_compression_ratio),
            timestep=timestep,
            kv_cache=self.kv_cache_pos,
            kv_cache_mouse=self.kv_cache_mouse_pos,
            kv_cache_keyboard=self.kv_cache_keyboard_pos,
            crossattn_cache=self.crossattn_cache_pos,
            current_start=current_start_frame * self.frame_seq_length,
            cache_start=cache_start_frame * self.frame_seq_length,
        )

        latents = sample_scheduler.step(
            flow_pred,
            t,
            noisy_input,
            return_dict=False)[0]

        return latents

    

    def _initialize_kv_cache(self, batch_size, dtype, device, num_output_frames=21):
        """
        Initialize a Per-GPU KV cache for the Wan model.

        For local_attn_size != -1, sliding-window cache sized to the window.
        For local_attn_size == -1, cache must fit the entire rollout — Stage-3
        uses 22 frames; the previous hardcoded 15 was undersized and caused
        an out-of-bounds slice when block 5+ writes past frame 15.
        """
        kv_cache_pos = []
        kv_cache_neg = []
        if self.local_attn_size != -1:
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # full causal: cover the whole rollout
            kv_cache_size = num_output_frames * self.frame_seq_length

        for _ in range(self.num_transformer_blocks):
            kv_cache_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_neg.append({
                "k": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 12, 128], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache_pos = kv_cache_pos  # always store the clean cache
        self.kv_cache_neg = kv_cache_neg  # always store the clean cache

    def _initialize_kv_cache_mouse_and_keyboard(self, batch_size, dtype, device, num_output_frames=21):
        kv_cache_mouse_pos = []
        kv_cache_keyboard_pos = []
        # Same sizing logic as main kv_cache: -1 means cover entire rollout.
        kv_cache_size = self.local_attn_size if self.local_attn_size != -1 else num_output_frames
        for _ in range(self.num_transformer_blocks):
            kv_cache_keyboard_pos.append({
                "k": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
            kv_cache_mouse_pos.append({
                "k": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "v": torch.zeros([batch_size * self.frame_seq_length, kv_cache_size, 16, 64], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache_mouse_pos = kv_cache_mouse_pos
        self.kv_cache_keyboard_pos = kv_cache_keyboard_pos

    def _initialize_crossattn_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache_pos = []
        crossattn_cache_neg = []
        for _ in range(self.num_transformer_blocks):
            crossattn_cache_pos.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })
            crossattn_cache_neg.append({
                "k": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, 257, 12, 128], dtype=dtype, device=device),
                "is_init": False
            })

        self.crossattn_cache_pos = crossattn_cache_pos
        self.crossattn_cache_neg = crossattn_cache_neg

    def _initialize_sample_scheduler(self, noise, sampling_steps=-1):
        if sampling_steps == -1:
            sampling_steps = self.sampling_steps
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")
        return sample_scheduler
