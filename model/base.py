from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from pipeline import SelfForcingTrainingPipeline,TeacherForcingTrainingPipeline,BidirectionalTrainingPipeline
from utils.loss import get_denoising_loss
from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper


class BaseModel(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self._initialize_models(args, device)

        self.device = device
        self.args = args
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32
        if hasattr(args, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(args.denoising_step_list, dtype=torch.long, device=self.device)
            if args.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32))).to(self.device)
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]
                
    def _initialize_models(self, args, device):

        self.real_model_name = getattr(args, "real_name", "Wan2.1-T2V-1.3B")
        self.fake_model_name = getattr(args, "fake_name", "Wan2.1-T2V-1.3B")
        self.iscausal = getattr(args, "causal", True)
        self.action_config = getattr(args, "action_config", None)
        model_kwargs = dict(getattr(args, "model_kwargs", {}))
        action_kwargs = None
        if self.action_config is not None:
            # local_attn_size is a CausalWanAttentionBlock-level kwarg, NOT
            # an ActionModule kwarg.  Popping avoids the "got multiple values
            # for keyword argument 'local_attn_size'" TypeError.
            action_kwargs = dict(self.action_config)
            action_kwargs.pop("local_attn_size", None)
            model_kwargs["action_config"] = action_kwargs

        # ── Sliding-window causal attention plumbing (SF/MG2 long-video) ──
        # local_attn_size (in units of blocks) lives at the DiT top level,
        # controls both the block_mask at train time and the KV-cache size
        # at inference.  sink_size=0 means the first frame may be evicted
        # from rolling KV, which is what Self-Forcing paper recommends to
        # avoid the image-latent distribution mismatch when the first frame
        # falls out of the window.  MG2 distilled uses local_attn_size=6,
        # sink_size=0.
        self.local_attn_size = getattr(args, "local_attn_size", -1)
        self.sink_size = getattr(args, "sink_size", 0)
        wrapper_extra = dict(
            local_attn_size=self.local_attn_size,
            sink_size=self.sink_size,
        )

        self.generator = WanDiffusionWrapper(
            **model_kwargs, is_causal=self.iscausal, **wrapper_extra)
        self.generator.model.requires_grad_(True)

        # For MG2 I2V DMD, real_score and fake_score must be action/I2V aware
        # too — they predict the same conditional distribution the generator
        # is being trained to match.  When action_config is set we clone the
        # generator's model_kwargs (model_name + action_config) so all three
        # modules share the same MG2 base topology.  Fallback to the legacy
        # T2V Wan-1.3B path when action_config is not set.
        if action_kwargs is not None:
            # Deep-copy to keep the three independent; they will load the
            # same MG2 base weights but own distinct parameter buffers.
            #
            # real_score = TEACHER = MG2 Foundation model, BID full attention.
            # MG2 base weights were trained with bidirectional full attention
            # (Matrix-Game-2.0/base_model/base_config.json: _class_name =
            # "WanModel").  Per CF orig (Causal-Forcing/model/base.py L34)
            # and DMD2 paper, the DMD teacher must match the pretraining
            # distribution — so is_causal=False, no sliding window.
            #
            # fake_score = CRITIC.  Per CF orig (L37) it is ALSO bid
            # (is_causal=False), starting from the same MG2 base weights as
            # real_score.  The (real - fake) DMD signal is well-defined
            # because both score networks live in the same hypothesis class
            # at init; the critic then drifts toward the student's
            # distribution through training.
            #
            # An earlier mistake was to give fake_score the causal +
            # sliding-window arch to "match" the student.  That breaks
            # alignment with CF orig and forces the critic to track an
            # attention regime it was not pretrained for, weakening the
            # DMD KL signal.
            self.generator = self.generator  # already built above with sliding
            self.real_score = WanDiffusionWrapper(
                **dict(model_kwargs), is_causal=False)
            self.fake_score = WanDiffusionWrapper(
                **dict(model_kwargs), is_causal=False)
        else:
            self.real_score = WanDiffusionWrapper(model_name=self.real_model_name, is_causal=False)
            self.fake_score = WanDiffusionWrapper(model_name=self.fake_model_name, is_causal=False)
        self.real_score.model.requires_grad_(False)
        self.fake_score.model.requires_grad_(True)

        # Optional: load a fine-tuned teacher checkpoint (e.g. Stage 0 SFT)
        # on top of the base model weights for real_score and fake_score.
        teacher_ckpt = getattr(args, "teacher_ckpt", None)
        if teacher_ckpt:
            print(f"[DMD] Loading teacher_ckpt: {teacher_ckpt}")
            raw = torch.load(teacher_ckpt, map_location="cpu")
            if isinstance(raw, dict):
                gen_sd = raw.get("generator", raw.get("generator_ema", raw))
            else:
                gen_sd = raw
            # Strip FSDP/checkpoint wrapper keys
            fixed = {}
            for k, v in gen_sd.items():
                k = k.replace("._fsdp_wrapped_module.", ".")
                k = k.replace("._checkpoint_wrapped_module.", ".")
                fixed[k] = v
            missing_r, _ = self.real_score.load_state_dict(fixed, strict=False)
            missing_f, _ = self.fake_score.load_state_dict(fixed, strict=False)
            print(f"  real_score: loaded, missing={len(missing_r)}")
            print(f"  fake_score: loaded, missing={len(missing_f)}")

        self.text_encoder = WanTextEncoder()
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper()
        self.vae.requires_grad_(False)

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _get_timestep(
            self,
            min_timestep: int,
            max_timestep: int,
            batch_size: int,
            num_frame: int,
            num_frame_per_block: int,
            uniform_timestep: bool = False
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, 1],
                device=self.device,
                dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep,
                max_timestep,
                [batch_size, num_frame],
                device=self.device,
                dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent = None,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v:
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        #
        # For i2v with initial_latent provided, we generate N-1 "new" frames
        # and the pipeline prepends initial_latent to get N total.  So the
        # SAMPLED num_generated_frames refers to the NEW frames, and the
        # constraint is on those:  num_generated_frames % num_frame_per_block == 0
        # with the total output being (num_generated_frames + 1) frames.
        #
        # i.e. total video length ∈ {22, 25, ..., 46} latent frames when
        # num_frame_per_block=3 and num_training_frames=45.  That's the SF
        # convention: generate in multiples of num_frame_per_block, then
        # ref-frame prepend.
        is_i2v = self.args.i2v and initial_latent is not None
        if is_i2v:
            # New-frame budget excluding the prepended ref frame.  Set the
            # config's `num_training_frames` to (1 + 3k) so that
            # (num_training_frames - 1) is divisible by num_frame_per_block.
            #   e.g. num_training_frames=46 -> new ∈ [21, 45], total ∈ [22, 46]
            min_num_frames = 21 if not self.args.independent_first_frame else 20
            max_num_frames = self.num_training_frames - 1
        else:
            min_num_frames = 20 if self.args.independent_first_frame else 21
            max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0, \
            f"max_num_frames={max_num_frames} not divisible by num_frame_per_block={self.num_frame_per_block}"
        assert min_num_frames % self.num_frame_per_block == 0, \
            f"min_num_frames={min_num_frames} not divisible by num_frame_per_block={self.num_frame_per_block}"
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        
        clean_image_or_video = None
        if clean_latent:
            clean_image_or_video = clean_latent.to(self.dtype)
            clean_image_or_video = clean_image_or_video.to(self.device)
            assert clean_image_or_video.shape == tuple(noise_shape), f"{clean_image_or_video.shape} != {tuple(noise_shape)}"

        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            clean_image_or_video=clean_image_or_video,
            **conditional_dict,
        )
        # Slice last 21 frames — I2V bypass: our initial_latent already
        # provides the first frame, so we skip the CF VAE decode+reencode path
        # (which also blows memory on H20 for 22+ frames).
        if self.args.i2v:
            pred_image_or_video_last_21 = pred_image_or_video
        elif pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to

    def _consistency_backward_simulation(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,clean_image_or_video=clean_image_or_video, **conditional_dict
        )
    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        # Compute frame_seq_length from model config so the kv_cache sizes
        # match the actual latent resolution instead of the default 1560
        # (which corresponds to the 480p Wan-T2V 1.3B setting).
        frame_seq_length = self._compute_frame_seq_length()
        vae_tcr = 4
        ac = getattr(self.args, "action_config", None)
        if ac is not None:
            try:
                vae_tcr = int(ac.get("vae_time_compression_ratio", 4))
            except AttributeError:
                vae_tcr = int(getattr(ac, "vae_time_compression_ratio", 4))
        self.inference_pipeline = SelfForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            frame_seq_length=frame_seq_length,
            vae_time_compression_ratio=vae_tcr,
            use_action=(ac is not None),
        )

    def _compute_frame_seq_length(self):
        """Derive patch-tokens-per-frame from config.image_or_video_shape
        and the generator's patch_size.  Falls back to the legacy 1560
        default if we cannot read the shape.
        """
        try:
            _, _, _, H_lat, W_lat = list(self.args.image_or_video_shape)
            p_h = self.generator.model.patch_size[1]
            p_w = self.generator.model.patch_size[2]
            return (H_lat // p_h) * (W_lat // p_w)
        except Exception as e:
            print(f"[warn] could not derive frame_seq_length ({e}); "
                  f"falling back to 1560")
            return 1560
        



class TeacherForcingModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None: # never met
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v: # never met
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None: # never met
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames

        # ========== TF clean video. todo: add noise for RTF. ==========
        clean_image_or_video = clean_latent.to(self.dtype)
        clean_image_or_video = clean_image_or_video.to(self.device)
        assert clean_image_or_video.shape == tuple(noise_shape), f"{clean_image_or_video.shape} != {tuple(noise_shape)}"

        # ==============================================================
        
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation_tf(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            clean_image_or_video=clean_image_or_video,
            **conditional_dict,
        )
        # Slice last 21 frames — I2V bypass: our initial_latent already
        # provides the first frame, so we skip the CF VAE decode+reencode path
        # (which also blows memory on H20 for 22+ frames).
        if self.args.i2v:
            pred_image_or_video_last_21 = pred_image_or_video
        elif pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to
        
    def _consistency_backward_simulation_tf(
        self,
        noise: torch.Tensor,
        clean_image_or_video: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - clean_image_or_video: clean GT video latent with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline_tf()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            clean_image_or_video=clean_image_or_video,
            **conditional_dict
        )
        
    def _initialize_inference_pipeline_tf(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = TeacherForcingTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            spatial_self=True
        )

class BidirectionalModel(BaseModel):
    def __init__(self, args, device):
        super().__init__(args, device)
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

    def _run_generator(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        clean_latent = None,
        initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.args, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None: # never met
            conditional_dict["initial_latent"] = initial_latent
        if self.args.i2v: # never met
            noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            noise_shape = image_or_video_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 20 if self.args.independent_first_frame else 21
        max_num_frames = self.num_training_frames - 1 if self.args.independent_first_frame else self.num_training_frames
        assert max_num_frames % self.num_frame_per_block == 0
        assert min_num_frames % self.num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.num_frame_per_block
        min_num_blocks = min_num_frames // self.num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.num_frame_per_block
        if self.args.independent_first_frame and initial_latent is None: # never met
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        noise_shape[1] = num_generated_frames
        
        pred_image_or_video, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation_bidirectional(
            noise=torch.randn(noise_shape,
                              device=self.device, dtype=self.dtype),
            **conditional_dict,
        )
        # Slice last 21 frames — I2V bypass: our initial_latent already
        # provides the first frame, so we skip the CF VAE decode+reencode path
        # (which also blows memory on H20 for 22+ frames).
        if self.args.i2v:
            pred_image_or_video_last_21 = pred_image_or_video
        elif pred_image_or_video.shape[1] > 21:
            with torch.no_grad():
                latent_to_decode = pred_image_or_video[:, :-20, ...]
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            pred_image_or_video_last_21 = torch.cat([image_latent, pred_image_or_video[:, -20:, ...]], dim=1)
        else:
            pred_image_or_video_last_21 = pred_image_or_video

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(pred_image_or_video_last_21, dtype=torch.bool)
            if self.args.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, :self.num_frame_per_block] = False
        else:
            gradient_mask = None

        pred_image_or_video_last_21 = pred_image_or_video_last_21.to(self.dtype)
        return pred_image_or_video_last_21, gradient_mask, denoised_timestep_from, denoised_timestep_to
        
    def _consistency_backward_simulation_bidirectional(
        self,
        noise: torch.Tensor,
        **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - clean_image_or_video: clean GT video latent with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline_bidirectional()

        return self.inference_pipeline.inference_with_trajectory(
            noise=noise,
            **conditional_dict
        )
        
    def _initialize_inference_pipeline_bidirectional(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = BidirectionalTrainingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            num_frame_per_block=self.num_frame_per_block,
            independent_first_frame=self.args.independent_first_frame,
            same_step_across_blocks=self.args.same_step_across_blocks,
            last_step_only=self.args.last_step_only,
            num_max_frames=self.num_training_frames,
            context_noise=self.args.context_noise,
            spatial_self=True
        )
