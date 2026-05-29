import gc
import inspect
import logging

from model import CausalDiffusion
from model.bidirectional_diffusion import BidirectionalDiffusion
from utils.dataset import cycle, LatentLMDBDataset, ActionLatentLMDBDataset, ShardingActionLatentLMDBDataset
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
import math
from utils.distributed import EMA_FSDP, barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from pipeline import (
    CausalDiffusionInferencePipeline,
    CausalInferencePipeline,
)

class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.output_path = config.logdir

        self._print_live_module_info(global_rank)

        # Step 2: Initialize the model and optimizer
        # If config.causal is False, this is Stage 0 Bid SFT — use the
        # bidirectional model class (full attention, no TF, no chunked
        # block mask). Default = causal (Stage 1+).
        if not getattr(config, "causal", True):
            self.model = BidirectionalDiffusion(config, device=self.device)
        else:
            self.model = CausalDiffusion(config, device=self.device)
        generator_wrap_strategy = getattr(config, "generator_fsdp_wrap_strategy", "size")
        generator_wrap_kwargs = dict(
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=generator_wrap_strategy,
        )
        if generator_wrap_strategy == "transformer":
            # For Bid SFT (causal=False) the model class is `WanModel`,
            # whose attention block is `WanAttentionBlock` (NOT
            # CausalWanAttentionBlock). Pick the right wrap target.
            if not getattr(config, "causal", True):
                from wan.modules.model import WanAttentionBlock
                generator_wrap_kwargs["transformer_module"] = {WanAttentionBlock}
            else:
                from wan.modules.causal_model import CausalWanAttentionBlock
                generator_wrap_kwargs["transformer_module"] = {CausalWanAttentionBlock}
        self.model.generator = fsdp_wrap(
            self.model.generator,
            **generator_wrap_kwargs,
        )

        if self.model.text_encoder is not None:
            self.model.text_encoder = fsdp_wrap(
                self.model.text_encoder,
                sharding_strategy=config.sharding_strategy,
                mixed_precision=config.mixed_precision,
                wrap_strategy=config.text_encoder_fsdp_wrap_strategy
            )

        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        if getattr(config, "action_config", None) and getattr(config, "action_data", False):
            if getattr(config, "action_sharded", False):
                dataset = ShardingActionLatentLMDBDataset(
                    config.data_path,
                    max_pair=int(1e8),
                    allowed_prefixes=getattr(config, "action_shard_prefixes", None),
                )
                if dist.get_rank() == 0:
                    print(dataset.summary())
            else:
                dataset = ActionLatentLMDBDataset(config.data_path, max_pair=int(1e8))
        else:
            dataset = LatentLMDBDataset(config.data_path, max_pair=int(1e8))
       
        self.dataset = dataset
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=8)

        if dist.get_rank() == 0:
            print("DATASET SIZE %d" % len(dataset))
        self.dataloader = cycle(dataloader)

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p
        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_FSDP(self.model.generator, decay=ema_weight)

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            if "generator" in state_dict:
                state_dict = state_dict["generator"]
                fixed = {}
                for k, v in state_dict.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            elif "model" in state_dict:
                state_dict = state_dict["model"]
            elif "generator_ema" in state_dict:
                gen_sd = state_dict["generator_ema"]
                fixed = {}
                for k, v in gen_sd.items():
                    if k.startswith("model._fsdp_wrapped_module."):
                        k = k.replace("model._fsdp_wrapped_module.", "model.", 1)
                    fixed[k] = v
                state_dict = fixed
            self.model.generator.load_state_dict(state_dict, strict=True)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm = 10.0
        self.previous_time = None
        self.delta_mean = None
        self.rtf_ema_ratio = getattr(self.config, "rtf_ema_ratio", 0.9) 
        self.eval_interval = getattr(self.config, "eval_interval", 0)      # 0 => disable
        self.eval_frames = getattr(self.config, "eval_num_output_frames", 21)
        self.eval_init = getattr(self.config, "eval_num_init_frames", 3)
        self.rtf_single_gpu_batch = getattr(self.config, "rtf_single_gpu_batch", 1)
        self.given_first_chunk = getattr(self.config, "given_first_chunk", True)
        if self.eval_interval:
            self.pipeline = CausalDiffusionInferencePipeline(config, device=self.device)
            self.pipeline.generator = self.model.generator
            self.pipeline.text_encoder = self.model.text_encoder
            
    def _print_live_module_info(self, global_rank):
        from wan.modules.causal_model import CausalWanAttentionBlock
        from wan.modules.model import WanI2VCrossAttention

        def extract_lines(fn, keywords):
            source_lines, start_line = inspect.getsourcelines(fn)
            selected = []
            for idx, line in enumerate(source_lines, start_line):
                if any(keyword in line for keyword in keywords):
                    selected.append(f"{idx}:{line.rstrip()}")
            return " | ".join(selected)

        block_file = inspect.getsourcefile(CausalWanAttentionBlock)
        i2v_file = inspect.getsourcefile(WanI2VCrossAttention)
        forward_lines = extract_lines(
            CausalWanAttentionBlock.forward,
            ["self.cross_attn(", "crossattn_cache", "self.action_model"],
        )
        print(
            f"CausalWanAttentionBlock.forward={inspect.signature(CausalWanAttentionBlock.forward)} "
            f"file={block_file}"
        )
        print(
            f"WanI2VCrossAttention.forward={inspect.signature(WanI2VCrossAttention.forward)} "
            f"file={i2v_file}"
        )

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

        if self.config.ema_start_step < self.step:
            state_dict = {
                "generator": generator_state_dict,
                "generator_ema": self.generator_ema.full_state_dict(self.model.generator),
            }
        else:
            state_dict = {
                "generator": generator_state_dict,
            }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self, batch):
        self.log_iters = 1

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # ─── Diagnostics: per-step, per-section timings ──────────────
        # We want to see how much time goes into VAE decode/encode vs
        # generator forward/backward so we can decide whether to cache
        # cond_concat, batch up the VAE, etc.  Keep this cheap: a few
        # CUDA events and a single python dict.
        def _make_timer():
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            return e

        def _elapsed_ms(start, end):
            # .elapsed_time is synchronous on both events.
            return float(start.elapsed_time(end))

        timers = {}
        t_step_start = _make_timer()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        if not self.config.load_raw_video:  # precomputed latent
            clean_latent = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)
        else:  # encode raw video to latent
            frames = batch["frames"].to(
                device=self.device, dtype=self.dtype)

            with torch.no_grad():
                clean_latent = self.model.vae.encode_to_latent(
                    frames).to(device=self.device, dtype=self.dtype)
        image_latent = clean_latent[:, 0:1, ]

        # ─── Stage-1 shape sanity print (once, rank 0, first step) ───
        # Guard against a silent 360p/480p resolution mismatch by dumping the
        # exact shapes that hit the trainer.  We print again every 1000 steps
        # just in case a bad shard sneaks in mid-run.
        if self.is_main_process and (self.step == 0 or self.step % 1000 == 0):
            cfg_shape = list(self.config.image_or_video_shape)
            mouse_shape = tuple(batch["mouse_condition"].shape) if "mouse_condition" in batch else None
            kbd_shape = tuple(batch["keyboard_condition"].shape) if "keyboard_condition" in batch else None
            print(
                f"config.image_or_video_shape={cfg_shape} "
                f"clean_latent={tuple(clean_latent.shape)} "
                f"mouse={mouse_shape} keyboard={kbd_shape}"
            )

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        t_after_load = _make_timer()

        # Step 2: Build MG2-style conditional infos.
        #
        # We need two things from the first frame:
        #   (a) visual_context: CLIP encoding, fed to the DiT as the img_emb
        #       cross-attention context.
        #   (b) img_cond: VAE-encoded [ref_frame, zeros×80] sequence, used
        #       by the I2V cond_concat (see Step 2.6 below).
        #
        # Both (a) and (b) need the FIRST PIXEL FRAME of the reference
        # image.  We can decode that frame once and reuse it.
        #
        # Correctness note on single-frame decode: MG2 (Wan2.1) VAE is
        # strictly time-causal, so decode(latent[:, 0:1]) produces the
        # exact same pixel tensor as decode(latent)[:, 0:1]
        # (bitwise verified in scripts/verify_vae_decode_equiv.py across
        # 5 random batches — max |diff| = 0.000000 on bf16).  Decoding
        # only 1 latent frame instead of 21 saves ~3s per step.
        with torch.no_grad():
            first_frame_pixels = self.model.vae.decode_to_pixel(image_latent)  # [B, 1, 3, H_pix, W_pix]
            visual_context = self.model.vae.encode_visual_context_from_pixels(first_frame_pixels)
        conditional_dict = {
            "visual_context": visual_context,
        }
        unconditional_dict = {}

        t_after_clip = _make_timer()

        # Step 2.5: Inject action conditions if available
        action_config = getattr(self.config, "action_config", None)
        if action_config is not None:
            target_keyboard_dim = action_config.get("keyboard_dim_in", 4)
            if "mouse_condition" in batch and "keyboard_condition" in batch:
                # Use real action data from dataset
                conditional_dict["mouse_condition"] = batch["mouse_condition"].to(
                    device=self.device, dtype=self.dtype)
                kbd = batch["keyboard_condition"].to(device=self.device, dtype=self.dtype)
                # Pad keyboard to target dim if needed (e.g. 4->6 for MG2 base model)
                if kbd.shape[-1] < target_keyboard_dim:
                    from utils.dataset import ActionLatentLMDBDataset
                    kbd = ActionLatentLMDBDataset.pad_keyboard(kbd, target_keyboard_dim)
                conditional_dict["keyboard_condition"] = kbd
            else:
                # Generate random action conditions as fallback
                num_frames = image_or_video_shape[1]
                vae_tcr = action_config.get("vae_time_compression_ratio", 4)
                num_raw_frames = (num_frames - 1) * vae_tcr + 1
                mouse_dim = action_config.get("mouse_dim_in", 2)
                keyboard_dim = target_keyboard_dim
                conditional_dict["mouse_condition"] = (
                    torch.rand(batch_size, num_raw_frames, mouse_dim, device=self.device, dtype=self.dtype) - 0.5) * 0.1
                conditional_dict["keyboard_condition"] = (
                    torch.rand(batch_size, num_raw_frames, keyboard_dim, device=self.device, dtype=self.dtype) > 0.5).float()
            unconditional_dict["mouse_condition"] = torch.zeros_like(conditional_dict["mouse_condition"])
            unconditional_dict["keyboard_condition"] = torch.zeros_like(conditional_dict["keyboard_condition"])

        # Step 2.6: Construct I2V cond_concat if model is I2V (in_dim > 16)
        #
        # MG2 base model expects cond_concat to be built the MG2 way:
        #   1. Take reference frame in PIXEL space
        #   2. Append zero-padding pixels: [ref_pixel, zeros x (4*(F-1))]  -> 81 pixel frames
        #   3. Run VAE encode on the whole sequence -> img_cond latent [B, 16, F, h, w]
        #   4. Build 4-channel mask: first latent frame all-ones, rest all-zeros
        #   5. Concat: cond_concat = [mask(4) | img_cond(16)]  -> 20 channels
        #
        # The previous implementation (ref_latent[:,1:]=0) used latent-space zeros,
        # which is OUT-OF-DISTRIBUTION for the base model.  The base model was trained
        # with the VAE response to zero-pixel frames (mean~-0.34, std~1.30 per frame),
        # NOT actual zeros.  See docs in this repo for the full analysis.
        #
        # Because we only have precomputed latent (no raw pixel) at training time, we
        # reconstruct pixel via VAE decode on the fly, take the first frame, then
        # re-encode with zero-padding.  This introduces a small (~6% RMS) VAE round-trip
        # loss on frame 0, but eliminates the much larger frame 1-20 mismatch.
        t_before_vae = _make_timer()
        t_after_decode = None
        t_after_encode = None
        if hasattr(self.model.generator, 'model') and getattr(self.model.generator.model, 'in_dim', 16) > 16:
            B, F, C, H, W = clean_latent.shape
            vae_tcr = int(getattr(self.config, "action_config", {}).get("vae_time_compression_ratio", 4)) if getattr(self.config, "action_config", None) else 4
            num_pixel_frames = (F - 1) * vae_tcr + 1

            with torch.no_grad():
                # (1) Reuse the first pixel frame already decoded in Step 2.
                # This avoids decoding all 21 latent frames; MG2's VAE is
                # strictly time-causal so decode(image_latent) is identical
                # to decode(clean_latent)[:, 0:1].  See scripts/verify_vae_decode_equiv.py.
                pixels = first_frame_pixels  # [B, 1, 3, H_pix, W_pix]
                H_pix, W_pix = pixels.shape[-2], pixels.shape[-1]
                assert pixels.shape[1] == 1, (
                    f"Expected single-frame pixel tensor, got F_pix={pixels.shape[1]}"
                )
                t_after_decode = _make_timer()
                # (2) first-frame + zero-padding in pixel space
                first_frame = pixels[:, 0:1].to(self.dtype)        # [B, 1, 3, H_pix, W_pix]
                pad_pix = torch.zeros(
                    B, num_pixel_frames - 1, 3, H_pix, W_pix,
                    device=self.device, dtype=self.dtype,
                )
                padded_pixels_BTCHW = torch.cat([first_frame, pad_pix], dim=1)
                padded_pixels_BCTHW = padded_pixels_BTCHW.permute(0, 2, 1, 3, 4)
                # (3) VAE encode -> img_cond latent [B, F, 16, H, W]
                img_cond = self.model.vae.encode_to_latent(padded_pixels_BCTHW).to(self.dtype)
                t_after_encode = _make_timer()
            assert img_cond.shape == (B, F, C, H, W), (
                f"img_cond shape {tuple(img_cond.shape)} != expected {(B, F, C, H, W)}"
            )
            # (4) 4-channel mask: first latent frame all-ones, rest zeros
            mask = torch.zeros(B, F, 4, H, W, device=self.device, dtype=self.dtype)
            mask[:, 0:1] = 1
            # (5) concat -> [B, F, 20, H, W]
            cond_concat = torch.cat([mask, img_cond], dim=2)
            conditional_dict["cond_concat"] = cond_concat

            # Lightweight diagnostic: on first few steps and every 500 steps, log
            # cond_concat statistics so we can spot regressions.
            if self.is_main_process and (self.step < 5 or self.step % 500 == 0):
                # shape print (resolution sanity check for 360p migration)
                print(
                    f"pixels={tuple(pixels.shape)} (H_pix={H_pix}, W_pix={W_pix}) "
                    f"cond_concat={tuple(cond_concat.shape)} img_cond={tuple(img_cond.shape)} "
                    f"mouse_cond={tuple(conditional_dict['mouse_condition'].shape)} "
                    f"keyboard_cond={tuple(conditional_dict['keyboard_condition'].shape)}"
                )
                with torch.no_grad():
                    cc = cond_concat.float()
                    img_c = img_cond.float()
                    stats = {
                        "cond_concat/mask_sum_frame0": float(mask[0, 0].sum().item()),
                        "cond_concat/mask_sum_rest": float(mask[0, 1:].sum().item()),
                        "cond_concat/img_cond_frame0_mean": float(img_c[0, 0].mean().item()),
                        "cond_concat/img_cond_frame0_std": float(img_c[0, 0].std().item()),
                        "cond_concat/img_cond_rest_mean": float(img_c[0, 1:].mean().item()),
                        "cond_concat/img_cond_rest_std": float(img_c[0, 1:].std().item()),
                        "cond_concat/overall_mean": float(cc.mean().item()),
                        "cond_concat/overall_std": float(cc.std().item()),
                    }
                    print(f"[step {self.step}] cond_concat stats: {stats}")
                    if not self.disable_wandb:
                        wandb.log(stats, step=self.step)

        t_after_cond = _make_timer()

        # Step 3: Train the generator
        generator_loss, log_dict = self.model.generator_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent
        )
        t_after_forward = _make_timer()

        self.generator_optimizer.zero_grad()
        generator_loss.backward()
        t_after_backward = _make_timer()

        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm)
        self.generator_optimizer.step()

        t_after_opt = _make_timer()
        torch.cuda.synchronize()  # make the Event timings valid

        # Increment the step since we finished gradient update
        self.step += 1

        wandb_loss_dict = {
            "generator_loss": generator_loss.item(),
            "generator_grad_norm": generator_grad_norm.item(),
        }

        # ─── Timing breakdown (always log, it's cheap) ───────────────
        timings = {
            "timing/data_load_ms": _elapsed_ms(t_step_start, t_after_load),
            "timing/clip_ms": _elapsed_ms(t_after_load, t_after_clip),
            "timing/vae_total_ms": _elapsed_ms(t_before_vae, t_after_cond),
            "timing/forward_ms": _elapsed_ms(t_after_cond, t_after_forward),
            "timing/backward_ms": _elapsed_ms(t_after_forward, t_after_backward),
            "timing/optimizer_ms": _elapsed_ms(t_after_backward, t_after_opt),
            "timing/step_total_ms": _elapsed_ms(t_step_start, t_after_opt),
        }
        if t_after_decode is not None and t_after_encode is not None:
            timings["timing/vae_decode_ms"] = _elapsed_ms(t_before_vae, t_after_decode)
            timings["timing/vae_encode_ms"] = _elapsed_ms(t_after_decode, t_after_encode)
        wandb_loss_dict.update(timings)

        # ─── Rich diagnostics every 5 steps for the first 50, then every 50 ───
        # These are the real signals of whether the model is actually learning
        # causal structure (not just fitting the TF single-step loss).
        do_diag = self.is_main_process and (
            self.step <= 10
            or (self.step <= 50 and self.step % 5 == 0)
            or self.step % 50 == 0
        )
        if do_diag:
            with torch.no_grad():
                x0 = log_dict["x0"].float()
                x0_pred = log_dict["x0_pred"].float()
                # residual between prediction and ground truth, broken down
                # by frame index so we can see if later frames are worse
                # (which would hint at the rope-TF fix mattering).
                resid_per_frame = ((x0 - x0_pred) ** 2).mean(dim=(0, 2, 3, 4))  # [F]
                diag = {
                    "diag/loss": float(generator_loss.item()),
                    "diag/grad_norm": float(generator_grad_norm.item()),
                    "diag/x0_mean": float(x0.mean().item()),
                    "diag/x0_std": float(x0.std().item()),
                    "diag/x0_abs_max": float(x0.abs().max().item()),
                    "diag/x0_pred_mean": float(x0_pred.mean().item()),
                    "diag/x0_pred_std": float(x0_pred.std().item()),
                    "diag/x0_pred_abs_max": float(x0_pred.abs().max().item()),
                    "diag/x0_mse": float(((x0 - x0_pred) ** 2).mean().item()),
                    "diag/x0_mse_frame0": float(resid_per_frame[0].item()),
                    "diag/x0_mse_frame_last": float(resid_per_frame[-1].item()),
                    "diag/x0_mse_frame_mid": float(resid_per_frame[len(resid_per_frame) // 2].item()),
                    "diag/std_ratio": float((x0_pred.std() / (x0.std() + 1e-8)).item()),  # >1 → overshoot, <1 → underfit
                }
                wandb_loss_dict.update(diag)
                print(
                    f"[step {self.step}] loss={diag['diag/loss']:.4f} "
                    f"grad_norm={diag['diag/grad_norm']:.3f} "
                    f"x0_mse={diag['diag/x0_mse']:.4f} "
                    f"(f0={diag['diag/x0_mse_frame0']:.3f}, "
                    f"fmid={diag['diag/x0_mse_frame_mid']:.3f}, "
                    f"fL={diag['diag/x0_mse_frame_last']:.3f}) "
                    f"x0_pred(std={diag['diag/x0_pred_std']:.3f}, ratio={diag['diag/std_ratio']:.2f}) "
                    f"| step_time={timings['timing/step_total_ms']/1000:.1f}s "
                    f"(vae={timings['timing/vae_total_ms']/1000:.1f}s "
                    f"fwd={timings['timing/forward_ms']/1000:.1f}s "
                    f"bwd={timings['timing/backward_ms']/1000:.1f}s)"
                )

        # Step 4: Logging
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)

        gc_interval = getattr(self.config, "gc_interval", 100)
        if self.step % gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()


    def train(self):
        total_train_steps = getattr(self.config, "total_train_steps", None)

        while True:
            if total_train_steps is not None and self.step >= total_train_steps:
                if self.is_main_process:
                    print(f"Reached total_train_steps={total_train_steps}, stopping training.")
                # Final save (only if no_save is False).
                if not getattr(self.config, "no_save", False):
                    torch.cuda.empty_cache()
                    self.save()
                barrier()
                return

            batch = next(self.dataloader)
            self.train_one_step(batch)
                
            no_save = getattr(self.config, "no_save", False)
            log_iters = getattr(self.config, "log_iters", 500)
            if (not no_save) and self.step % log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            barrier()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
