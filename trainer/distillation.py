import gc
import logging
from utils.dataset import (
    cycle,
    TextDataset,
    ActionLatentLMDBDataset,
    ShardingActionLatentLMDBDataset,
)
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import set_seed
import torch.distributed as dist
from omegaconf import OmegaConf
from model import DMD
import torch
import wandb
import time
import os


class Trainer:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.world_size = dist.get_world_size()

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

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        else:
            raise ValueError("Invalid distribution matching loss")

        # Save pretrained model state_dicts to CPU
        self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "generator_cpu_offload", False)
        )

        self.model.real_score = fsdp_wrap(
            self.model.real_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "real_score_cpu_offload", False)
        )

        self.model.fake_score = fsdp_wrap(
            self.model.fake_score,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.fake_score_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "fake_score_cpu_offload", False)
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        )

        if not config.no_visualize or config.load_raw_video or getattr(config, "action_config", None):
            # Need VAE on GPU when:
            # - visualizing (legacy)
            # - load_raw_video (encode raw frames)
            # - action_config is set (MG2 I2V needs VAE for Step 2.6
            #   cond_concat construction, see trainer/diffusion.py).
            self.model.vae = self.model.vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        # Optimizer factory: AdamW8bit halves Adam state memory (fp32 momentum
        # + variance → int8) which is critical in Stage-3 DMD where we hold
        # optimizer states for TWO 1.8B trainables (generator + fake_score).
        # Enable via config: adam_8bit: true (default false for back-compat).
        _use_adam_8bit = bool(getattr(config, "adam_8bit", False))
        if _use_adam_8bit:
            try:
                from bitsandbytes.optim import AdamW8bit as _AdamWClass
                if dist.get_rank() == 0:
                    print("[optim] Using bitsandbytes AdamW8bit for generator + critic")
            except ImportError:
                if dist.get_rank() == 0:
                    print("[optim][warn] adam_8bit=true but bitsandbytes unavailable; "
                          "falling back to torch.optim.AdamW")
                _AdamWClass = torch.optim.AdamW
        else:
            _AdamWClass = torch.optim.AdamW

        self.generator_optimizer = _AdamWClass(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = _AdamWClass(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader.
        #
        # For MG2 I2V DMD we need action conditions AND a reference frame
        # per sample, both of which live in the action-latent LMDBs built
        # for Stage-1/Stage-2.  Otherwise fall back to the legacy
        # text-only dataset (T2V DMD).
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
            dataset = TextDataset(config.data_path)
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
            self.model.generator.load_state_dict(
                state_dict, strict=not bool(getattr(self.config, "action_config", None))
            )
            # NOTE: fake_score (critic) and real_score (teacher) intentionally
            # stay on the original Foundation model weights loaded from
            # model_kwargs.model_name — they are NOT overwritten by
            # generator_ckpt.  This matches the CF/DMD2 original design
            # (Causal-Forcing/model/base.py L34/L37): both score networks
            # start from the same teacher prior; the critic then drifts
            # toward the student's distribution through training, and
            # DMD's (real - fake) signal stays well-defined throughout.
            # Loading Stage-2 ckpt into fake_score (an earlier "fix" we
            # carried) collapses the critic onto the student's prior at
            # init, weakening the DMD KL signal — reverted here.
            # self.fake_score_state_dict_cpu was already cached from the
            # original-base fake_score at __init__ time (L67).

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        if self.config.ema_start_step < self.step:
            state_dict = {
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

    def save_critic(self):
        print("Start gathering distributed model states...")
        
        critic_state_dict = fsdp_state_dict(
            self.model.fake_score)

        
        state_dict = critic_state_dict

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))
            
    def fwdbwd_one_step(self, batch, train_generator, clean_latent=None):
        self.model.eval()  # prevent any randomness (e.g. dropout)

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts / latents.
        text_prompts = batch["prompts"]
        action_config = getattr(self.config, "action_config", None)

        # For MG2 I2V with action data, the reference frame is the first
        # latent in each clip (clean_latent from action LMDB).  Fall back
        # to legacy ode_latent / None for older configs.
        if action_config is not None and "clean_latent" in batch:
            clean_latent_batch = batch["clean_latent"].to(
                device=self.device, dtype=self.dtype)  # [B, F, C, H, W]
            image_latent = clean_latent_batch[:, 0:1]   # [B, 1, C, H, W]
        elif self.config.i2v and "ode_latent" in batch:
            clean_latent_batch = None
            image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
                device=self.device, dtype=self.dtype)
        else:
            clean_latent_batch = None
            image_latent = None

        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        # ─── Stage-3 shape sanity print (step 0 + every 1000 steps) ───
        if self.is_main_process and (self.step == 0 or self.step % 1000 == 0):
            mouse_shape = tuple(batch["mouse_condition"].shape) if "mouse_condition" in batch else None
            kbd_shape = tuple(batch["keyboard_condition"].shape) if "keyboard_condition" in batch else None
            cl_shape = tuple(clean_latent_batch.shape) if clean_latent_batch is not None else None
            il_shape = tuple(image_latent.shape) if image_latent is not None else None
            print(
                f"[shape][DMD step {self.step}] train_generator={train_generator} "
                f"config.image_or_video_shape={image_or_video_shape} "
                f"clean_latent={cl_shape} image_latent={il_shape} "
                f"mouse={mouse_shape} kbd={kbd_shape}"
            )

        # Step 2: Extract the conditional infos (text)
        with torch.no_grad():
            conditional_dict = self.model.text_encoder(
                text_prompts=text_prompts)

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * batch_size)
                unconditional_dict = {k: v.detach()
                                      for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

        # Step 2.5: Inject action conditions if action_config is present
        if action_config is not None:
            target_kbd_dim = action_config.get("keyboard_dim_in", 4)
            vae_tcr = action_config.get("vae_time_compression_ratio", 4)
            # SF rollout uses num_training_frames up to 46 latent frames.
            # Dataset only ships 21 latent → 81 pixel frames of action, so
            # we extend via tile+slice to match whatever the SF pipeline
            # will ask for.  Each training step's actual rollout length is
            # sampled randomly inside model/base.py, up to num_training_frames;
            # we size action to num_training_frames here so the pipeline's
            # cond_current slicing always lands in-bounds.
            num_training_frames = getattr(self.config, "num_training_frames", 21)
            target_pixel_frames = (num_training_frames - 1) * vae_tcr + 1  # e.g. 181
            if "mouse_condition" in batch and "keyboard_condition" in batch:
                mouse_cond = batch["mouse_condition"].to(device=self.device, dtype=self.dtype)
                kbd = batch["keyboard_condition"].to(device=self.device, dtype=self.dtype)
                # Pad 4-dim WASD to 6-dim for MG2 base.
                if kbd.shape[-1] < target_kbd_dim:
                    kbd = ActionLatentLMDBDataset.pad_keyboard(kbd, target_kbd_dim)
                keyboard_cond = kbd
                # Extend along time axis to target_pixel_frames via tile +
                # truncate (leaves first 81 frames as GT action, repeats
                # thereafter as a sequence-extending fallback for SF rollout).
                if mouse_cond.shape[1] < target_pixel_frames:
                    reps = (target_pixel_frames + mouse_cond.shape[1] - 1) // mouse_cond.shape[1]
                    mouse_cond = mouse_cond.repeat(1, reps, 1)[:, :target_pixel_frames]
                if keyboard_cond.shape[1] < target_pixel_frames:
                    reps = (target_pixel_frames + keyboard_cond.shape[1] - 1) // keyboard_cond.shape[1]
                    keyboard_cond = keyboard_cond.repeat(1, reps, 1)[:, :target_pixel_frames]
            else:
                # Generate random action conditions as fallback
                num_raw_frames = target_pixel_frames
                mouse_dim = action_config.get("mouse_dim_in", 2)
                mouse_cond = (torch.rand(batch_size, num_raw_frames, mouse_dim, device=self.device, dtype=self.dtype) - 0.5) * 0.1
                keyboard_cond = (torch.rand(batch_size, num_raw_frames, target_kbd_dim, device=self.device, dtype=self.dtype) > 0.5).float()
            conditional_dict["mouse_condition"] = mouse_cond
            conditional_dict["keyboard_condition"] = keyboard_cond
            unconditional_dict["mouse_condition"] = torch.zeros_like(mouse_cond)
            unconditional_dict["keyboard_condition"] = torch.zeros_like(keyboard_cond)

        # Step 2.6: Construct I2V cond_concat + visual_context for MG2.
        # Mirrors trainer/diffusion.py and trainer/naive_cd.py — one
        # single-frame VAE decode, reused for both.  Uses image_latent
        # (the first latent frame) as the I2V reference image.
        if image_latent is not None and hasattr(self.model.generator, 'model') \
                and getattr(self.model.generator.model, 'in_dim', 16) > 16:
            B = image_latent.shape[0]
            # For long rollout we size cond_concat to num_training_frames
            # (the maximum rollout length SF can ask for).  The pipeline's
            # cond_current slices it per-block, so over-allocating is fine.
            num_training_frames = getattr(self.config, "num_training_frames", None)
            F_total = num_training_frames if num_training_frames is not None else image_or_video_shape[1]
            _, C, H, W = image_or_video_shape[1:]
            F = F_total
            vae_tcr = int(action_config.get("vae_time_compression_ratio", 4)) if action_config else 4
            num_pixel_frames = (F - 1) * vae_tcr + 1

            with torch.no_grad():
                # Single-frame decode (Wan2.1 VAE is time-causal; see
                # scripts/verify_vae_decode_equiv.py).
                first_frame_pixels = self.model.vae.decode_to_pixel(image_latent)  # [B, 1, 3, H_pix, W_pix]
                H_pix, W_pix = first_frame_pixels.shape[-2], first_frame_pixels.shape[-1]
                first_frame = first_frame_pixels[:, 0:1].to(self.dtype)
                pad_pix = torch.zeros(
                    B, num_pixel_frames - 1, 3, H_pix, W_pix,
                    device=self.device, dtype=self.dtype,
                )
                padded_pixels_BTCHW = torch.cat([first_frame, pad_pix], dim=1)
                padded_pixels_BCTHW = padded_pixels_BTCHW.permute(0, 2, 1, 3, 4)
                img_cond = self.model.vae.encode_to_latent(padded_pixels_BCTHW).to(self.dtype)
                visual_context = self.model.vae.encode_visual_context_from_pixels(first_frame_pixels)
            mask = torch.zeros(B, F, 4, H, W, device=self.device, dtype=self.dtype)
            mask[:, 0:1] = 1
            cond_concat = torch.cat([mask, img_cond], dim=2)
            conditional_dict["cond_concat"] = cond_concat
            conditional_dict["visual_context"] = visual_context
            # unconditional dict also needs cond_concat/visual_context
            # since _compute_kl_grad calls real_score/fake_score with it.
            unconditional_dict["cond_concat"] = cond_concat
            unconditional_dict["visual_context"] = visual_context

        # Step 3: Store gradients for the generator (if training the generator)
        if train_generator:
            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None
            )

            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm})

            return generator_log_dict
        else:
            generator_log_dict = {}

        # Step 4: Store gradients for the critic (if training the critic)
        critic_loss, critic_log_dict = self.model.critic_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent if self.config.i2v else None
        )

        critic_loss.backward()
        critic_grad_norm = self.model.fake_score.clip_grad_norm_(
            self.max_grad_norm_critic)

        critic_log_dict.update({"critic_loss": critic_loss,
                                "critic_grad_norm": critic_grad_norm})

        return critic_log_dict


    def train(self):
        start_step = self.step
       
        total_train_steps = getattr(self.config, 'total_train_steps', None)
        while True:
            if total_train_steps is not None and self.step >= total_train_steps:
                if self.is_main_process:
                    print(f"Reached total_train_steps={total_train_steps}, stopping training.")
                # Final save
                if not self.config.no_save:
                    torch.cuda.empty_cache()
                    self.save()
                break

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0

            # Train the generator
            if TRAIN_GENERATOR:
                self.generator_optimizer.zero_grad(set_to_none=True)
                
                batch = next(self.dataloader)
                generator_log_dict = self.fwdbwd_one_step(batch, True)

                self.generator_optimizer.step()
                if self.generator_ema is not None:
                    self.generator_ema.update(self.model.generator)
                
                
                

            # Train the critic
            self.critic_optimizer.zero_grad(set_to_none=True)
            batch = next(self.dataloader)
            critic_log_dict = self.fwdbwd_one_step(batch, False)
                
            self.critic_optimizer.step()

            # Increment the step since we finished gradient update
            self.step += 1

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                if TRAIN_GENERATOR:
                    wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item(),
                            "dmdtrain_gradient_norm": generator_log_dict["dmdtrain_gradient_norm"].mean().item()
                        }
                    )

                wandb_loss_dict.update(
                    {
                        "critic_loss": critic_log_dict["critic_loss"].mean().item(),
                        "critic_grad_norm": critic_log_dict["critic_grad_norm"].mean().item()
                    }
                )

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

                # Always print to stdout regardless of wandb setting
                log_str = f"step={self.step}"
                for k, v in wandb_loss_dict.items():
                    log_str += f" | {k}={v:.4f}"
                print(log_str, flush=True)

            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time
