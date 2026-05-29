import gc
import logging
from utils.dataset import cycle
from utils.dataset import LatentLMDBDataset, ActionLatentLMDBDataset, ShardingActionLatentLMDBDataset
from utils.distributed import EMA_FSDP, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os
from model.consistency_distillation import NaiveConsistency



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
        self.model = NaiveConsistency(config, device=self.device)

        self.model.generator = fsdp_wrap(
            self.model.generator,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )
        
        self.model.generator_ema = fsdp_wrap(
            self.model.generator_ema,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.generator_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.teacher = fsdp_wrap(
            self.model.teacher,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.real_score_fsdp_wrap_strategy,
            cpu_offload=True
        )

        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=config.sharding_strategy,
            mixed_precision=config.mixed_precision,
            wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
            cpu_offload=True
        )

        # Move the VAE (used to construct cond_concat / visual_context) to GPU.
        # Stage-1 trainer does the same (see trainer/diffusion.py:86).
        self.model.vae = self.model.vae.to(
            device=self.device,
            dtype=torch.bfloat16 if config.mixed_precision else torch.float32,
        )

        
        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.generator_ema = EMA_FSDP(self.model.generator, decay=self.config.ema_weight)
        
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
        # 7. Load the causal diffusion model as the teacher model
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
                state_dict, strict=True
            )
            self.model.teacher.load_state_dict(
                state_dict, strict=True
            )

        #############################################################################################################
        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None
        
        

    def save(self):
        print("Start gathering distributed model states...")
        generator_state_dict = fsdp_state_dict(
            self.model.generator)

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

            
    def fwdbwd_one_step(self, batch, clean_latent=None):
        self.model.eval()

        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Step 1: Get the next batch of text prompts
        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        image_or_video_shape = list(self.config.image_or_video_shape)
        # clean_latent arrives from DataLoader on CPU in float32; move to the
        # training device and dtype so VAE (GPU/bf16) can consume it.
        if clean_latent is not None:
            clean_latent = clean_latent.to(device=self.device, dtype=self.dtype)
        image_or_video_shape[0] = batch_size

        # ─── Stage-2 shape sanity print (resolution/360p sanity check) ───
        if self.is_main_process and (self.step == 0 or self.step % 1000 == 0):
            cfg_shape = list(self.config.image_or_video_shape)
            mouse_shape = tuple(batch["mouse_condition"].shape) if "mouse_condition" in batch else None
            kbd_shape = tuple(batch["keyboard_condition"].shape) if "keyboard_condition" in batch else None
            cl_shape = tuple(clean_latent.shape) if clean_latent is not None else None
            print(
                f"[shape][CD step {self.step}] "
                f"config.image_or_video_shape={cfg_shape} "
                f"clean_latent={cl_shape} "
                f"mouse={mouse_shape} keyboard={kbd_shape}"
            )

        # Step 2: Extract the conditional infos
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

        # Step 2.5: Inject action conditions if available
        action_config = getattr(self.config, "action_config", None)
        if action_config is not None:
            target_kbd_dim = action_config.get("keyboard_dim_in", 4)
            if "mouse_condition" in batch and "keyboard_condition" in batch:
                conditional_dict["mouse_condition"] = batch["mouse_condition"].to(
                    device=self.device, dtype=self.dtype)
                kbd = batch["keyboard_condition"].to(device=self.device, dtype=self.dtype)
                # Pad 4-dim WASD from data to 6-dim for MG2 base model.
                if kbd.shape[-1] < target_kbd_dim:
                    kbd = ActionLatentLMDBDataset.pad_keyboard(kbd, target_kbd_dim)
                conditional_dict["keyboard_condition"] = kbd
            else:
                num_frames = image_or_video_shape[1]
                vae_tcr = action_config.get("vae_time_compression_ratio", 4)
                num_raw_frames = (num_frames - 1) * vae_tcr + 1
                mouse_dim = action_config.get("mouse_dim_in", 2)
                conditional_dict["mouse_condition"] = (
                    torch.rand(batch_size, num_raw_frames, mouse_dim, device=self.device, dtype=self.dtype) - 0.5) * 0.1
                conditional_dict["keyboard_condition"] = (
                    torch.rand(batch_size, num_raw_frames, target_kbd_dim, device=self.device, dtype=self.dtype) > 0.5).float()
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
        if hasattr(self.model.generator, 'model') and getattr(self.model.generator.model, 'in_dim', 16) > 16:
            B, F, C, H, W = clean_latent.shape
            vae_tcr = int(action_config.get("vae_time_compression_ratio", 4)) if action_config else 4
            num_pixel_frames = (F - 1) * vae_tcr + 1

            with torch.no_grad():
                # (1) Decode ONLY the first latent frame to pixel space.
                # MG2's Wan2.1 VAE is strictly time-causal, so this is
                # bitwise identical to decode(clean_latent)[:, 0:1] but
                # ~3s/step faster.  Verified in
                # scripts/verify_vae_decode_equiv.py (max |diff| = 0).
                first_frame_pixels = self.model.vae.decode_to_pixel(clean_latent[:, 0:1])  # [B, 1, 3, H_pix, W_pix]
                H_pix, W_pix = first_frame_pixels.shape[-2], first_frame_pixels.shape[-1]
                # (2) first-frame + zero-padding in pixel space
                first_frame = first_frame_pixels[:, 0:1].to(self.dtype)
                pad_pix = torch.zeros(
                    B, num_pixel_frames - 1, 3, H_pix, W_pix,
                    device=self.device, dtype=self.dtype,
                )
                padded_pixels_BTCHW = torch.cat([first_frame, pad_pix], dim=1)
                padded_pixels_BCTHW = padded_pixels_BTCHW.permute(0, 2, 1, 3, 4)
                # (3) VAE encode -> img_cond latent [B, F, 16, H, W]
                img_cond = self.model.vae.encode_to_latent(padded_pixels_BCTHW).to(self.dtype)
            # (4) 4-channel mask: first latent frame all-ones, rest zeros
            mask = torch.zeros(B, F, 4, H, W, device=self.device, dtype=self.dtype)
            mask[:, 0:1] = 1
            # (5) concat -> [B, F, 20, H, W]
            cond_concat = torch.cat([mask, img_cond], dim=2)
            conditional_dict["cond_concat"] = cond_concat
            # visual_context reuses the same first pixel frame — no extra decode.
            with torch.no_grad():
                visual_context = self.model.vae.encode_visual_context_from_pixels(first_frame_pixels)
            conditional_dict["visual_context"] = visual_context
            # CD 4.16 speedup calls teacher with BOTH conditional_dict and
            # unconditional_dict for CFG (v_pred = v_uncond + s*(v_cond-v_uncond)).
            # The I2V conds (cond_concat, visual_context) are the reference
            # frame — they must be present in both dicts (we only drop the
            # action signal for uncond, not the ref frame).
            unconditional_dict["cond_concat"] = cond_concat
            unconditional_dict["visual_context"] = visual_context
        else:
            # T2V models: only visual_context needed.  One single-frame decode.
            with torch.no_grad():
                first_frame_pixels = self.model.vae.decode_to_pixel(clean_latent[:, 0:1])
                visual_context = self.model.vae.encode_visual_context_from_pixels(first_frame_pixels)
            conditional_dict["visual_context"] = visual_context
            unconditional_dict["visual_context"] = visual_context

        # Step 3: Store gradients for the generator (if training the generator)
        generator_loss, generator_log_dict = self.model.generator_loss(
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            clean_latent=clean_latent,
            ema_model = self.generator_ema
        )
        generator_loss.backward()
        generator_grad_norm = self.model.generator.clip_grad_norm_(
            self.max_grad_norm_generator)

        generator_log_dict.update({"generator_loss": generator_loss,
                                    "generator_grad_norm": generator_grad_norm})

        return generator_log_dict
        

   

    def train(self):
        start_step = self.step
        total_train_steps = getattr(self.config, 'total_train_steps', None)

        while True:
            if total_train_steps is not None and self.step >= total_train_steps:
                if self.is_main_process:
                    print(f"Reached total_train_steps={total_train_steps}, stopping training.")
                if not self.config.no_save:
                    torch.cuda.empty_cache()
                    self.save()
                break

            self.generator_optimizer.zero_grad(set_to_none=True)

            batch = next(self.dataloader)
            generator_log_dict = self.fwdbwd_one_step(batch, clean_latent=batch["clean_latent"])
            

            self.generator_optimizer.step()
            if self.generator_ema is not None:
                self.generator_ema.update(self.model.generator)
            
              

            # Increment the step since we finished gradient update
            self.step += 1

           
            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0:
                torch.cuda.empty_cache()
                self.save()
                torch.cuda.empty_cache()

            # Logging
            if self.is_main_process:
                wandb_loss_dict = {}
                wandb_loss_dict.update(
                        {
                            "generator_loss": generator_log_dict["generator_loss"].mean().item(),
                            "generator_grad_norm": generator_log_dict["generator_grad_norm"].mean().item()
                        }
                    )

              

                if not self.disable_wandb:
                    wandb.log(wandb_loss_dict, step=self.step)

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
