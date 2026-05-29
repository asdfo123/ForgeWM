import inspect
import json
import os
import types
from typing import List, Optional

import torch
from einops import rearrange
from safetensors.torch import load_file as load_safetensors
from torch import nn

from utils.scheduler import FlowMatchScheduler, SchedulerInterface
from wan.modules.causal_model import CausalWanModel
from wan.modules.clip import CLIPModel
from wan.modules.model import WanModel
from wan.modules.vae import _video_vae


class WanTextEncoder(nn.Module):
    """Stage1-only stub kept for import compatibility."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, text_prompts: List[str]) -> dict:
        return {}


class WanVAEWrapper(nn.Module):
    def __init__(
        self,
        vae_path: str = "ckpts/MG2-base/Wan2.1_VAE.pth",
        clip_checkpoint_path: str = "ckpts/MG2-base/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth",
        clip_tokenizer_path: str = "ckpts/MG2-base/xlm-roberta-large",
    ):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        self.model = _video_vae(
            pretrained_path=vae_path,
            z_dim=16,
        ).eval().requires_grad_(False)

        self.clip = CLIPModel(
            dtype=torch.float32,
            device=torch.device("cpu"),
            checkpoint_path=clip_checkpoint_path,
            tokenizer_path=clip_tokenizer_path,
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        device, dtype = pixel.device, pixel.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        target_dtype = next(self.model.parameters()).dtype
        latent = latent.to(dtype=target_dtype)
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [
            self.mean.to(device=device, dtype=dtype),
            1.0 / self.std.to(device=device, dtype=dtype),
        ]

        decode_function = self.model.cached_decode if use_cache else self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def encode_visual_context_from_pixels(self, pixel: torch.Tensor) -> torch.Tensor:
        if pixel.ndim != 5:
            raise ValueError(f"Expected pixel shape [B, F, C, H, W], got {tuple(pixel.shape)}")

        first_frame = pixel[:, 0].float()
        target_device = first_frame.device
        self.clip.model.to(target_device)
        videos = [img[:, None, :, :] for img in first_frame]
        with torch.no_grad():
            visual_context = self.clip.visual(videos)
        return visual_context

    def encode_visual_context_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        pixels = self.decode_to_pixel(latent)
        return self.encode_visual_context_from_pixels(pixels)


class WanDiffusionWrapper(nn.Module):
    @staticmethod
    def _resolve_model_path(model_name: str) -> str:
        # Accept absolute paths, relative paths (./ckpts/...), or bare names (legacy)
        if os.path.isabs(model_name) or os.path.exists(model_name):
            return model_name
        # Legacy fallback: bare model name → look in wan_models/
        fallback = f"wan_models/{model_name}/"
        if os.path.exists(fallback):
            return fallback
        # Return as-is and let downstream raise a clear error
        return model_name

    @staticmethod
    def _find_first_existing(directory: str, candidates: list[str]) -> Optional[str]:
        for candidate in candidates:
            path = os.path.join(directory, candidate)
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def _load_local_state_dict(weights_path: str) -> dict:
        if weights_path.endswith(".safetensors"):
            return load_safetensors(weights_path, device="cpu")

        state_dict = torch.load(weights_path, map_location="cpu")
        if isinstance(state_dict, dict):
            for key in ("state_dict", "model", "generator", "generator_ema"):
                nested = state_dict.get(key)
                if isinstance(nested, dict):
                    return nested
        if not isinstance(state_dict, dict):
            raise TypeError(f"Unsupported checkpoint format at {weights_path}")
        return state_dict

    @staticmethod
    def _build_model_from_config(model_cls, config_dict: dict, overrides: dict):
        valid_keys = set(inspect.signature(model_cls.__init__).parameters) - {"self"}
        init_kwargs = {
            key: value
            for key, value in config_dict.items()
            if not key.startswith("_") and key in valid_keys
        }
        init_kwargs.update({key: value for key, value in overrides.items() if key in valid_keys})
        return model_cls(**init_kwargs)

    @classmethod
    def _load_local_model(
        cls,
        model_path: str,
        is_causal: bool,
        local_attn_size: int,
        sink_size: int,
        action_config: Optional[dict],
    ):
        if not os.path.isdir(model_path):
            return None

        config_path = cls._find_first_existing(model_path, ["config.json", "base_config.json"])
        weights_path = cls._find_first_existing(
            model_path,
            [
                "diffusion_pytorch_model.safetensors",
                "base_distill.safetensors",
                "diffusion_pytorch_model.bin",
                "pytorch_model.bin",
            ],
        )
        if config_path is None or weights_path is None:
            return None

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        model_cls = CausalWanModel if is_causal else WanModel
        overrides = {}
        if is_causal:
            overrides["local_attn_size"] = local_attn_size
            overrides["sink_size"] = sink_size
            overrides["action_config"] = {} if action_config is None else dict(action_config)
        else:
            # Bid `WanModel` also accepts action_config for I2V w/ MG2-style
            # mouse+keyboard injection (see wan/modules/model.py:283).  We
            # need to pass it so that ActionModule is constructed and the
            # base ckpt's .action_model.* weights load correctly.  Without
            # this, the bid model would be built without ActionModule and
            # all action_model.* keys would land in `unexpected`, causing
            # _load_local_model to fail.
            overrides["action_config"] = {} if action_config is None else dict(action_config)

        model = cls._build_model_from_config(model_cls, config_dict, overrides)
        state_dict = cls._load_local_state_dict(weights_path)
        candidates = [state_dict]
        if any(key.startswith("model.") for key in state_dict):
            candidates.append({
                key[len("model."):] if key.startswith("model.") else key: value
                for key, value in state_dict.items()
            })

        # When action is ablated (action_config is None / empty), the base ckpt
        # still contains .action_model.* weights. Those are legitimately unused
        # and should not be treated as a load failure.
        action_disabled = not overrides.get("action_config")
        # When action is enabled on a SUBSET of blocks (e.g. MG2 distilled
        # recipe: blocks=[0..14] on a 30-layer DiT), the ckpt may have
        # action_model.* weights for blocks that our model doesn't instantiate.
        # Those are also legitimately unused and should not fail the load.
        action_blocks_subset = None
        if not action_disabled:
            ac_blocks = overrides.get("action_config", {}).get("blocks", None)
            num_layers = config_dict.get("num_layers", 30)
            if ac_blocks is not None and len(ac_blocks) < num_layers:
                action_blocks_subset = set(int(b) for b in ac_blocks)

        best_missing = None
        best_unexpected = None
        for candidate in candidates:
            missing, unexpected = model.load_state_dict(candidate, strict=False)
            if action_disabled:
                unexpected = [k for k in unexpected if ".action_model." not in k]
            elif action_blocks_subset is not None:
                import re
                def _in_subset(k):
                    m = re.match(r"blocks\.(\d+)\.action_model\.", k)
                    if m is None:
                        return True  # not an action_model key, keep as-is
                    return int(m.group(1)) in action_blocks_subset
                unexpected = [k for k in unexpected if _in_subset(k)]
            if not missing and not unexpected:
                return model
            if best_missing is None or len(missing) + len(unexpected) < len(best_missing) + len(best_unexpected):
                best_missing, best_unexpected = missing, unexpected

        raise RuntimeError(
            f"Failed to load local model from {model_path}. "
            f"Missing keys: {best_missing[:10]}, unexpected keys: {best_unexpected[:10]}"
        )

    def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=5.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0,
        action_config=None,
    ):
        super().__init__()

        model_path = self._resolve_model_path(model_name)
        self.model = self._load_local_model(
            model_path=model_path,
            is_causal=is_causal,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            action_config=action_config,
        )
        if self.model is None:
            if is_causal:
                self.model = CausalWanModel.from_pretrained(
                    model_path,
                    local_attn_size=local_attn_size,
                    sink_size=sink_size,
                    action_config={} if action_config is None else dict(action_config),
                    low_cpu_mem_usage=False,
                )
            else:
                self.model = WanModel.from_pretrained(
                    model_path,
                    action_config={} if action_config is None else dict(action_config),
                    low_cpu_mem_usage=False,
                )
        self.model.eval()

        self.uniform_timestep = not is_causal
        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = None
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        try:
            self.model.enable_gradient_checkpointing()
        except TypeError:
            self.model._set_gradient_checkpointing(None, True)

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps],
        )

        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps],
        )
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def _prepare_model_kwargs(self, conditional_dict: dict, device: torch.device, dtype: torch.dtype) -> dict:
        visual_context = conditional_dict["visual_context"].to(device=device, dtype=dtype)
        cond_concat = conditional_dict["cond_concat"].permute(0, 2, 1, 3, 4).to(device=device, dtype=dtype)
        mouse_condition = conditional_dict.get("mouse_condition", conditional_dict.get("mouse_cond", None))
        keyboard_condition = conditional_dict.get("keyboard_condition", conditional_dict.get("keyboard_cond", None))

        kwargs = {
            "visual_context": visual_context,
            "cond_concat": cond_concat,
        }
        if mouse_condition is not None:
            kwargs["mouse_cond"] = mouse_condition.to(device=device, dtype=dtype)
        if keyboard_condition is not None:
            kwargs["keyboard_cond"] = keyboard_condition.to(device=device, dtype=dtype)
        return kwargs

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        kv_cache_mouse: Optional[List[dict]] = None,
        kv_cache_keyboard: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        del classify_mode, concat_time_embeddings

        def normalize_latent_layout(latent: torch.Tensor, name: str) -> torch.Tensor:
            if latent.ndim != 5:
                raise ValueError(f"Expected 5D latent tensor for {name}, got {tuple(latent.shape)}")
            if latent.shape[2] == 16:
                return latent
            if latent.shape[1] == 16:
                return latent.permute(0, 2, 1, 3, 4)
            raise ValueError(f"Cannot infer latent layout for {name} from shape {tuple(latent.shape)}")

        latent_bfchw = normalize_latent_layout(noisy_image_or_video, "noisy_image_or_video")
        clean_latent_bfchw = None if clean_x is None else normalize_latent_layout(clean_x, "clean_x")

        input_timestep = timestep[:, 0] if self.uniform_timestep else timestep
        # Use the latent's own dtype (bf16 under mixed_precision) as the
        # model's effective forward dtype.  Querying self.model.dtype can
        # return fp32 on FSDP-wrapped modules because the underlying module
        # still carries fp32 master params — but forward pass sees bf16
        # shards, and our inputs (x + cond_concat) must match.
        model_dtype = latent_bfchw.dtype

        model_kwargs = self._prepare_model_kwargs(
            conditional_dict,
            device=latent_bfchw.device,
            dtype=model_dtype,
        )

        latent_input = latent_bfchw.permute(0, 2, 1, 3, 4).to(model_dtype)
        if kv_cache is not None:
            flow_pred = self.model(
                latent_input,
                t=input_timestep,
                kv_cache=kv_cache,
                kv_cache_mouse=kv_cache_mouse,
                kv_cache_keyboard=kv_cache_keyboard,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                **model_kwargs,
            ).permute(0, 2, 1, 3, 4)
        else:
            forward_kwargs = dict(model_kwargs)
            if clean_latent_bfchw is not None:
                forward_kwargs["clean_x"] = clean_latent_bfchw.permute(0, 2, 1, 3, 4).to(model_dtype)
                forward_kwargs["aug_t"] = aug_t
            flow_pred = self.model(
                latent_input,
                t=input_timestep,
                **forward_kwargs,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=latent_bfchw.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])
        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()
