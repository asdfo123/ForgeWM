from utils.lmdb_ import get_array_shape_from_lmdb, retrieve_row_from_lmdb
from torch.utils.data import Dataset
import numpy as np
import torch
import lmdb
import json
from pathlib import Path
from PIL import Image
import os


class TextDataset(Dataset):
    def __init__(self, prompt_path, extended_prompt_path=None):
        with open(prompt_path, encoding="utf-8") as f:
            self.prompt_list = [line.rstrip() for line in f]

        if extended_prompt_path is not None:
            with open(extended_prompt_path, encoding="utf-8") as f:
                self.extended_prompt_list = [line.rstrip() for line in f]
            assert len(self.extended_prompt_list) == len(self.prompt_list)
        else:
            self.extended_prompt_list = None

    def __len__(self):
        return len(self.prompt_list)

    def __getitem__(self, idx):
        batch = {
            "prompts": self.prompt_list[idx],
            "idx": idx,
        }
        if self.extended_prompt_list is not None:
            batch["extended_prompts"] = self.extended_prompt_list[idx]
        return batch


class ODERegressionLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }





class ActionODERegressionLMDBDataset(Dataset):
    """ODE regression dataset extended with mouse and keyboard action conditions."""

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.mouse_shape = get_array_shape_from_lmdb(self.env, 'mouse_conditions')
        self.keyboard_shape = get_array_shape_from_lmdb(self.env, 'keyboard_conditions')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - ode_latent: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width)
            - mouse_condition: Tensor of mouse movements
            - keyboard_condition: Tensor of keyboard states
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env, "prompts", str, idx
        )
        mouse_cond = retrieve_row_from_lmdb(
            self.env, "mouse_conditions", np.float32, idx,
            shape=self.mouse_shape[1:]
        )
        keyboard_cond = retrieve_row_from_lmdb(
            self.env, "keyboard_conditions", np.float32, idx,
            shape=self.keyboard_shape[1:]
        )
        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32),
            "mouse_condition": torch.tensor(mouse_cond, dtype=torch.float32),
            "keyboard_condition": torch.tensor(keyboard_cond, dtype=torch.float32),
        }


class LatentLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        """
        Outputs:
            - prompts: List of Strings
            - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env,
            "prompts", str, idx
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1]
        }


class ActionLatentLMDBDataset(Dataset):
    """LatentLMDBDataset extended with mouse and keyboard action conditions."""

    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.env = lmdb.open(data_path, readonly=True,
                             lock=False, readahead=False, meminit=False)

        self.latents_shape = get_array_shape_from_lmdb(self.env, 'latents')
        self.mouse_shape = get_array_shape_from_lmdb(self.env, 'mouse_conditions')
        self.keyboard_shape = get_array_shape_from_lmdb(self.env, 'keyboard_conditions')
        self.max_pair = max_pair

    def __len__(self):
        return min(self.latents_shape[0], self.max_pair)

    def __getitem__(self, idx):
        latents = retrieve_row_from_lmdb(
            self.env,
            "latents", np.float16, idx, shape=self.latents_shape[1:]
        )
        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.env, "prompts", str, idx
        )
        mouse_cond = retrieve_row_from_lmdb(
            self.env, "mouse_conditions", np.float32, idx,
            shape=self.mouse_shape[1:]
        )
        keyboard_cond = retrieve_row_from_lmdb(
            self.env, "keyboard_conditions", np.float32, idx,
            shape=self.keyboard_shape[1:]
        )
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1],
            "mouse_condition": torch.tensor(mouse_cond, dtype=torch.float32),
            "keyboard_condition": torch.tensor(keyboard_cond, dtype=torch.float32),
        }

    @staticmethod
    def pad_keyboard(keyboard_cond, target_dim):
        """Pad keyboard condition to target_dim if needed (e.g. 4->6)."""
        if keyboard_cond.shape[-1] < target_dim:
            pad = torch.zeros(*keyboard_cond.shape[:-1], target_dim - keyboard_cond.shape[-1],
                              dtype=keyboard_cond.dtype, device=keyboard_cond.device)
            return torch.cat([keyboard_cond, pad], dim=-1)
        return keyboard_cond


class ShardingActionLatentLMDBDataset(Dataset):
    """Sharded variant of ActionLatentLMDBDataset.

    `data_path` points to a directory whose immediate children are each
    a stand-alone LMDB shard (e.g. .../action_data_2003/shard00).  Only
    children that actually look like an LMDB (contain `data.mdb`) and
    have the expected `latents_shape` key are included, so junk
    subdirs like `logs/` or `smoke/` are skipped automatically.
    """

    def __init__(self, data_path: str, max_pair: int = int(1e8),
                 allowed_prefixes=None):
        self.envs = []
        self.index = []
        self.latents_shape = []
        self.mouse_shape = []
        self.keyboard_shape = []
        self.shard_names = []

        for fname in sorted(os.listdir(data_path)):
            if allowed_prefixes is not None and not any(
                    fname.startswith(p) for p in allowed_prefixes):
                continue
            shard_path = os.path.join(data_path, fname)
            if not os.path.isdir(shard_path):
                continue
            if not os.path.exists(os.path.join(shard_path, 'data.mdb')):
                continue

            env = lmdb.open(shard_path, readonly=True,
                            lock=False, readahead=False, meminit=False)
            # Skip shards that did not finalize their shape metadata.
            try:
                lat_shape = get_array_shape_from_lmdb(env, 'latents')
                mouse_shape = get_array_shape_from_lmdb(env, 'mouse_conditions')
                kbd_shape = get_array_shape_from_lmdb(env, 'keyboard_conditions')
            except Exception:
                env.close(); continue

            shard_id = len(self.envs)
            self.envs.append(env)
            self.shard_names.append(fname)
            self.latents_shape.append(lat_shape)
            self.mouse_shape.append(mouse_shape)
            self.keyboard_shape.append(kbd_shape)
            for local_i in range(lat_shape[0]):
                self.index.append((shard_id, local_i))

        self.max_pair = max_pair

        if not self.envs:
            raise RuntimeError(
                f"No usable LMDB shards found under {data_path}")

    def summary(self):
        lines = [f"{len(self.envs)} shards, {len(self.index)} total clips"]
        for i, (name, shp) in enumerate(zip(self.shard_names, self.latents_shape)):
            lines.append(f"  [{i:2d}] {name:20s}  clips={shp[0]}")
        return '\n'.join(lines)

    def __len__(self):
        return min(len(self.index), self.max_pair)

    def __getitem__(self, idx):
        shard_id, local_idx = self.index[idx]
        env = self.envs[shard_id]

        latents = retrieve_row_from_lmdb(
            env, "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:])
        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            env, "prompts", str, local_idx)
        mouse_cond = retrieve_row_from_lmdb(
            env, "mouse_conditions", np.float32, local_idx,
            shape=self.mouse_shape[shard_id][1:])
        keyboard_cond = retrieve_row_from_lmdb(
            env, "keyboard_conditions", np.float32, local_idx,
            shape=self.keyboard_shape[shard_id][1:])
        return {
            "prompts": prompts,
            "clean_latent": torch.tensor(latents, dtype=torch.float32)[-1],
            "mouse_condition": torch.tensor(mouse_cond, dtype=torch.float32),
            "keyboard_condition": torch.tensor(keyboard_cond, dtype=torch.float32),
        }


class ShardingLMDBDataset(Dataset):
    def __init__(self, data_path: str, max_pair: int = int(1e8)):
        self.envs = []
        self.index = []

        for fname in sorted(os.listdir(data_path)):
            path = os.path.join(data_path, fname)
            env = lmdb.open(path,
                            readonly=True,
                            lock=False,
                            readahead=False,
                            meminit=False)
            self.envs.append(env)

        self.latents_shape = [None] * len(self.envs)
        for shard_id, env in enumerate(self.envs):
            self.latents_shape[shard_id] = get_array_shape_from_lmdb(env, 'latents')
            for local_i in range(self.latents_shape[shard_id][0]):
                self.index.append((shard_id, local_i))

            # print("shard_id ", shard_id, " local_i ", local_i)

        self.max_pair = max_pair

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        """
            Outputs:
                - prompts: List of Strings
                - latents: Tensor of shape (num_denoising_steps, num_frames, num_channels, height, width). It is ordered from pure noise to clean image.
        """
        shard_id, local_idx = self.index[idx]

        latents = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "latents", np.float16, local_idx,
            shape=self.latents_shape[shard_id][1:]
        )

        if len(latents.shape) == 4:
            latents = latents[None, ...]

        prompts = retrieve_row_from_lmdb(
            self.envs[shard_id],
            "prompts", str, local_idx
        )

        return {
            "prompts": prompts,
            "ode_latent": torch.tensor(latents, dtype=torch.float32)
        }



class TextImagePairDataset(Dataset):
    def __init__(
        self,
        data_dir,
        transform=None,
        eval_first_n=-1,
        pad_to_multiple_of=None
    ):
        """
        Args:
            data_dir (str): Path to the directory containing:
                - target_crop_info_*.json (metadata file)
                - */ (subdirectory containing images with matching aspect ratio)
            transform (callable, optional): Optional transform to be applied on the image
        """
        self.transform = transform
        data_dir = Path(data_dir)

        # Find the metadata JSON file
        metadata_files = list(data_dir.glob('target_crop_info_*.json'))
        if not metadata_files:
            raise FileNotFoundError(f"No metadata file found in {data_dir}")
        if len(metadata_files) > 1:
            raise ValueError(f"Multiple metadata files found in {data_dir}")

        metadata_path = metadata_files[0]
        # Extract aspect ratio from metadata filename (e.g. target_crop_info_26-15.json -> 26-15)
        aspect_ratio = metadata_path.stem.split('_')[-1]

        # Use aspect ratio subfolder for images
        self.image_dir = data_dir / aspect_ratio
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        # Load metadata
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        eval_first_n = eval_first_n if eval_first_n != -1 else len(self.metadata)
        self.metadata = self.metadata[:eval_first_n]

        # Verify all images exist
        for item in self.metadata:
            image_path = self.image_dir / item['file_name']
            if not image_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")

        self.dummy_prompt = "DUMMY PROMPT"
        self.pre_pad_len = len(self.metadata)
        if pad_to_multiple_of is not None and len(self.metadata) % pad_to_multiple_of != 0:
            # Duplicate the last entry
            self.metadata += [self.metadata[-1]] * (
                pad_to_multiple_of - len(self.metadata) % pad_to_multiple_of
            )

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        """
        Returns:
            dict: A dictionary containing:
                - image: PIL Image
                - caption: str
                - target_bbox: list of int [x1, y1, x2, y2]
                - target_ratio: str
                - type: str
                - origin_size: tuple of int (width, height)
        """
        item = self.metadata[idx]

        # Load image
        image_path = self.image_dir / item['file_name']
        image = Image.open(image_path).convert('RGB')

        # Apply transform if specified
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'prompts': item['caption'],
            'target_bbox': item['target_crop']['target_bbox'],
            'target_ratio': item['target_crop']['target_ratio'],
            'type': item['type'],
            'origin_size': (item['origin_width'], item['origin_height']),
            'idx': idx
        }



def cycle(dl):
    while True:
        for data in dl:
            yield data
