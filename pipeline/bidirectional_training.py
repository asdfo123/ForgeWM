from utils.wan_wrapper import WanDiffusionWrapper
from utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class BidirectionalTrainingPipeline:
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
                 spatial_self: bool = True,
                 **kwargs):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.frame_seq_length = 1560
        self.num_frame_per_block = num_frame_per_block
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache1 = None
        self.kv_cache2 = None
        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only
        self.kv_cache_size = num_max_frames * self.frame_seq_length
        
        self.spatial_self = spatial_self

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
        
        # Step 3: Temporal denoising loop
        all_num_frames = [self.num_frame_per_block] * num_blocks 
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(len(all_num_frames), num_denoising_steps, device=noise.device)
        start_gradient_frame_index = num_output_frames - 21 # always 0 as long as we train 21 latent frames
        if start_gradient_frame_index != 0:
            raise NotImplementedError("start_gradient_frame_index is always 0 as long as we train 21 latent frames")
        
        
        noisy_input = noise
        for index, current_timestep in enumerate(self.denoising_step_list):
            # self.same_step_across_blocks is True
            if self.same_step_across_blocks:
                exit_flag = (index == exit_flags[0])
            else:
                raise NotImplementedError('Here t is a scalar denoting that all chunks are at the same t, but in the future we may set t a tensor denoting different chunks')  # Only backprop at the randomly selected timestep (consistent across all ranks)
            timestep = torch.ones(
                [batch_size, self.num_frame_per_block*num_blocks],
                device=noise.device,
                dtype=torch.int64) * current_timestep

            if not exit_flag:
                with torch.no_grad():
                    _,denoised_pred = self.generator(
                        noisy_image_or_video=noisy_input,
                        conditional_dict=conditional_dict,
                        timestep=timestep
                    )
                    next_timestep = self.denoising_step_list[index + 1]
                    noisy_input = self.scheduler.add_noise(
                        denoised_pred.flatten(0, 1),
                        torch.randn_like(denoised_pred.flatten(0, 1)),
                        next_timestep * torch.ones(
                            [batch_size * self.num_frame_per_block*num_blocks], device=noise.device, dtype=torch.long)
                    ).unflatten(0, denoised_pred.shape[:2])
                    print('denoise')
            else:
                _,output = self.generator(
                            noisy_image_or_video=noisy_input,
                            conditional_dict=conditional_dict,
                            timestep=timestep
                        )
                print('final denoise')
                break
        # ======================= SF -> TF modification ends  ============================
        
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

    