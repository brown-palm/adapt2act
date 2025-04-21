import os
import time
import random
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Union
from diffusers import AnimateDiffPipeline, LCMScheduler, MotionAdapter, DDIMScheduler
from diffusers.pipelines.animatediff import AnimateDiffPipelineOutput
from diffusers.utils import (USE_PEFT_BACKEND,
    deprecate,
    logging,
    replace_example_docstring,
    scale_lora_layers,
    unscale_lora_layers,
    export_to_gif)

from diffusers import AnimateDiffSparseControlNetPipeline, StableDiffusionPipeline
from diffusers.models import AutoencoderKL, SparseControlNetModel
from diffusers.schedulers import DPMSolverMultistepScheduler
from peft import PeftModel, LoraConfig
from transformers import CLIPTextModel, CLIPTokenizer

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from avdc.goal_diffusion import GoalGaussianDiffusion_SD, Trainer
from avdc.unet import UnetMW_SD
from einops import rearrange

import vc_models
from vc_models.models.vit import model_utils

import PIL
PipelineImageInput = Union[
    PIL.Image.Image,
    np.ndarray,
    torch.Tensor,
    List[PIL.Image.Image],
    List[np.ndarray],
    List[torch.Tensor],
]

def symlog(x):
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def get_lora_sd_pipeline(
    ckpt_dir, base_model_name_or_path=None, dtype=torch.float16, device="cuda", adapter_name="default"
):
    unet_sub_dir = os.path.join(ckpt_dir, "unet")
    text_encoder_sub_dir = os.path.join(ckpt_dir, "text_encoder")
    if os.path.exists(text_encoder_sub_dir) and base_model_name_or_path is None:
        config = LoraConfig.from_pretrained(text_encoder_sub_dir)
        base_model_name_or_path = config.base_model_name_or_path

    if base_model_name_or_path is None:
        raise ValueError("Please specify the base model name or path")

    pipe = StableDiffusionPipeline.from_pretrained(base_model_name_or_path, torch_dtype=dtype).to(device)
    pipe.unet = PeftModel.from_pretrained(pipe.unet, unet_sub_dir, adapter_name=adapter_name)

    if os.path.exists(text_encoder_sub_dir):
        pipe.text_encoder = PeftModel.from_pretrained(
            pipe.text_encoder, text_encoder_sub_dir, adapter_name=adapter_name
        )

    if dtype in (torch.float16, torch.bfloat16):
        pipe.unet.half()
        pipe.text_encoder.half()

    pipe.to(device)
    return pipe


def convert_motion_module(original_state_dict):
    converted_state_dict = {}
    for k, v in original_state_dict.items():
        if "pos_encoder" in k:
            continue

        else:
            temp = ".".join(k.split('.')[1:])
            #print(temp)
            converted_state_dict[
                temp.replace(".norms.0", ".norm1")
                .replace(".norms.1", ".norm2")
                .replace(".ff_norm", ".norm3")
                .replace(".attention_blocks.0", ".attn1")
                .replace(".attention_blocks.1", ".attn2")
                .replace(".temporal_transformer", "")
            ] = v

    return converted_state_dict


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class InvDynamics(nn.Module):
    def __init__(self, action_dim=4):
        super(InvDynamics, self).__init__()
        self.action_dim = action_dim

        self.visual_model, self.embd_size, self.model_transforms, self.model_info = model_utils.load_model(model_utils.VC1_BASE_NAME)

        self.inv_model = nn.Linear(2*self.embd_size, self.action_dim)

    def forward(self, obs):
        # Get Action
        b, n, c, h, w = obs.shape
        obs = torch.reshape(obs, [b*n, c, h, w])
        obs = self.model_transforms(obs)
        embed = self.visual_model(obs)
        reshaped = torch.reshape(embed, [b, -1])
        return self.inv_model(reshaped)

    def calculate_loss(self, obs, action):
        pred_action = self.forward(obs)
        mse = F.mse_loss(pred_action, action)
        return mse

    @torch.no_grad()
    def calculate_test_loss(self, obs, action):
        pred_action = self.forward(obs)
        mse = F.mse_loss(pred_action, action)
        return mse


class AnimateDiff(nn.Module):
    def __init__(self, device, fp16=True):
        super().__init__()

        self.device = device

        print(f'[INFO] loading animatediff...')

        self.precision_t = torch.float16 if fp16 else torch.float32

        model_id = "sd-legacy/stable-diffusion-v1-5"
        controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
        lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=self.precision_t).to(device)
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.precision_t).to(device)

        pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(model_id, motion_adapter=adapter, controlnet=controlnet, torch_dtype=self.precision_t).to(device)
        pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

        pipe.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        # enable memory savings
        pipe.enable_vae_slicing()
        #pipe.enable_model_cpu_offload()

        self.pipe = pipe

        self.vae = pipe.vae
        self.encode_prompt = pipe.encode_prompt
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        print(f'[INFO] loaded animatediff!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            negative_prompt,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @torch.no_grad()
    def encode_imgs(self, rgb_input):
        # rgb_input: [B, 3, H, W]

        rgb_input = F.interpolate(rgb_input, (512, 512), mode='bilinear', align_corners=False)

        normalized = self.normalize(rgb_input)

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_videotadpole_reward(self, text_embeddings, latents, t, alignment_scale=2000, recon_scale=200, noise=None):

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latent_model_input = torch.cat([batch_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        source_noise = noise.permute(1,0,2,3).unsqueeze(0)

        alignment_pred = ((noise_pred_text - noise_pred_uncond)**2).mean([0,1,3,4])
        pos_natural_pred = ((noise_pred_text - source_noise)**2).mean([0,1,3,4])
        uncond_natural_pred = ((noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
        recon_pred = uncond_natural_pred - pos_natural_pred

        rewards = symlog(alignment_scale*alignment_pred) + symlog(recon_scale*recon_pred)
        return rewards


class AnimateDiffDreamBooth(nn.Module):
    def __init__(self, device, domain='humanoid', fp16=True):
        super().__init__()

        self.device = device

        print(f'[INFO] loading animatediff dreambooth...')

        self.precision_t = torch.float16 if fp16 else torch.float32

        assert domain in ['humanoid', 'dog', 'mw']
        model_id = "sd-legacy/stable-diffusion-v1-5"
        path_to_saved_model = f'./checkpoints/dreambooth/{domain}_lora'
        sd_db_pipe = get_lora_sd_pipeline(Path(path_to_saved_model), dtype=self.precision_t, adapter_name="subject")

        motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
        controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
        lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"

        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=self.precision_t).to(device)
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.precision_t).to(device)

        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        pipe = AnimateDiffSparseControlNetPipeline(
            vae=sd_db_pipe.vae,
            text_encoder=sd_db_pipe.text_encoder,
            tokenizer=sd_db_pipe.tokenizer,
            unet=sd_db_pipe.unet.merge_and_unload(),
            motion_adapter=motion_adapter,
            controlnet=controlnet,
            scheduler=scheduler
        ).to(device)
        pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

        # enable memory savings
        pipe.enable_vae_slicing()

        self.pipe = pipe

        self.vae = pipe.vae
        self.encode_prompt = pipe.encode_prompt
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        print(f'[INFO] loaded animatediff dreambooth!')

    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            negative_prompt,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @torch.no_grad()
    def encode_imgs(self, rgb_input):
        # rgb_input: [B, 3, H, W]

        rgb_input = F.interpolate(rgb_input, (512, 512), mode='bilinear', align_corners=False)

        normalized = self.normalize(rgb_input)

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_videotadpole_reward(self, text_embeddings, latents, t, alignment_scale=2000, recon_scale=200, noise=None):

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latent_model_input = torch.cat([batch_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        source_noise = noise.permute(1,0,2,3).unsqueeze(0)

        alignment_pred = ((noise_pred_text - noise_pred_uncond)**2).mean([0,1,3,4])
        pos_natural_pred = ((noise_pred_text - source_noise)**2).mean([0,1,3,4])
        uncond_natural_pred = ((noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
        recon_pred = uncond_natural_pred - pos_natural_pred

        rewards = symlog(alignment_scale*alignment_pred) + symlog(recon_scale*recon_pred)
        return rewards


class AnimateDiffDirectFT(nn.Module):
    def __init__(self, device, fp16=True, domain='mw'):
        super().__init__()

        self.device = device

        self.precision_t = torch.float16 if fp16 else torch.float32

        assert domain in ['humanoid', 'dog', 'mw']
        ckpt_path = f'./checkpoints/animatediff_finetuned/{domain}_finetuned.ckpt'

        print(f'[INFO] loading animatediff finetuned...')

        adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-3", torch_dtype=self.precision_t).to(device)

        unet_checkpoint_path = torch.load(ckpt_path, map_location="cpu")
        state_dict = convert_motion_module(unet_checkpoint_path["state_dict"])
        m, u  = adapter.load_state_dict(state_dict, strict=False)

        # load SD 1.5 based finetuned model
        model_id = "sd-legacy/stable-diffusion-v1-5"
        lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"
        controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.precision_t).to(device)
        pipe = AnimateDiffSparseControlNetPipeline.from_pretrained(model_id, motion_adapter=adapter, controlnet=controlnet, torch_dtype=self.precision_t).to(device)
        pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

        pipe.scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        # enable memory savings
        pipe.enable_vae_slicing()

        self.pipe = pipe

        self.vae = pipe.vae
        self.encode_prompt = pipe.encode_prompt
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        print(f'[INFO] loaded animatediff finetuned!')
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            negative_prompt,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @torch.no_grad()
    def encode_imgs(self, rgb_input):
        # rgb_input: [B, 3, H, W]

        rgb_input = F.interpolate(rgb_input, (512, 512), mode='bilinear', align_corners=False)

        normalized = self.normalize(rgb_input)

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_videotadpole_reward(self, text_embeddings, latents, t, alignment_scale=2000, recon_scale=200, noise=None):

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latent_model_input = torch.cat([batch_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        source_noise = noise.permute(1,0,2,3).unsqueeze(0)

        alignment_pred = ((noise_pred_text - noise_pred_uncond)**2).mean([0,1,3,4])
        pos_natural_pred = ((noise_pred_text - source_noise)**2).mean([0,1,3,4])
        uncond_natural_pred = ((noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
        recon_pred = uncond_natural_pred - pos_natural_pred

        rewards = symlog(alignment_scale*alignment_pred) + symlog(recon_scale*recon_pred)
        return rewards


class AnimateDiffProbAdaptation(nn.Module):
    def __init__(self, device, domain='humanoid', fp16=True, use_suboptimal=False):
        super().__init__()

        self.device = device
        assert domain in ['humanoid', 'dog', 'mw']

        print(f'[INFO] loading animatediff...')

        self.precision_t = torch.float16 if fp16 else torch.float32

        model_id = "sd-legacy/stable-diffusion-v1-5"
        motion_adapter_id = "guoyww/animatediff-motion-adapter-v1-5-3"
        controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
        lora_adapter_id = "guoyww/animatediff-motion-lora-v1-5-3"

        sd_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=self.precision_t).to(device)
        motion_adapter = MotionAdapter.from_pretrained(motion_adapter_id, torch_dtype=self.precision_t).to(device)
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.precision_t).to(device)

        scheduler = DDIMScheduler.from_pretrained(
            model_id,
            subfolder="scheduler",
            clip_sample=False,
            timestep_spacing="linspace",
            beta_schedule="linear",
            steps_offset=1,
        )

        pipe = AnimateDiffSparseControlNetPipeline(
            vae=sd_pipe.vae,
            text_encoder=sd_pipe.text_encoder,
            tokenizer=sd_pipe.tokenizer,
            unet=sd_pipe.unet,
            motion_adapter=motion_adapter,
            controlnet=controlnet,
            scheduler=scheduler
        ).to(device)
        pipe.load_lora_weights(lora_adapter_id, adapter_name="motion_lora")

        # enable memory savings
        pipe.enable_vae_slicing()

        # Create model

        self.pipe = pipe

        self.vae = pipe.vae
        self.encode_prompt = pipe.encode_prompt
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        print(f'[INFO] loaded animatediff!')

        print(f'[INFO] loading in-domain checkpoint...')
        self.avdc_trainer = self.load_avdc(domain, use_suboptimal)
        print(f'[INFO] loaded in-domain checkpoint!')
    

    def load_avdc(self, domain, use_suboptimal):
        avdc_ckpt_number = 24

        avdc_pretrained_model = "openai/clip-vit-base-patch32"
        avdc_tokenizer = CLIPTokenizer.from_pretrained(avdc_pretrained_model)
        avdc_text_encoder = CLIPTextModel.from_pretrained(avdc_pretrained_model)
        avdc_unet = UnetMW_SD()
        avdc = GoalGaussianDiffusion_SD(
            channels=4*(9-1),
            model=avdc_unet,
            image_size=(64, 64),
            timesteps=1000,
            sampling_timesteps=100,
            loss_type='l2',
            objective='pred_noise',
            beta_schedule = 'linear',
            min_snr_loss_weight = True,
        )

        train_set = valid_set = [None] # dummy

        avdc_trainer = Trainer(
            diffusion_model=avdc,
            tokenizer=avdc_tokenizer, 
            text_encoder=avdc_text_encoder,
            train_set=train_set,
            valid_set=valid_set,
            train_lr=1e-4,
            train_num_steps =60000,
            save_and_sample_every =2500,
            ema_update_every = 10,
            ema_decay = 0.999,
            train_batch_size = 16,
            valid_batch_size = 32,
            gradient_accumulate_every = 1,
            num_samples = 1, 
            results_folder =f'./checkpoints/in_domain/{domain}{"_suboptimal" if use_suboptimal else ""}',
            fp16 =True,
            amp=True,
        )

        # load checkpoint for avdc
        avdc_trainer.load(avdc_ckpt_number)

        return avdc_trainer
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            negative_prompt,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @torch.no_grad()
    def encode_imgs(self, rgb_input):
        # rgb_input: [B, 3, H, W]

        rgb_input = F.interpolate(rgb_input, (512, 512), mode='bilinear', align_corners=False)

        normalized = self.normalize(rgb_input)

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_videotadpole_reward(self, text_embeddings, latents, t, prior_strength, alignment_scale=2000, recon_scale=200, noise=None, avdc_text_embeddings=None, avdc_cond=None, inverted_probadap=False):

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latent_model_input = torch.cat([batch_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        avdc_latent_input = rearrange(batch_input.clone(), 'b c f h w -> b (f c) h w')
        avdc_noise_pred_text = self.avdc_trainer.model.model(
            torch.cat([avdc_latent_input, avdc_cond], dim=1).float(), 
            torch.tensor([t], device=self.device), 
            avdc_text_embeddings).half()

        avdc_noise_pred_uncond = self.avdc_trainer.model.model(
            torch.cat([avdc_latent_input, avdc_cond], dim=1).float(), 
            torch.tensor([t], device=self.device), 
            avdc_text_embeddings * 0.0).half()
                
        avdc_noise_pred_text = rearrange(avdc_noise_pred_text, 'b (f c) h w -> b c f h w', c=4)
        avdc_noise_pred_uncond = rearrange(avdc_noise_pred_uncond, 'b (f c) h w -> b c f h w', c=4)


        source_noise = noise.permute(1,0,2,3).unsqueeze(0)

        
        if inverted_probadap:
            adapted_noise_pred_text = noise_pred_text + prior_strength * avdc_noise_pred_text
            alignment_pred = ((adapted_noise_pred_text - noise_pred_uncond)**2).mean([0,1,3,4])
            pos_natural_pred = ((adapted_noise_pred_text - source_noise)**2).mean([0,1,3,4])
            uncond_natural_pred = ((noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
            recon_pred = uncond_natural_pred - pos_natural_pred
        else:
            adapted_noise_pred_text = avdc_noise_pred_text + prior_strength * noise_pred_text
            alignment_pred = ((adapted_noise_pred_text - avdc_noise_pred_uncond)**2).mean([0,1,3,4])
            pos_natural_pred = ((adapted_noise_pred_text - source_noise)**2).mean([0,1,3,4])
            uncond_natural_pred = ((avdc_noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
            recon_pred = uncond_natural_pred - pos_natural_pred

        rewards = symlog(alignment_scale*alignment_pred) + symlog(recon_scale*recon_pred)
        return rewards


class AnimateLCM(nn.Module):
    def __init__(self, device, fp16=True):
        super().__init__()

        self.device = device

        print(f'[INFO] loading animatelcm...')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=self.precision_t)
        
        controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.precision_t).to(device)
        pipe = AnimateDiffSparseControlNetPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, controlnet=controlnet, torch_dtype=self.precision_t).to(device)
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

        pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
        pipe.set_adapters(["lcm-lora"], [0.8])

        pipe.enable_vae_slicing()

        self.pipe = pipe
    
        self.vae = pipe.vae
        self.encode_prompt = pipe.encode_prompt
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        print(f'[INFO] loaded animatelcm!')
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            negative_prompt,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @torch.no_grad()
    def encode_imgs(self, rgb_input):
        # rgb_input: [B, 3, H, W]

        rgb_input = F.interpolate(rgb_input, (512, 512), mode='bilinear', align_corners=False)

        normalized = self.normalize(rgb_input)

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_videotadpole_reward(self, text_embeddings, latents, t, alignment_scale=2000, recon_scale=200, noise=None):

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latent_model_input = torch.cat([batch_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        source_noise = noise.permute(1,0,2,3).unsqueeze(0)

        alignment_pred = ((noise_pred_text - noise_pred_uncond)**2).mean([0,1,3,4])
        pos_natural_pred = ((noise_pred_text - source_noise)**2).mean([0,1,3,4])
        uncond_natural_pred = ((noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
        recon_pred = uncond_natural_pred - pos_natural_pred

        rewards = symlog(alignment_scale*alignment_pred) + symlog(recon_scale*recon_pred)
        return rewards


class AnimateLCMProbAdaptation(nn.Module):
    def __init__(self, device, domain='humanoid', fp16=True, use_suboptimal=False):
        super().__init__()

        self.device = device

        print(f'[INFO] loading animatelcm...')

        self.precision_t = torch.float16 if fp16 else torch.float32

        # Create model
        adapter = MotionAdapter.from_pretrained("wangfuyun/AnimateLCM", torch_dtype=self.precision_t)
        
        controlnet_id = "guoyww/animatediff-sparsectrl-rgb"
        controlnet = SparseControlNetModel.from_pretrained(controlnet_id, torch_dtype=self.precision_t).to(device)
        pipe = AnimateDiffSparseControlNetPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, controlnet=controlnet, torch_dtype=self.precision_t).to(device)

        # probably not needed
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config, beta_schedule="linear")

        pipe.load_lora_weights("wangfuyun/AnimateLCM", weight_name="AnimateLCM_sd15_t2v_lora.safetensors", adapter_name="lcm-lora")
        pipe.set_adapters(["lcm-lora"], [0.8])

        pipe.enable_vae_slicing()

        self.pipe = pipe

        self.vae = pipe.vae
        self.encode_prompt = pipe.encode_prompt
        self.unet = pipe.unet
        self.scheduler = pipe.scheduler

        print(f'[INFO] loaded animatelcm!')

        print(f'[INFO] loading in-domain checkpoint...')
        self.avdc_trainer = self.load_avdc(domain, use_suboptimal)
        print(f'[INFO] loaded in-domain checkpoint!')
    

    def load_avdc(self, domain, use_suboptimal):
        avdc_ckpt_number = 24

        avdc_pretrained_model = "openai/clip-vit-base-patch32"
        avdc_tokenizer = CLIPTokenizer.from_pretrained(avdc_pretrained_model)
        avdc_text_encoder = CLIPTextModel.from_pretrained(avdc_pretrained_model)
        avdc_unet = UnetMW_SD()
        avdc = GoalGaussianDiffusion_SD(
            channels=4*(9-1),
            model=avdc_unet,
            image_size=(64, 64),
            timesteps=1000,
            sampling_timesteps=100,
            loss_type='l2',
            objective='pred_noise',
            beta_schedule = 'linear',
            min_snr_loss_weight = True,
        )

        train_set = valid_set = [None] # dummy

        avdc_trainer = Trainer(
            diffusion_model=avdc,
            tokenizer=avdc_tokenizer, 
            text_encoder=avdc_text_encoder,
            train_set=train_set,
            valid_set=valid_set,
            train_lr=1e-4,
            train_num_steps =60000,
            save_and_sample_every =2500,
            ema_update_every = 10,
            ema_decay = 0.999,
            train_batch_size = 16,
            valid_batch_size = 32,
            gradient_accumulate_every = 1,
            num_samples = 1, 
            results_folder =f'./checkpoints/in_domain/{domain}{"_suboptimal" if use_suboptimal else ""}',
            fp16 =True,
            amp=True,
        )

        # load checkpoint for avdc
        avdc_trainer.load(avdc_ckpt_number)

        return avdc_trainer
    
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            self.device,
            1,
            True,
            negative_prompt,
        )
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])
        return prompt_embeds

    @torch.no_grad()
    def normalize(self, images):
        """
        Normalize an image array to [-1,1].
        """
        return 2.0 * images - 1.0

    @torch.no_grad()
    def encode_imgs(self, rgb_input):
        # rgb_input: [B, 3, H, W]

        rgb_input = F.interpolate(rgb_input, (512, 512), mode='bilinear', align_corners=False)

        normalized = self.normalize(rgb_input)

        latents = self.vae.encode(normalized).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        return latents

    @torch.no_grad()
    def get_videotadpole_reward(self, text_embeddings, latents, t, prior_strength, alignment_scale=2000, recon_scale=200, noise=None, avdc_text_embeddings=None, avdc_cond=None, inverted_probadap=False):

        if noise is None:
            noise = torch.randn_like(latents)

        latents_noisy = self.scheduler.add_noise(latents, noise, t)

        batch_input = latents_noisy.permute(1,0,2,3).unsqueeze(0)
        latent_model_input = torch.cat([batch_input] * 2)

        noise_pred = self.unet(
            latent_model_input,
            t,
            encoder_hidden_states=text_embeddings,
        ).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)

        avdc_latent_input = rearrange(batch_input.clone(), 'b c f h w -> b (f c) h w')
        avdc_noise_pred_text = self.avdc_trainer.model.model(
            torch.cat([avdc_latent_input, avdc_cond], dim=1).float(), 
            torch.tensor([t], device=self.device), 
            avdc_text_embeddings).half()

        avdc_noise_pred_uncond = self.avdc_trainer.model.model(
            torch.cat([avdc_latent_input, avdc_cond], dim=1).float(), 
            torch.tensor([t], device=self.device), 
            avdc_text_embeddings * 0.0).half()
                
        avdc_noise_pred_text = rearrange(avdc_noise_pred_text, 'b (f c) h w -> b c f h w', c=4)
        avdc_noise_pred_uncond = rearrange(avdc_noise_pred_uncond, 'b (f c) h w -> b c f h w', c=4)


        source_noise = noise.permute(1,0,2,3).unsqueeze(0)


        if inverted_probadap:
            adapted_noise_pred_text = noise_pred_text + prior_strength * avdc_noise_pred_text
            alignment_pred = ((adapted_noise_pred_text - noise_pred_uncond)**2).mean([0,1,3,4])
            pos_natural_pred = ((adapted_noise_pred_text - source_noise)**2).mean([0,1,3,4])
            uncond_natural_pred = ((noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
            recon_pred = uncond_natural_pred - pos_natural_pred
        else:
            adapted_noise_pred_text = avdc_noise_pred_text + prior_strength * noise_pred_text
            alignment_pred = ((adapted_noise_pred_text - avdc_noise_pred_uncond)**2).mean([0,1,3,4])
            pos_natural_pred = ((adapted_noise_pred_text - source_noise)**2).mean([0,1,3,4])
            uncond_natural_pred = ((avdc_noise_pred_uncond - source_noise)**2).mean([0,1,3,4])
            recon_pred = uncond_natural_pred - pos_natural_pred

        rewards = symlog(alignment_scale*alignment_pred) + symlog(recon_scale*recon_pred)
        return rewards


@torch.no_grad()
def probadap_sampling(
    animatediff_pipe: AnimateDiffSparseControlNetPipeline,
    avdc_trainer: Trainer,
    animatediff_prompt: Optional[Union[str, List[str]]] = None,
    avdc_prompt: Optional[Union[str, List[str]]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_frames: int = 16,
    num_inference_steps: int = 25,
    guidance_scale: float = 7.5,
    negative_prompt: Optional[Union[str, List[str]]] = None,
    num_videos_per_prompt: int = 1,
    eta: float = 0.0,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    prompt_embeds: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.Tensor] = None,
    ip_adapter_image: Optional[PipelineImageInput] = None,
    ip_adapter_image_embeds: Optional[List[torch.Tensor]] = None,
    conditioning_frames: Optional[List[PipelineImageInput]] = None,
    conditioning_latents: Optional[List[PipelineImageInput]] = None,
    is_conditioning_animatediff: bool = False,
    output_type: str = "pil",
    return_dict: bool = True,
    cross_attention_kwargs: Optional[Dict[str, Any]] = None,
    controlnet_conditioning_scale: Union[float, List[float]] = 1.0,
    controlnet_frame_indices: List[int] = [0],
    guess_mode: bool = False,
    clip_skip: Optional[int] = None,
    callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    is_inverted: bool = False,

    prior_strength: float = 0.,

    **kwargs,
):
    controlnet = animatediff_pipe.controlnet
    callback_steps = None

    # 0. Default height and width to unet
    height = height or animatediff_pipe.unet.config.sample_size * animatediff_pipe.vae_scale_factor
    width = width or animatediff_pipe.unet.config.sample_size * animatediff_pipe.vae_scale_factor
    num_videos_per_prompt = 1

    # 1. Check inputs. Raise error if not correct
    animatediff_pipe.check_inputs(
        prompt=animatediff_prompt,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        ip_adapter_image=ip_adapter_image,
        ip_adapter_image_embeds=ip_adapter_image_embeds,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        image=conditioning_frames,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
    )

    animatediff_pipe._guidance_scale = guidance_scale
    animatediff_pipe._clip_skip = clip_skip
    animatediff_pipe._cross_attention_kwargs = cross_attention_kwargs

    # 2. Define call parameters
    if animatediff_prompt is not None and isinstance(animatediff_prompt, str):
        batch_size = 1
    elif animatediff_prompt is not None and isinstance(animatediff_prompt, list):
        batch_size = len(animatediff_prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = animatediff_pipe._execution_device

    global_pool_conditions = (
        controlnet.config.global_pool_conditions
        if isinstance(controlnet, SparseControlNetModel)
        else controlnet.nets[0].config.global_pool_conditions
    )
    guess_mode = guess_mode or global_pool_conditions

    # 3. Encode input prompt
    text_encoder_lora_scale = (
        animatediff_pipe.cross_attention_kwargs.get("scale", None) if animatediff_pipe.cross_attention_kwargs is not None else None
    )
    prompt_embeds, negative_prompt_embeds = animatediff_pipe.encode_prompt(
        animatediff_prompt,
        device,
        num_videos_per_prompt,
        animatediff_pipe.do_classifier_free_guidance,
        negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        lora_scale=text_encoder_lora_scale,
        clip_skip=animatediff_pipe.clip_skip,
    )
    avdc_prompt = avdc_prompt if isinstance(avdc_prompt, list) else [avdc_prompt] * batch_size
    avdc_prompt_embeds = avdc_trainer.encode_batch_text(avdc_prompt)
    # For classifier free guidance, we need to do two forward passes.
    # Here we concatenate the unconditional and text embeddings into a single batch
    # to avoid doing two forward passes
    if animatediff_pipe.do_classifier_free_guidance:
        prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

    # 4. Prepare IP-Adapter embeddings
    if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
        image_embeds = animatediff_pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image,
            ip_adapter_image_embeds,
            device,
            batch_size * num_videos_per_prompt,
            animatediff_pipe.do_classifier_free_guidance,
        )

    # 5. Prepare controlnet conditioning
    if conditioning_frames is not None:

        conditioning_frames = animatediff_pipe.prepare_image(conditioning_frames, width, height, device, controlnet.dtype)
        controlnet_cond, controlnet_cond_mask = animatediff_pipe.prepare_sparse_control_conditioning(
            conditioning_frames, num_frames, controlnet_frame_indices, device, controlnet.dtype
        )
        avdc_cond = rearrange(conditioning_frames.clone(), 'b c f h w -> b (f c) h w')
    if conditioning_latents is not None:
        conditioning_latents = conditioning_latents.to(device)

    # 6. Prepare timesteps
    animatediff_pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = animatediff_pipe.scheduler.timesteps

    # 7. Prepare latent variables
    num_channels_latents = animatediff_pipe.unet.config.in_channels
    latents = animatediff_pipe.prepare_latents(
        batch_size * num_videos_per_prompt,
        num_channels_latents,
        num_frames,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    

    # 8. Prepare extra step kwargs.
    extra_step_kwargs = animatediff_pipe.prepare_extra_step_kwargs(generator, eta)

    # 9. Add image embeds for IP-Adapter
    added_cond_kwargs = (
        {"image_embeds": image_embeds}
        if ip_adapter_image is not None or ip_adapter_image_embeds is not None
        else None
    )

    num_free_init_iters = animatediff_pipe._free_init_num_iters if animatediff_pipe.free_init_enabled else 1
    for free_init_iter in range(num_free_init_iters):
        if animatediff_pipe.free_init_enabled:
            latents, timesteps = animatediff_pipe._apply_free_init(
                latents, free_init_iter, num_inference_steps, device, latents.dtype, generator
            )

        animatediff_pipe._num_timesteps = len(timesteps)
        num_warmup_steps = len(timesteps) - num_inference_steps * animatediff_pipe.scheduler.order

        # 10. Denoising loop
        with animatediff_pipe.progress_bar(total=animatediff_pipe._num_timesteps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                animatediff_latent_input = torch.cat([latents] * 2) if animatediff_pipe.do_classifier_free_guidance else latents
                animatediff_latent_input = animatediff_pipe.scheduler.scale_model_input(animatediff_latent_input, t)

                down_block_res_samples = mid_block_res_sample = None
                if is_conditioning_animatediff and controlnet is not None:
                    avdc_latent_input = rearrange(latents[:, :, 1:].clone(), 'b c f h w -> b (f c) h w')
                    if guess_mode and animatediff_pipe.do_classifier_free_guidance:
                        # Infer SparseControlNetModel only for the conditional batch.
                        control_model_input = latents
                        control_model_input = animatediff_pipe.scheduler.scale_model_input(control_model_input, t)
                        controlnet_prompt_embeds = prompt_embeds.chunk(2)[1]
                    else:
                        control_model_input = animatediff_latent_input
                        controlnet_prompt_embeds = prompt_embeds

                    down_block_res_samples, mid_block_res_sample = controlnet(
                        control_model_input,
                        t,
                        encoder_hidden_states=controlnet_prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_mask=controlnet_cond_mask,
                        conditioning_scale=controlnet_conditioning_scale,
                        guess_mode=guess_mode,
                        return_dict=False,
                    )
                else:
                    avdc_latent_input = rearrange(latents.clone(), 'b c f h w -> b (f c) h w')
                
                # predict the noise residual
                animatediff_noise_pred = animatediff_pipe.unet(
                    animatediff_latent_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    added_cond_kwargs=added_cond_kwargs,
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                ).sample

                avdc_noise_pred_text = avdc_trainer.model.model(
                    torch.cat([avdc_latent_input, avdc_cond], dim=1).float(), 
                    torch.tensor([t], device=device), 
                    avdc_prompt_embeds).half()
                
                avdc_noise_pred_uncond = avdc_trainer.model.model(
                    torch.cat([avdc_latent_input, avdc_cond], dim=1).float(), 
                    torch.tensor([t], device=device), 
                    avdc_prompt_embeds * 0.0).half()
                
                avdc_noise_pred_text = rearrange(avdc_noise_pred_text, 'b (f c) h w -> b c f h w', c=4)
                avdc_noise_pred_uncond = rearrange(avdc_noise_pred_uncond, 'b (f c) h w -> b c f h w', c=4)

                # perform guidance
                if animatediff_pipe.do_classifier_free_guidance:
                    animatediff_noise_pred_uncond, animatediff_noise_pred_text = animatediff_noise_pred.chunk(2)
                    animatediff_noise_pred_cfg = animatediff_noise_pred_uncond + guidance_scale * (animatediff_noise_pred_text - animatediff_noise_pred_uncond)

                    # compute the previous noisy sample x_t -> x_t-1
                    # latents = animatediff_pipe.scheduler.step(animatediff_noise_pred_cfg, t, latents, **extra_step_kwargs).prev_sample
                    if is_conditioning_animatediff:
                        if is_inverted:
                            adapted_noise_pred_text = animatediff_noise_pred_text[:, :, 1:] + prior_strength * avdc_noise_pred_text
                            adapted_noise_pred = animatediff_noise_pred_uncond[:, :, 1:] + guidance_scale * (adapted_noise_pred_text - animatediff_noise_pred_uncond[:, :, 1:])
                            adapted_noise_pred = torch.cat([animatediff_noise_pred_cfg[:, :, :1], adapted_noise_pred], dim=2)
                        else:
                            adapted_noise_pred_text = avdc_noise_pred_text + prior_strength * animatediff_noise_pred_text[:, :, 1:]
                            adapted_noise_pred = avdc_noise_pred_uncond + guidance_scale * (adapted_noise_pred_text - avdc_noise_pred_uncond)
                            adapted_noise_pred = torch.cat([animatediff_noise_pred_cfg[:, :, :1], adapted_noise_pred], dim=2)
                    else:
                        if is_inverted:
                            adapted_noise_pred_text = animatediff_noise_pred_text + prior_strength * avdc_noise_pred_text
                            adapted_noise_pred = animatediff_noise_pred_uncond + guidance_scale * (adapted_noise_pred_text - animatediff_noise_pred_uncond)
                        else:
                            adapted_noise_pred_text = avdc_noise_pred_text + prior_strength * animatediff_noise_pred_text
                            adapted_noise_pred = avdc_noise_pred_uncond + guidance_scale * (adapted_noise_pred_text - avdc_noise_pred_uncond)
                    latents = animatediff_pipe.scheduler.step(adapted_noise_pred, t, latents, **extra_step_kwargs).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(animatediff_pipe, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % animatediff_pipe.scheduler.order == 0):
                    progress_bar.update()

    # 11. Post processing
    # if is_conditioning_animatediff:
    #     latents = latents[:, :, 1:]  # get rid of the conditioning frame
    if output_type == "latent":
        video = latents
    else:
        video_tensor = animatediff_pipe.decode_latents(latents)
        video = animatediff_pipe.video_processor.postprocess_video(video=video_tensor, output_type=output_type)

    # 12. Offload all models
    animatediff_pipe.maybe_free_model_hooks()

    if not return_dict:
        return (video,)

    return AnimateDiffPipelineOutput(frames=video)