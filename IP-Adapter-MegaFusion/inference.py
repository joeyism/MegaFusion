import os
import torch
import argparse
import numpy as np
from PIL import Image
from typing import Optional, List
from ip_adapter import IPAdapter
from torchvision import transforms
from diffusers import AutoencoderKL, DDIMScheduler

from ip_adapter.pipeline_stable_diffusion import StableDiffusionPipeline

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./test/", type=str)
    parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-v1-5/', type=str)
    parser.add_argument('--image_encoder', default='./ckpt/image_encoder/', type=str)
    parser.add_argument('--ip_ckpt', default='./ckpt/ip-adapter_sd15.bin', type=str)
    parser.add_argument('--ref_image', default='./ref_image.png', type=str)
    parser.add_argument('--mixed_precision', default='fp16', type=str)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--stage_resolutions', default=[512, 768, 1024], type=int, nargs='+')
    parser.add_argument('--stage_steps', default=[40, 5, 5], type=int, nargs='+')

    return parser

def test(
    pretrained_model_path: str,
    image_encoder_path: str,
    ip_ckpt: str,
    ref_image: str,
    logdir: str,
    num_inference_steps,
    mixed_precision: Optional[str] = "fp16",
    stage_resolutions: Optional[List[int]] = [512, 768, 1024],
    stage_steps: Optional[List[int]] = [40, 5, 5]
):
    if not os.path.exists(logdir):
        os.mkdir(logdir)

    device = "cuda:0"

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    vae = AutoencoderKL.from_pretrained(os.path.join(pretrained_model_path, 'vae')).to(dtype=weight_dtype, device=device)

    pipe = StableDiffusionPipeline.from_pretrained(
        pretrained_model_path,
        torch_dtype=weight_dtype,
        scheduler=noise_scheduler,
        vae=vae,
        feature_extractor=None,
        safety_checker=None,
        low_cpu_mem_usage=False
    )

    # load ip-adapter
    ip_model = IPAdapter(pipe, image_encoder_path, ip_ckpt, device)
    
    ref_image = Image.open(ref_image).resize((256, 256))
    noise_scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps_stage_1, timesteps_stage_2, timesteps_stage_3 = noise_scheduler.timesteps[:stage_steps[0]], noise_scheduler.timesteps[stage_steps[0]:stage_steps[0] + stage_steps[1]], noise_scheduler.timesteps[stage_steps[0] + stage_steps[1]:]

    def encode_image(image):
        tensor = transforms.ToTensor()(image).float() * 2. - 1.
        tensor = tensor.unsqueeze(0).to(device, dtype=weight_dtype)
        latents = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
        
        return latents
    
    with torch.no_grad():
        # Stage 1, 40 steps
        x_0_predict = ip_model.generate(pil_image=ref_image, num_samples=1, width=stage_resolutions[0], height=stage_resolutions[0], num_inference_steps=num_inference_steps, stage_timesteps=timesteps_stage_1, seed=42)
        
        # Stage 2, 5 steps
        x_0_predict = x_0_predict[0].resize((stage_resolutions[1], stage_resolutions[1]), Image.Resampling.BICUBIC)
        latents = encode_image(x_0_predict)
        noise = torch.randn_like(latents)
        latents_MR = noise_scheduler.add_noise(latents, noise, timesteps_stage_2[0])
        x_0_predict = ip_model.generate(pil_image=ref_image, latent=latents_MR, num_samples=1, width=stage_resolutions[1], height=stage_resolutions[1], num_inference_steps=num_inference_steps, stage_timesteps=timesteps_stage_2, seed=42)

        # Stage 3, 5 steps
        x_0_predict = x_0_predict[0].resize((stage_resolutions[2], stage_resolutions[2]), Image.Resampling.BICUBIC)
        latents = encode_image(x_0_predict)
        noise = torch.randn_like(latents)
        latents_HR = noise_scheduler.add_noise(latents, noise, timesteps_stage_3[0])
        output_image = ip_model.generate(pil_image=ref_image, latent=latents_HR, num_samples=1, width=stage_resolutions[2], height=stage_resolutions[2], num_inference_steps=num_inference_steps, stage_timesteps=timesteps_stage_3, seed=42)

        output_image[0].save(os.path.join(logdir, "example_MegaFusion.png"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    test(
        pretrained_model_path=args.ckpt,
        image_encoder_path=args.image_encoder,
        ip_ckpt=args.ip_ckpt,
        ref_image=args.ref_image,
        logdir=args.logdir,
        num_inference_steps=args.num_inference_steps,
        stage_resolutions=args.stage_resolutions,
        mixed_precision=args.mixed_precision,
        stage_steps=args.stage_steps
    )