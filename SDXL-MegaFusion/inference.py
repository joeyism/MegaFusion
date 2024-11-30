import os
import random
import argparse
import torch
from PIL import Image
from typing import Optional, List
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger
from diffusers import AutoencoderKL
from diffusers.utils.import_utils import is_xformers_available
from transformers import AutoTokenizer, CLIPTextModel, CLIPTextModelWithProjection

from model.scheduling_ddim import DDIMScheduler
from model.unet_2d_condition import UNet2DConditionModel
from model.pipeline_xl import StableDiffusionXLPipeline

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-xl-base-1.0/', type=str)
    parser.add_argument('--prompt', default="An astronaut riding a horse on the moon.", type=str)    
    parser.add_argument('--mixed_precision', default='fp16', type=str)
    parser.add_argument('--guidance_scale', default=7.0, type=float)
    parser.add_argument('--if_reschedule', default=False, type=bool)
    parser.add_argument('--if_dilation', default=False, type=bool)
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--stage_resolutions', default=[1024, 2048], type=int, nargs='+') # [1024, 2048]
    parser.add_argument('--stage_steps', default=[40, 10], type=int, nargs='+') # [40, 10]
    # parser.add_argument('--stage_resolutions', default=[1024, 1536, 2048], type=int, nargs='+') # [1024, 2048]
    # parser.add_argument('--stage_steps', default=[40, 5, 5], type=int, nargs='+') # [40, 10]
    
    return parser

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.0,
    mixed_precision: Optional[str] = "fp16",   # "no", "fp16", "bf16"
    if_reschedule: Optional[bool] = False,
    if_dilation: Optional[bool] = False,
    stage_resolutions: Optional[List[int]] = [1024, 1536, 2048], # [1024, 2048]
    stage_steps: Optional[List[int]] = [40, 5, 5] # [40, 10]
):
    
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    accelerator = Accelerator(mixed_precision=mixed_precision)

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer", use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2", use_fast=False)
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder", from_flax=False)
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2", from_flax=False)
    # if_dilation: control whether to use dilation conv in unet's mid block or not
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet", from_flax=False, if_dilation=if_dilation)
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="sdxl-vae-fp16-fix", low_cpu_mem_usage=False, device_map=None, from_flax=False)
    # Haoning: pay attention here, some useless parameters are inevitably intialized.
    # And, you should use the fixed float16 version of vae to avoid the error.
    # vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae_1_0", low_cpu_mem_usage=False, device_map=None, from_flax=False)
    scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    scheduler_high = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler", reschedule_res=stage_resolutions[1])
    
    pipeline = StableDiffusionXLPipeline(
        vae=vae,
        text_encoder=text_encoder,
        text_encoder_2=text_encoder_2,
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        unet=unet,
        scheduler=scheduler,
        force_zeros_for_empty_prompt=True,
    )

    if is_xformers_available():
        try:
            pipeline.enable_xformers_memory_efficient_attention()
        except Exception as e:
            logger.warning(
                "Could not enable memory efficient attention. Make sure xformers is installed" f" correctly and a GPU is available: {e}"
            )
    unet, pipeline = accelerator.prepare(unet, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM")

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    unet.eval()
    
    sample_seed = random.randint(0, 100000)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(sample_seed)
    shape = (1, 4, stage_resolutions[0] // 8, stage_resolutions[0] // 8) # Init latents
    noise_latents = torch.randn(shape, generator=generator, device=accelerator.device, dtype=weight_dtype).to(accelerator.device)

    def encode_image(image):
        tensor = transforms.ToTensor()(image).float() * 2. - 1.
        tensor = tensor.unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        latents = vae.encode(tensor).latent_dist.sample() * vae.config.scaling_factor
        
        return latents

    def generate_image(prompt, height, width, latents, num_inference_steps, guidance_scale, timesteps):
        _, x_0_predict = pipeline(
            prompt=prompt,
            height=height,
            width=width,
            latents=latents,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            stage_timesteps=timesteps
        )
        return x_0_predict.images[0]

    # The following code is for the setting of our MegaFusion paper.
    scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    scheduler_high.set_timesteps(num_inference_steps, device=accelerator.device)

    timesteps_stage_1 = scheduler.timesteps[:stage_steps[0]]
    timesteps_stage_2 = scheduler_high.timesteps[stage_steps[0]:stage_steps[0] + stage_steps[1]]

    with torch.no_grad():
        # Stage 1, 40 steps
        x_0_predict = generate_image(prompt, stage_resolutions[0], stage_resolutions[0], noise_latents, num_inference_steps, guidance_scale, timesteps_stage_1)

        # Stage 2, 10 steps
        x_0_predict = x_0_predict.resize((stage_resolutions[1], stage_resolutions[1]), Image.Resampling.BICUBIC)
        latents = encode_image(x_0_predict)
        noise = torch.randn_like(latents)
        # if_reschedule controls whether to reschedule the scheduler or not
        pipeline.scheduler = scheduler_high if if_reschedule else scheduler
        latents_HR = pipeline.scheduler.add_noise(latents, noise, timesteps_stage_2[4]) if if_reschedule else pipeline.scheduler.add_noise(latents, noise, timesteps_stage_2[0])
        output_image = generate_image(prompt, stage_resolutions[1], stage_resolutions[1], latents_HR, num_inference_steps, guidance_scale, timesteps_stage_2)

        output_image.save(os.path.join(logdir, "generated.png"))


    # scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    # scheduler_mid = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler", reschedule_res=stage_resolutions[1])
    # scheduler_high = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler", reschedule_res=stage_resolutions[2])

    # # The following code is for more flexible setting of our MegaFusion.
    # scheduler.set_timesteps(num_inference_steps, device=accelerator.device)
    # scheduler_mid.set_timesteps(num_inference_steps, device=accelerator.device)
    # scheduler_high.set_timesteps(num_inference_steps, device=accelerator.device)

    # timesteps_stage_1 = scheduler.timesteps[:stage_steps[0]]
    # timesteps_stage_2 = scheduler_mid.timesteps[stage_steps[0]:stage_steps[0] + stage_steps[1]]
    # timesteps_stage_3 = scheduler_high.timesteps[stage_steps[0] + stage_steps[1]:]

    # with torch.no_grad():
    #     # Stage 1, 40 steps
    #     x_0_predict = generate_image(prompt, stage_resolutions[0], stage_resolutions[0], noise_latents, num_inference_steps, guidance_scale, timesteps_stage_1)

    #     # Stage 2, 5 steps
    #     x_0_predict = x_0_predict.resize((stage_resolutions[1], stage_resolutions[1]), Image.Resampling.BICUBIC)
    #     latents = encode_image(x_0_predict)
    #     noise = torch.randn_like(latents)
    #     # if_reschedule controls whether to reschedule the scheduler or not
    #     pipeline.scheduler = scheduler_mid if if_reschedule else scheduler
    #     latents_MR = pipeline.scheduler.add_noise(latents, noise, timesteps_stage_2[2]) if if_reschedule else pipeline.scheduler.add_noise(latents, noise, timesteps_stage_2[0])
    #     output_image = generate_image(prompt, stage_resolutions[1], stage_resolutions[1], latents_MR, num_inference_steps, guidance_scale, timesteps_stage_2)
        
    #     # Stage 3, 5 steps
    #     x_0_predict = x_0_predict.resize((stage_resolutions[2], stage_resolutions[2]), Image.Resampling.BICUBIC)
    #     latents = encode_image(x_0_predict)
    #     noise = torch.randn_like(latents)
    #     # if_reschedule controls whether to reschedule the scheduler or not
    #     pipeline.scheduler = scheduler_high if if_reschedule else scheduler
    #     latents_HR = pipeline.scheduler.add_noise(latents, noise, timesteps_stage_3[2]) if if_reschedule else pipeline.scheduler.add_noise(latents, noise, timesteps_stage_3[0])
    #     output_image = generate_image(prompt, stage_resolutions[2], stage_resolutions[2], latents_HR, num_inference_steps, guidance_scale, timesteps_stage_3)

    #     output_image.save(os.path.join(logdir, "generated.png"))


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    test(
        pretrained_model_path=args.ckpt,
        logdir=args.logdir,
        prompt=args.prompt,
        mixed_precision=args.mixed_precision,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        if_reschedule=args.if_reschedule,
        if_dilation=args.if_dilation,
        stage_resolutions=args.stage_resolutions,
        stage_steps=args.stage_steps
    )

# CUDA_VISIBLE_DEVICES=0 accelerate launch inference.py