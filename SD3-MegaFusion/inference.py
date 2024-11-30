import os
import random
import argparse
import torch
import torch.utils.data
import torch.utils.checkpoint
from PIL import Image
from typing import Optional, List
from torchvision import transforms
from accelerate import Accelerator
from accelerate.logging import get_logger

from diffusers.models.autoencoders import AutoencoderKL
from transformers import CLIPTextModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from model.transformer_sd3 import SD3Transformer2DModel
from model.pipeline import StableDiffusion3Pipeline
from model.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

logger = get_logger(__name__)

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./inference/", type=str)
    parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-3-medium-diffusers/', type=str)
    parser.add_argument('--prompt', default="A black cat is running in the rain.", type=str)    
    parser.add_argument('--num_inference_steps', default=28, type=int)
    parser.add_argument('--guidance_scale', default=7.0, type=float)
    parser.add_argument('--mixed_precision', default='fp16', type=str) # "fp16", "no"
    parser.add_argument('--stage_resolutions', default=[1024, 2048], type=int, nargs='+')
    parser.add_argument('--stage_steps', default=[20, 8], type=int, nargs='+')

    return parser

def test(
    pretrained_model_path: str,
    logdir: str,
    prompt: str,
    num_inference_steps: int = 28,
    guidance_scale: float = 7.0,
    mixed_precision: Optional[str] = "fp16",   # "fp16", "no"
    stage_resolutions: Optional[List[int]] = [1024, 2048],
    stage_steps: Optional[List[int]] = [20, 8]
):
    if not os.path.exists(logdir):
        os.mkdir(logdir)
        
    accelerator = Accelerator(mixed_precision=mixed_precision)

    transformer =  SD3Transformer2DModel.from_pretrained(pretrained_model_path, subfolder="transformer")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(pretrained_model_path, subfolder="text_encoder_2")
    tokenizer_2 = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer_2")
    text_encoder_3 = T5EncoderModel.from_pretrained(pretrained_model_path, subfolder="text_encoder_3")
    tokenizer_3 = T5TokenizerFast.from_pretrained(pretrained_model_path, subfolder="tokenizer_3")

    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    
    pipeline = StableDiffusion3Pipeline(
        transformer = transformer,
        scheduler = scheduler,
        vae = vae,
        text_encoder = text_encoder, 
        tokenizer = tokenizer,
        text_encoder_2 = text_encoder_2,
        tokenizer_2 = tokenizer_2,
        text_encoder_3 = text_encoder_3,
        tokenizer_3 = tokenizer_3,
    )
    
    # Removing the memory-intensive 4.7B parameter T5-XXL text encoder during inference can significantly decrease the memory requirements for SD3 with only a slight loss in performance.
    # pipeline = StableDiffusion3Pipeline(
    #     transformer = transformer,
    #     scheduler = scheduler,
    #     vae = vae,
    #     text_encoder = text_encoder, 
    #     tokenizer = tokenizer,
    #     text_encoder_2 = text_encoder_2,
    #     tokenizer_2 = tokenizer_2,
    #     text_encoder_3 = None,
    #     tokenizer_3 = None,
    # )
    
    transformer, pipeline = accelerator.prepare(transformer, pipeline)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(accelerator.device, dtype=weight_dtype)
    transformer.to(accelerator.device, dtype=weight_dtype)
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    
    if accelerator.is_main_process:
        accelerator.init_trackers("SimpleSDM-3")

    vae.eval()
    text_encoder.eval()
    text_encoder_2.eval()
    text_encoder_3.eval()
    transformer.eval()
    
    sample_seed = random.randint(0, 100000)
    generator = torch.Generator(device=accelerator.device)
    generator.manual_seed(sample_seed)
    shape = (1, 16, stage_resolutions[0] // 8, stage_resolutions[0] // 8) # Init latents
    noise_latents = torch.randn(shape, generator=generator, device=accelerator.device, dtype=weight_dtype).to(accelerator.device)
    
    pipeline.enable_model_cpu_offload()
    
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
    
    scheduler.set_timesteps(num_inference_steps, device=accelerator.device)    
    timesteps_stage_1, timesteps_stage_2 = scheduler.timesteps[:stage_steps[0]], scheduler.timesteps[stage_steps[0]:]
    
    with torch.no_grad():
        # Stage 1, 20 steps
        x_0_predict = generate_image(prompt, stage_resolutions[0], stage_resolutions[0], noise_latents, num_inference_steps, guidance_scale, timesteps_stage_1)
        # Stage 2, 8 steps
        x_0_predict = x_0_predict.resize((stage_resolutions[1], stage_resolutions[1]), Image.Resampling.BICUBIC)
        latents = encode_image(x_0_predict)
        noise = torch.randn_like(latents)
        latents_HR = pipeline.scheduler.scale_noise(sample=latents, noise=noise, timestep=timesteps_stage_2[0])
        output_image = generate_image(prompt, stage_resolutions[1], stage_resolutions[1], latents_HR, num_inference_steps, guidance_scale, timesteps_stage_2)

        output_image.save(os.path.join(logdir, "example_MegaFusion.png"))
    
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
        stage_resolutions=args.stage_resolutions,
        stage_steps=args.stage_steps
    )

# CUDA_VISIBLE_DEVICES=0 python inference.py --prompt "A cat is running in the rain."