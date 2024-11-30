import os
import cv2
import torch
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, List
from diffusers import ControlNetModel, DDIMScheduler
from model.pipeline_controlnet import StableDiffusionControlNetPipeline

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--logdir', default="./test/", type=str)
    parser.add_argument('--prompt', default="the mona lisa", type=str)
    parser.add_argument('--ckpt', default='./ckpt/stable-diffusion-v1-5/', type=str)    # "stable-diffusion-v1-5/stable-diffusion-v1-5"
    parser.add_argument('--controlnet_path', default='./ckpt/sd-controlnet-canny/', type=str)
    parser.add_argument('--condition_image', default='./condition_image.png', type=str)
    parser.add_argument('--mixed_precision', default='fp16', type=str) # 'fp16', 'bf16', 'no'
    parser.add_argument('--num_inference_steps', default=50, type=int)
    parser.add_argument('--stage_resolutions', default=[512, 768, 1024], type=int, nargs='+')
    parser.add_argument('--stage_steps', default=[40, 5, 5], type=int, nargs='+')

    return parser

def test(
    pretrained_model_path: str,
    prompt: str,
    controlnet_path: str,
    condition_image: str,
    logdir: str,
    num_inference_steps,
    mixed_precision: Optional[str] = "fp16",
    stage_resolutions: Optional[List[int]] = [512, 768, 1024],
    stage_steps: Optional[List[int]] = [40, 5, 5]
):

    device = "cuda:0"

    if not os.path.exists(logdir):
        os.mkdir(logdir)

    weight_dtype = torch.float32
    if mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    
    image = Image.open(condition_image)
    image = np.array(image)
    image = cv2.Canny(image, threshold1=100, threshold2=200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)

    controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=weight_dtype)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(pretrained_model_path, controlnet=controlnet, torch_dtype=weight_dtype)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()

    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps_stage_1, timesteps_stage_2, timesteps_stage_3 = pipe.scheduler.timesteps[:stage_steps[0]], pipe.scheduler.timesteps[stage_steps[0]:stage_steps[0] + stage_steps[1]], pipe.scheduler.timesteps[stage_steps[0] + stage_steps[1]:]

    def encode_image(image):
        tensor = transforms.ToTensor()(image).float() * 2. - 1.
        tensor = tensor.unsqueeze(0).to(device, dtype=weight_dtype)
        latents = pipe.vae.encode(tensor).latent_dist.sample() * pipe.vae.config.scaling_factor

        return latents
    
    # Stage 1, 40 steps
    _, x_0_predict = pipe(prompt=prompt, image=canny_image, num_samples=1, width=stage_resolutions[0], height=stage_resolutions[0], num_inference_steps=num_inference_steps, stage_timesteps=timesteps_stage_1, return_dict=True)
    
    # Stage 2, 5 steps
    x_0_predict = x_0_predict.images[0].resize((stage_resolutions[1], stage_resolutions[1]), Image.Resampling.BICUBIC)
    latents = encode_image(x_0_predict)
    noise = torch.randn_like(latents)
    latents_MR = pipe.scheduler.add_noise(latents, noise, timesteps_stage_2[0])
    _, x_0_predict = pipe(prompt=prompt, image=canny_image, latents=latents_MR, num_samples=1, width=stage_resolutions[1], height=stage_resolutions[1], num_inference_steps=num_inference_steps, stage_timesteps=timesteps_stage_2, return_dict=True)
    
    # Stage 3, 5 steps
    x_0_predict = x_0_predict.images[0].resize((stage_resolutions[2], stage_resolutions[2]), Image.Resampling.BICUBIC)
    latents = encode_image(x_0_predict)
    noise = torch.randn_like(latents)
    latents_HR = pipe.scheduler.add_noise(latents, noise, timesteps_stage_3[0])
    _, output_image = pipe(prompt=prompt, image=canny_image, latents=latents_HR, num_samples=1, width=stage_resolutions[2], height=stage_resolutions[2], num_inference_steps=num_inference_steps, stage_timesteps=timesteps_stage_3, return_dict=True)
    output_image.images[0].save(os.path.join(logdir, "example_MegaFusion.png"))
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    test(
        prompt=args.prompt,
        pretrained_model_path=args.ckpt,
        controlnet_path=args.controlnet_path,
        condition_image=args.condition_image,
        logdir=args.logdir,
        num_inference_steps=args.num_inference_steps,
        stage_resolutions=args.stage_resolutions,
        mixed_precision=args.mixed_precision,
        stage_steps=args.stage_steps
    )