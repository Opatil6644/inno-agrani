"""
This script performs AI video generation using the Wan Video Diffusion model. 
It utilizes a text-to-video pipeline to generate a video sequence 
based on a detailed prompt of a ferret in a forest. 
"""


# pip install ftfy
import torch
import numpy as np
from diffusers import AutoModel, WanPipeline
from diffusers.hooks.group_offloading import apply_group_offloading
from diffusers.utils import export_to_video, load_image
from transformers import UMT5EncoderModel
import argparse
import sys



def video_generation(model, output_path):
    text_encoder = UMT5EncoderModel.from_pretrained(model, subfolder="text_encoder", torch_dtype=torch.bfloat16, revision="38ec498cb3208fb688890f8cc7e94ede2cbd7f68")
    vae = AutoModel.from_pretrained(model, subfolder="vae", torch_dtype=torch.float32, revision="38ec498cb3208fb688890f8cc7e94ede2cbd7f68")
    transformer = AutoModel.from_pretrained(model, subfolder="transformer", torch_dtype=torch.bfloat16, revision="38ec498cb3208fb688890f8cc7e94ede2cbd7f68")

    # group-offloading
    onload_device = torch.device("cuda")
    offload_device = torch.device("cpu")
    apply_group_offloading(text_encoder,
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="block_level",
        num_blocks_per_group=4
    )
    transformer.enable_group_offload(
        onload_device=onload_device,
        offload_device=offload_device,
        offload_type="leaf_level",
        use_stream=True
    )

    pipeline = WanPipeline.from_pretrained(
        model,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=torch.bfloat16
    )
    pipeline.to("cuda")

    prompt = """
    The camera rushes from far to near in a low-angle shot, 
    revealing a white ferret on a log. It plays, leaps into the water, and emerges, as the camera zooms in 
    for a close-up. Water splashes berry bushes nearby, while moss, snow, and leaves blanket the ground. 
    Birch trees and a light blue sky frame the scene, with ferns in the foreground. Side lighting casts dynamic 
    shadows and warm highlights. Medium composition, front view, low angle, with depth of field.
    """
    negative_prompt = """
    Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, 
    low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, 
    misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards
    """

    output = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_frames=9,
        guidance_scale=5.0,
    ).frames[0]
    export_to_video(output, f"{output_path}/output.mp4", fps=16)


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Script to run video gen model")
    parser.add_argument("--model_name", help="Video gen model to be used")
    parser.add_argument("--results_path", help="Output path to save the video")

    args = parser.parse_args()

    try:
        print(f"Starting video generation using model: {args.model_name}...")
        video_generation(args.model_name, args.results_path)
        print(f"Successfully saved video to: {args.results_path}")

    except Exception as e:
        # Catch-all for model-specific errors (CUDA, Torch, etc.)
        print(f"An unexpected error occurred during video generation: {e}")
        # Optionally: import traceback; traceback.print_exc() 
        sys.exit(0)


if __name__ == "__main__":
    main()