import torch
from diffusers import StableDiffusionPipeline

repo_id = "/Users/reasonknow/Downloads/models/d/beenyou_r13"
pipeline = StableDiffusionPipeline.from_pretrained(repo_id, proxies={"https": "http://127.0.0.1:1081"})

prompt = "handsome male,big muscle,suit,feather coat,monochrome photography,dutch angle,outdoor"
negative_prompt = "nsfw,ng_deepnegative_v1_75t,badhandv4,(worst quality:2),(low quality:2),(normal quality:2),lowres,watermark,"
generator = torch.manual_seed(1345692449)
pipeline.clip_skip = 2

with torch.no_grad():
    pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=30,
        guidance_scale=7,
        generator=generator
    )

print(pipeline)
