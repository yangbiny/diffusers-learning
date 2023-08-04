from diffusers import DDPMScheduler, UNet2DModel

from torch import torch
import PIL.Image
import numpy as np
import tqdm


def display_sample(sample, i):
    img_processed = sample.cpu().permute(0, 2, 3, 1)
    img_processed = (img_processed + 1.0) * 127.5
    img_processed = img_processed.numpy().astype(np.uint8)

    image_pil = PIL.Image.fromarray(img_processed[0])
    image_pil.save("/Users/reasonknow/Desktop/test.png")


repo_id = "/Users/reasonknow/.cache/huggingface/hub/models--google--ddpm-cat-256/snapshots/a30605ddf01544f8b55415aacdf2d3d3070e7113"
model = UNet2DModel.from_pretrained(repo_id, proxies={'http': 'http://127.0.0.1:1081',
                                                      'https': 'http://127.0.0.1:1081'})
config = model.config
torch.manual_seed(0)
noisy_sample = torch.randn((1, config.in_channels, config.sample_size, config.sample_size))

with torch.no_grad():
    noisy_residual = model(sample=noisy_sample, timestep=2).sample

scheduler = DDPMScheduler.from_config(repo_id)

less_noisy_sample = scheduler.step(model_output=noisy_residual, timestep=2, sample=noisy_sample).prev_sample

for i, t in enumerate(tqdm.tqdm(scheduler.timesteps)):
    # 1. predict noise residual
    with torch.no_grad():
        residual = model(noisy_sample, t).sample

    # 2. compute less noisy image and set x_t -> x_t-1
    noisy_sample = scheduler.step(residual, t, noisy_sample).prev_sample

    # 3. optionally look at image
    if (i + 1) % 50 == 0:
        display_sample(noisy_sample, i + 1)
