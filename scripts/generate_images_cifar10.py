import math
import numpy as np
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import torch
import einops

from toy_diffusion.models.unet import UNet
from toy_diffusion.sampling import sample_ddim
from toy_diffusion.training.diffusion import DiffusionModule
from toy_diffusion.utils.beta_schedules import cosine_beta_schedule


def parse_args():
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--checkpoint", required=True)
    argument_parser.add_argument("-n", type=int, help="num samples", required=True)

    return argument_parser.parse_args()

def grid_dims(n: int) -> tuple[int, int]:
    if n < 1:
        raise ValueError("n must be >= 1")
    rows = int(math.sqrt(n))
    while n % rows != 0 and n / rows > rows + 1:
        rows -= 1
    cols = math.ceil(n / rows)
    return rows, cols

if __name__ == "__main__":
    args = parse_args()

    model = DiffusionModule(UNet(), cosine_beta_schedule, timesteps=1000, lr=0)

    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])

    n_rows, n_cols = grid_dims(args.n)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

    images = sample_ddim(model, (args.n, 3, 64, 64), inference_steps=50, temperature=0.0).numpy()

    for i, ax in enumerate(axes.flatten()):
        img = np.floor(images[i] * 255).astype(np.uint8)
        img = einops.rearrange(img, 'c h w -> h w c')
        ax.imshow(img)
        ax.axis("off")

    plt.savefig("generated.png", bbox_inches="tight", dpi=150)
    plt.close()
