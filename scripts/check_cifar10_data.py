import math

import matplotlib.pyplot as plt

from toy_diffusion.data import CIFAR10DataModule


def denormalize(img):
    # img: (C, H, W), normalized with mean=std=0.5
    return (img * 0.5 + 0.5).clamp(0, 1)


def show_grid(images, n=16, title="Sample images from CIFAR10DataModule"):
    n = min(n, images.size(0))
    cols = int(math.sqrt(n))
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows == 1 and cols == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]
    elif cols == 1:
        axes = [[ax] for ax in axes]

    for i in range(rows * cols):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        ax.axis("off")
        if i < n:
            img = denormalize(images[i]).permute(1, 2, 0).cpu().numpy()
            ax.imshow(img)

    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dm = CIFAR10DataModule(batch_size=64, num_workers=2, seed=42)
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))  # should be images only: shape [B, 3, 32, 32]

    print(f"Batch type: {type(batch)}")
    print(f"Batch shape: {batch.shape}")
    print(f"Batch dtype: {batch.dtype}")
    print(f"Min/Max: {batch.min().item():.3f}, {batch.max().item():.3f}")

    show_grid(batch, n=16, title="Train batch samples (images only)")
