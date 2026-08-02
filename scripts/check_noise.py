import matplotlib.pyplot as plt
from toy_diffusion.data import CIFAR10DataModule
from toy_diffusion.training.diffusion import (
    DiffusionModule,
    linear_beta_schedule,
    cosine_beta_schedule,
)


def denormalize(img):
    # img: (C, H, W), normalized with mean=std=0.5
    return (img * 0.5 + 0.5).clamp(0, 1)


def show_noise_progression(image, timesteps=10, title="Noise application", schedule="cosine"):
    fig, axes = plt.subplots(1, timesteps, figsize=(timesteps * 2, 2))

    diffusion = DiffusionModule(
        None,
        cosine_beta_schedule if schedule == "cosine" else linear_beta_schedule,
        timesteps,
        1e-3
    )

    for t in range(timesteps):
        ax = axes[t]
        ax.axis("off")

        img = diffusion.add_noise(image, t)[0]
        img = denormalize(img).permute(1, 2, 0).cpu().numpy()

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

    show_noise_progression(batch[0:1], timesteps=16)
