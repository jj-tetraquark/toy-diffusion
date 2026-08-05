from argparse import ArgumentParser

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from toy_diffusion.data.cifar10_datamodule import CIFAR10DataModule
from toy_diffusion.models.unet import UNet
from toy_diffusion.training.diffusion import DiffusionModule
from toy_diffusion.utils.beta_schedules import cosine_beta_schedule


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    datamodule = CIFAR10DataModule(num_workers=4, batch_size=args.batch_size, images_only=True)

    model = DiffusionModule(
        UNet(),
        cosine_beta_schedule,
        timesteps=1000,
        lr=args.lr,
    )

    logger = TensorBoardLogger("tb_logs", name="diffusion_cifar10")

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="epoch{epoch:02d}-val_loss{val_loss:.2f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = L.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )
