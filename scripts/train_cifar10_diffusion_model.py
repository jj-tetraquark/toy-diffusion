import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from toy_diffusion.models.unet import UNet
from toy_diffusion.data.cifar10_datamodule import CIFAR10DataModule
from toy_diffusion.training.diffusion import DiffusionModule, cosine_beta_schedule

if __name__ == "__main__":
    datamodule = CIFAR10DataModule(num_workers=4)

    model = DiffusionModule(
        UNet(),
        cosine_beta_schedule,
        timesteps=1000,
        lr=3e-4,
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
        max_epochs=60,
        callbacks=[checkpoint_cb],
    )

    trainer.fit(
        model,
        datamodule=datamodule,
    )
