import torch
import torch.nn.functional as F
import lightning as L


def linear_beta_schedule(timestamps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timestamps)


def cosine_beta_schedule(timestamps, s=0.008):
    t = torch.arange(timestamps + 1)
    f_x = torch.cos((t / timestamps + s) / (1 + s) * torch.pi / 2) ** 2
    a_t = f_x / f_x[0]
    B_t = 1 - a_t[1:] / (a_t[:-1])
    return torch.clip(B_t, 1e-4, 0.9999)


class DiffusionModule(L.LightningModule):
    def __init__(self, model, beta_schedule, timesteps, lr, noise=None):
        super().__init__()
        self.model = model
        self.timesteps = timesteps

        self._lr = lr

        betas = beta_schedule(timesteps)
        self.register_buffer("_alphas_cumprod", (1 - betas).cumprod(dim=0))

        if noise is None:
            self._noise = torch.randn_like

    def add_noise(self, x_0, t):
        noise = self._noise(x_0)

        a_bar = self._alphas_cumprod[t].view(-1, 1, 1, 1)
        x_t = a_bar.sqrt() * x_0 + (1 - a_bar).sqrt() * noise

        return x_t, noise

    def _do_step(self, batch, batch_idx):
        x = batch
        batch_size = x.shape[0]

        t = torch.randint(0, self.timesteps, (batch_size, 1)).long().to(self.device)

        noised_images, noise = self.add_noise(x, t)
        noise_pred = self.model(noised_images, t)

        loss = F.mse_loss(noise_pred, noise)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._do_step(batch, batch_idx)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._do_step(batch, batch_idx)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=30, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
