import torch
from torch import nn

from toy_diffusion.models.components import (
    AttentionBlock,
    ResidualBlock,
    SinusoidalPositionalEmbedding,
)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        channels=(128, 256, 1024, 2048),
        num_groups=32,
        dropout=0.1,
    ):
        super().__init__()

        self._time_emb_dim = time_emb_dim
        self._time_emebedding = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self._conv_in = nn.Conv2d(in_channels, channels[0], 3, padding="same")

        self._down_blocks = nn.ModuleList()
        self._down_samples = nn.ModuleList()
        ch = channels[0]

        midpoint_channels = channels[-1]

        for in_ch, out_ch in zip(channels[:-1], channels[1:], strict=True):
            self._down_blocks.append(
                ResidualBlock(in_ch, out_ch, time_emb_dim, num_groups, dropout)
            )
            self._down_samples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))

        self._mid_block1 = ResidualBlock(
            midpoint_channels, midpoint_channels, time_emb_dim, num_groups, dropout
        )
        self._mid_attention = AttentionBlock(midpoint_channels)
        self._mid_block2 = ResidualBlock(
            midpoint_channels, midpoint_channels, time_emb_dim, num_groups, dropout
        )

        self._up_blocks = nn.ModuleList()
        self._up_samples = nn.ModuleList()

        for in_ch, out_ch in zip(channels[-1:0:-1], channels[-2::-1], strict=True):
            self._up_blocks.append(
                ResidualBlock(in_ch + out_ch, out_ch, time_emb_dim, num_groups, dropout)
            )
            self._up_samples.append(
                nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1)
            )

        self._out = nn.Sequential(
            nn.GroupNorm(num_groups, channels[0]),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding="same"),
        )

    @property
    def time_embedding_dim(self):
        return self._time_emb_dim

    def get_time_embedding(self, timestep):
        return self._time_embedding(timestep)

    def forward(self, x, timestep):
        t_emb = self.get_time_embedding(timestep)
        return self.forward_with_embedding(self, x, t_emb)

    def forward_with_embedding(self, x, emb):
        h = self._conv_in(x)

        skip_connections = []
        for down_block, down_sample in zip(
            self._down_blocks, self._down_samples, strict=True
        ):
            h = down_block(h, emb)
            skip_connections.append(h)
            h = down_sample(h)

        h = self._mid_block1(h, emb)
        h = self._mid_attention(h)
        h = self._mid_block2(h, emb)

        for up_sample, up_block in zip(self._up_samples, self._up_blocks, strict=True):
            h = up_sample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, emb)

        h = self._out(h)
        return h


if __name__ == "__main__":
    model = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        channels=[128, 256, 1024, 2048, 4096],
    )

    print(model)

    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 1000, (4, 1))
    with torch.no_grad():
        output = model(x, t)

    print(f"Input shape: {x.shape}, output shape: {output.shape}")
