import einops
import torch
from torch import nn


class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("Only use even embedding dimension sizes")

        half = dim // 2
        denom = torch.pow(10000, torch.arange(half) / half)
        self.register_buffer("_denom", denom, persistent=False)

    def forward(self, pos):
        embedding = pos / self._denom
        return torch.cat((embedding.sin(), embedding.cos()), dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim, num_groups=32, dropout=0.1):
        super().__init__()
        self._time_proj = nn.Linear(time_embedding_dim, out_channels)

        self._block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding="same")
        )

        self._block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding="same")
        )

        if in_channels != out_channels:
            self._res_conv = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self._res_conv = nn.Identity()

    def forward(self, x, time_emb):
        h = self._block1(x)
        time_emb = self._time_proj(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self._block2(h)
        return h + self._res_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h = self.norm(x)
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)

        width = q.shape[-1]
        q = einops.rearrange(q, 'b c h w -> b (h w) c')
        k = einops.rearrange(k, 'b c h w -> b c (h w)')
        v = einops.rearrange(v, 'b c h w -> b (h w) c')

        d_k = k.shape[1]

        attn = torch.softmax((q @ k) * (d_k**-0.5), dim=-1)
        h = einops.rearrange(attn @ v, 'b (h w) c -> b c h w', w=width)

        return x + self.proj(h)

class UNet(nn.Module):
    def __init__(
        self,
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        channels=(128, 256, 1024, 2048),
        num_groups=32,
        dropout=0.1
    ):
        super().__init__()

        self._time_emebedding = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim)
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

    def forward(self, x, timestep):
        t_emb = self._time_emebedding(timestep)

        h = self._conv_in(x)

        skip_connections = []
        for down_block, down_sample in zip(self._down_blocks, self._down_samples, strict=True):
            h = down_block(h, t_emb)
            skip_connections.append(h)
            h = down_sample(h)

        h = self._mid_block1(h, t_emb)
        h = self._mid_attention(h)
        h = self._mid_block2(h, t_emb)

        for up_sample, up_block in zip(self._up_samples, self._up_blocks, strict=True):
            h = up_sample(h)
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=1)
            h = up_block(h, t_emb)

        h = self._out(h)
        return h


if __name__ == '__main__':
    model = UNet(
        in_channels=3,
        out_channels=3,
        time_emb_dim=128,
        channels=[128, 256, 1024, 2048, 4096]
    )

    print(model)

    x = torch.randn(4, 3, 64, 64)
    t = torch.randint(0, 1000, (4,1))
    with torch.no_grad():
        output = model(x, t)

    print(f"Input shape: {x.shape}, output shape: {output.shape}")
