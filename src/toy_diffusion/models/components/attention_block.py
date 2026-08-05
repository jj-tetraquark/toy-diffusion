import einops
import torch
from torch import nn


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
        q = einops.rearrange(q, "b c h w -> b (h w) c")
        k = einops.rearrange(k, "b c h w -> b c (h w)")
        v = einops.rearrange(v, "b c h w -> b (h w) c")

        d_k = k.shape[1]

        attn = torch.softmax((q @ k) * (d_k**-0.5), dim=-1)
        h = einops.rearrange(attn @ v, "b (h w) c -> b c h w", w=width)

        return x + self.proj(h)
