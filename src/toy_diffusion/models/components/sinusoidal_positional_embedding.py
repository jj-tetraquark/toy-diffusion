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
