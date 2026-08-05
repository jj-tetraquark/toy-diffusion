from torch import nn


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_embedding_dim, num_groups=32, dropout=0.1
    ):
        super().__init__()
        self._time_proj = nn.Linear(time_embedding_dim, out_channels)

        self._block1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding="same"),
        )

        self._block2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding="same"),
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
