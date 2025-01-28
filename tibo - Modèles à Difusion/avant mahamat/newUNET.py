import math
from typing import Optional, Tuple, Union, List

import torch
from torch import nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class TimeEmbedding(nn.Module):
    """
    ### Embeddings for $t$
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels  # Stocke le nombre total de canaux pour l'embedding
        # First linear layer
        self.lin1 = nn.Linear(n_channels // 4, n_channels)
        # Activation
        self.act = Swish()
        # Second linear layer
        self.lin2 = nn.Linear(n_channels, n_channels)

    def forward(self, t: torch.Tensor):
        # Create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)

        # Transform with the MLP
        emb = self.act(self.lin1(emb))
        emb = self.lin2(emb)

        return emb

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels)
        self.act1 = Swish()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels)
        self.act2 = Swish()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x, t):
        h = self.conv1(self.act1(self.norm1(x)))
        h += self.time_emb(t)[:, :, None, None]
        h = self.conv2(self.dropout(self.act2(self.norm2(h))))
        return h + self.shortcut(x)

class AttentionBlock(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, n_channels)
        self.qkv = nn.Conv2d(n_channels, n_channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(n_channels, n_channels, kernel_size=1)
        self.scale = (n_channels // 8) ** -0.5

    def forward(self, x):
        batch, channels, height, width = x.shape
        x_norm = self.norm(x).view(batch, channels, -1)
        q, k, v = torch.chunk(self.qkv(x_norm), 3, dim=1)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).view(batch, channels, height, width)
        return x + self.proj_out(out)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, t):
        x = self.res(x, t)
        return self.attn(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn):
        super().__init__()
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x, skip, t):
        x = torch.cat([x, skip], dim=1)
        x = self.res(x, t)
        return self.attn(x)

class MiddleBlock(nn.Module):
    def __init__(self, n_channels, time_channels):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x, t):
        x = self.res1(x, t)
        x = self.attn(x)
        return self.res2(x, t)

class UNet(nn.Module):
    def __init__(self, image_channels=3, n_channels=64, ch_mults=(1, 2, 4, 8), is_attn=(False, False, True, True)):
        super().__init__()
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)
        self.time_emb = TimeEmbedding(n_channels * 4)
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        prev_channels = n_channels

        for i, mult in enumerate(ch_mults):
            out_channels = n_channels * mult
            self.down.append(DownBlock(prev_channels, out_channels, n_channels * 4, is_attn[i]))
            prev_channels = out_channels
            if i != len(ch_mults) - 1:
                self.down.append(nn.Conv2d(prev_channels, prev_channels, kernel_size=3, stride=2, padding=1))

        self.middle = MiddleBlock(prev_channels, n_channels * 4)

        for i, mult in reversed(list(enumerate(ch_mults))):
            out_channels = n_channels * mult
            self.up.append(UpBlock(prev_channels, out_channels, n_channels * 4, is_attn[i]))
            prev_channels = out_channels
            if i != 0:
                self.up.append(nn.ConvTranspose2d(prev_channels, prev_channels, kernel_size=4, stride=2, padding=1))

        self.final = nn.Conv2d(n_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x, t):
        t = self.time_emb(t)
        x = self.image_proj(x)
        skips = []

        for layer in self.down:
            x = layer(x, t)
            if isinstance(layer, DownBlock):
                skips.append(x)

        x = self.middle(x, t)

        for layer in self.up:
            if isinstance(layer, UpBlock):
                x = layer(x, skips.pop(), t)
            else:
                x = layer(x)

        return self.final(x)
