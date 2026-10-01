from __future__ import annotations

import math
import torch
from torch import nn


class ConvNeXtLiteBlock(nn.Module):
    def __init__(self, channels: int, expansion: int = 4, drop: float = 0.0) -> None:
        super().__init__(); hidden = channels * expansion
        self.dw = nn.Conv2d(channels, channels, 7, padding=3, groups=channels); self.norm = nn.GroupNorm(1, channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1); self.act = nn.GELU(); self.pw2 = nn.Conv2d(hidden, channels, 1); self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dw(x); y = self.norm(y); y = self.pw2(self.act(self.pw1(y))); return x + self.drop(y)


def sine_position_encoding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if dim % 2: raise ValueError("d_model must be even for sine positional encoding")
    pos = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32); pe[:, 0::2] = torch.sin(pos * div); pe[:, 1::2] = torch.cos(pos * div)
    return pe.to(dtype=dtype)
