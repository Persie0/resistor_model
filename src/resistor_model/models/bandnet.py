from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from resistor_model.data.augment import make_chromatic_channels
from .blocks import ConvNeXtLiteBlock, sine_position_encoding


class _VisualEncoder(nn.Module):
    def __init__(self, in_channels: int, base: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, stride=(2, 2), padding=1), nn.GroupNorm(1, base), nn.GELU(), ConvNeXtLiteBlock(base),
            nn.Conv2d(base, base * 2, 3, stride=(2, 2), padding=1), nn.GroupNorm(1, base * 2), nn.GELU(), ConvNeXtLiteBlock(base * 2),
            nn.Conv2d(base * 2, base * 4, 3, stride=(2, 1), padding=1), nn.GroupNorm(1, base * 4), nn.GELU(), ConvNeXtLiteBlock(base * 4), nn.Conv2d(base * 4, out_channels, 1),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor: return self.net(x)


class ResistorBandNet(nn.Module):
    def __init__(self, *, num_colors: int = 12, max_bands: int = 6, sequence_bins: int = 256, base_channels: int = 48, d_model: int = 256, transformer_layers: int = 4, transformer_heads: int = 8, slot_decoder_layers: int = 2, dropout: float = 0.1, use_chromatic_branch: bool = True) -> None:
        super().__init__()
        if d_model % transformer_heads: raise ValueError("d_model must be divisible by transformer_heads")
        self.num_colors = num_colors; self.max_bands = max_bands; self.sequence_bins = sequence_bins; self.use_chromatic_branch = use_chromatic_branch
        self.visual = _VisualEncoder(3, base_channels, d_model)
        if use_chromatic_branch:
            chroma_channels = max(base_channels // 2, 8); chroma_out = max(d_model // 2, 16)
            self.chroma = _VisualEncoder(5, chroma_channels, chroma_out); self.fuse = nn.Sequential(nn.Conv2d(d_model + chroma_out, d_model, 1), nn.GroupNorm(1, d_model), nn.GELU())
        else:
            self.chroma = None; self.fuse = nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=transformer_heads, dim_feedforward=d_model * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=transformer_heads, dim_feedforward=d_model * 3, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.slot_decoder = nn.TransformerDecoder(decoder_layer, num_layers=slot_decoder_layers, norm=nn.LayerNorm(d_model)); self.slot_queries = nn.Parameter(torch.randn(max_bands, d_model) * 0.02)
        self.dense_head = nn.Conv1d(d_model, num_colors + 1, 1); self.slot_exist = nn.Linear(d_model, 1); self.slot_color = nn.Linear(d_model, num_colors)
        self.slot_center_head = nn.Linear(d_model, 1); self.slot_width_head = nn.Linear(d_model, 1); self.count_head = nn.Linear(d_model, max_bands + 1)

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        visual = self.visual(rgb)
        if self.use_chromatic_branch:
            chroma = self.chroma(make_chromatic_channels(rgb))
            if chroma.shape[-2:] != visual.shape[-2:]: chroma = F.interpolate(chroma, size=visual.shape[-2:], mode="bilinear", align_corners=False)
            visual = self.fuse(torch.cat([visual, chroma], dim=1))
        seq = visual.mean(dim=2).transpose(1, 2); pos = sine_position_encoding(seq.shape[1], seq.shape[2], seq.device, seq.dtype)
        return self.transformer(seq + pos.unsqueeze(0))

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encode(rgb); dense = self.dense_head(memory.transpose(1, 2)); dense = F.interpolate(dense, size=self.sequence_bins, mode="linear", align_corners=False)
        queries = self.slot_queries.unsqueeze(0).expand(rgb.shape[0], -1, -1); slots = self.slot_decoder(queries, memory); pooled = memory.mean(dim=1)
        return {"dense_logits": dense, "slot_exist_logits": self.slot_exist(slots).squeeze(-1), "slot_color_logits": self.slot_color(slots), "slot_center": torch.sigmoid(self.slot_center_head(slots).squeeze(-1)), "slot_width": torch.sigmoid(self.slot_width_head(slots).squeeze(-1)), "count_logits": self.count_head(pooled), "embedding": F.normalize(pooled, dim=-1)}
