"""BandNet-Mobile-1D: TFLite-friendly reader for pre-rectified body-only crops.

Designed for the 2-stage mobile pipeline where a segmenter has already produced a
perfectly horizontal resistor-body crop with unknown 180-degree direction.

Compared to ResistorBandNetV2 (``bandnet_v2.py``) this model:

* collapses the height axis early (H carries almost no sequence information once
  the crop is rectified) and keeps full width resolution,
* uses a single depthwise-separable CNN trunk instead of a dual RGB + chromatic
  encoder (chroma is a cheap 2-channel log-ratio prepend, no second encoder),
* replaces the multi-head 1D Transformer + Transformer slot decoder with a
  depthwise 1D-TCN plus a tiny single-head cross-attention slot reader,
* uses only TFLite/NNAPI-safe ops: Conv/DW-Conv, BatchNorm, HardSwish, Softmax,
  MatMul, linear interpolate. No GroupNorm/LayerNorm/GELU/MultiheadAttention.

The output dict matches V1/V2 exactly so ``train.py``, ``losses.py``,
``metrics.py``, ``decoding.py`` and ``export.py`` work unchanged::

    dense_logits      [B, num_colors+1, sequence_bins]
    slot_exist_logits [B, max_bands]
    slot_color_logits [B, max_bands, num_colors]
    slot_center       [B, max_bands]  (sigmoid, left-to-right in crop space)
    slot_width        [B, max_bands]  (sigmoid)
    count_logits      [B, max_bands+1]
    embedding         [B, d_model]    (L2-normalized)
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

ARCHITECTURE_NAME = "mobile1d"


def _cheap_chroma2(rgb: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Log-ratio chroma ``[r-g, b-g]`` directly on [0, 1] RGB.

    Cheaper and more quantization-friendly than the sRGB-linearized 5-channel
    variant in ``data.augment.make_chromatic_channels`` (no pow/where). Gives
    the trunk explicit illumination-invariant hue cues without a 2nd encoder.
    """
    x = rgb.clamp_min(eps).log()
    r_g = x[:, 0:1] - x[:, 1:2]
    b_g = x[:, 2:3] - x[:, 1:2]
    return torch.cat([r_g, b_g], dim=1)


class InvertedResidual2D(nn.Module):
    """MobileNetV2-style bottleneck: expand -> DW -> project + residual."""

    def __init__(self, in_ch: int, out_ch: int, *, stride: tuple[int, int] = (1, 1), expand: int = 4) -> None:
        super().__init__()
        hidden = max(in_ch * expand, 8)
        self.use_residual = stride == (1, 1) and in_ch == out_ch
        self.expand = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        sh, sw = stride
        self.dw = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, stride=stride, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        del sh, sw

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.project(self.dw(self.expand(x)))
        return x + y if self.use_residual else y


class HeightPool(nn.Module):
    """Collapse H -> 1 with learned attention + mean (TFLite-safe).

    Mirrors ``HeightAttentionPool`` in v2 but without GroupNorm: a 1x1 score
    conv, softmax over H, then a Conv1d mixer over [attended, mean].
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)
        self.mix = nn.Sequential(
            nn.Conv1d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.Hardswish(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=2)
        attended = (x * weights).sum(dim=2)
        pooled = torch.cat([attended, x.mean(dim=2)], dim=1)
        return self.mix(pooled)


class SE1D(nn.Module):
    """Squeeze-excitation over channels for a [B, C, L] sequence."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=2)
        s = torch.sigmoid(self.fc2(F.relu(self.fc1(s), inplace=True))).unsqueeze(-1)
        return x * s


class TCNBlock(nn.Module):
    """Depthwise-separable 1D conv + SE + residual (no attention)."""

    def __init__(self, channels: int, *, kernel: int = 7, dropout: float = 0.1) -> None:
        super().__init__()
        pad = kernel // 2
        self.dw = nn.Conv1d(channels, channels, kernel, padding=pad, groups=channels, bias=False)
        self.bn1 = nn.BatchNorm1d(channels)
        self.pw = nn.Conv1d(channels, channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(channels)
        self.se = SE1D(channels)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.hardswish(self.bn1(self.dw(x)), inplace=True)
        y = self.bn2(self.pw(y))
        y = self.se(y)
        return x + self.drop(y)


class LiteSlotReader(nn.Module):
    """Tiny single-head cross-attention slot reader (no TransformerDecoder).

    6 learned queries attend to the 1D memory with plain scaled dot-product
    attention (matmul + softmax only), followed by a small FFN. ~100x cheaper
    than ``nn.TransformerDecoder`` and free of LayerNorm/Multihead ops.
    """

    def __init__(self, d_model: int, *, num_layers: int = 2, dropout: float = 0.1) -> None:
        super().__init__()
        self.num_layers = max(int(num_layers), 1)
        self.query_proj = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(self.num_layers)])
        self.key_proj = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(self.num_layers)])
        self.ff1 = nn.ModuleList([nn.Linear(d_model, d_model * 2) for _ in range(self.num_layers)])
        self.ff2 = nn.ModuleList([nn.Linear(d_model * 2, d_model) for _ in range(self.num_layers)])
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.scale = 1.0 / math.sqrt(max(d_model, 1))

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        q = queries
        for i in range(self.num_layers):
            k = self.key_proj[i](memory)
            qq = self.query_proj[i](q)
            scores = torch.matmul(qq, k.transpose(1, 2)) * self.scale
            attn = torch.softmax(scores, dim=-1)
            q = q + self.drop(torch.matmul(attn, memory))
            q = q + self.drop(self.ff2[i](F.hardswish(self.ff1[i](q), inplace=True)))
        return q


class ResistorBandNetMobile1D(nn.Module):
    """Mobile reader with the same output contract as V1/V2."""

    def __init__(
        self,
        *,
        num_colors: int = 12,
        max_bands: int = 6,
        sequence_bins: int = 192,
        backbone: str = "mobile1d",
        pretrained: bool = False,  # accepted for config compat; trunk trains from scratch
        base_channels: int = 24,
        d_model: int = 128,
        transformer_layers: int = 3,  # reinterpreted as TCN layers
        transformer_heads: int = 4,  # accepted for config compat; unused (single-head)
        slot_decoder_layers: int = 2,  # LiteSlotReader layers
        dropout: float = 0.1,
        drop_path: float = 0.0,  # accepted for config compat; unused
        conv_kernel: int = 7,
        use_chromatic_branch: bool = True,  # cheap 2ch prepend, NOT a 2nd encoder
    ) -> None:
        super().__init__()
        del backbone, pretrained, transformer_heads, drop_path
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd integer")
        self.num_colors = int(num_colors)
        self.max_bands = int(max_bands)
        self.sequence_bins = int(sequence_bins)
        self.use_chroma2 = bool(use_chromatic_branch)

        base = max(int(base_channels), 8)
        d = max(int(d_model), 16)
        stem_in = 5 if self.use_chroma2 else 3

        c1, c2, c3 = base, base * 2, base * 3
        self.stem = nn.Sequential(
            nn.Conv2d(stem_in, base, 3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(base),
            nn.Hardswish(inplace=True),
        )
        # H: 1/2 -> 1/4 -> 1/8 -> 1/16 ; W: 1/2 -> 1/2 -> 1/2 -> 1/2
        # e.g. 64x384 -> 32x192 -> 16x192 -> 8x192 -> 4x192
        self.stage1 = InvertedResidual2D(base, c1, stride=(2, 1))
        self.stage2 = InvertedResidual2D(c1, c2, stride=(2, 1))
        self.stage3 = InvertedResidual2D(c2, c3, stride=(2, 1))
        self.proj = nn.Sequential(
            nn.Conv2d(c3, d, 1, bias=False),
            nn.BatchNorm2d(d),
            nn.Hardswish(inplace=True),
        )
        self.height_pool = HeightPool(d)

        n_tcn = max(int(transformer_layers), 1)
        self.tcn = nn.Sequential(
            *[TCNBlock(d, kernel=int(conv_kernel), dropout=float(dropout)) for _ in range(n_tcn)]
        )

        self.slot_queries = nn.Parameter(torch.randn(int(max_bands), d) * 0.02)
        self.slot_reader = LiteSlotReader(d, num_layers=int(slot_decoder_layers), dropout=float(dropout))

        self.dense_head = nn.Conv1d(d, int(num_colors) + 1, 1)
        self.slot_exist = nn.Linear(d, 1)
        # No LayerNorm/GELU here on purpose: Linear-Hardswish-Linear is
        # quant-friendly and sufficient after BN-rich trunk/TCN.
        self.slot_color = nn.Sequential(
            nn.Linear(d, d),
            nn.Hardswish(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(d, int(num_colors)),
        )
        self.slot_center_head = nn.Linear(d, 1)
        self.slot_width_head = nn.Linear(d, 1)
        self.count_head = nn.Linear(d, int(max_bands) + 1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return []

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        if self.use_chroma2:
            x = torch.cat([rgb, _cheap_chroma2(rgb)], dim=1)
        else:
            x = rgb
        feat = self.proj(self.stage3(self.stage2(self.stage1(self.stem(x)))))
        seq = self.height_pool(feat).transpose(1, 2)  # [B, W', d]
        return self.tcn(seq.transpose(1, 2)).transpose(1, 2)  # [B, W', d]

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encode(rgb)
        dense = self.dense_head(memory.transpose(1, 2))
        if dense.shape[-1] != self.sequence_bins:
            dense = F.interpolate(dense, size=self.sequence_bins, mode="linear", align_corners=False)
        queries = self.slot_queries.unsqueeze(0).expand(rgb.shape[0], -1, -1)
        slots = self.slot_reader(queries, memory)
        pooled = memory.mean(dim=1)
        return {
            "dense_logits": dense,
            "slot_exist_logits": self.slot_exist(slots).squeeze(-1),
            "slot_color_logits": self.slot_color(slots),
            "slot_center": torch.sigmoid(self.slot_center_head(slots).squeeze(-1)),
            "slot_width": torch.sigmoid(self.slot_width_head(slots).squeeze(-1)),
            "count_logits": self.count_head(pooled),
            "embedding": F.normalize(pooled, dim=-1),
        }


__all__ = ["ARCHITECTURE_NAME", "ResistorBandNetMobile1D"]
