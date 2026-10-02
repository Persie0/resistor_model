"""ResistorBandNetV2 with stronger chromatic features and local sequence context."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from resistor_model.data.augment import make_chromatic_channels
from .bandnet import _VisualEncoder
from .blocks import sine_position_encoding

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
RESNET_BACKBONES = ("resnet18", "resnet34")
BACKBONES = RESNET_BACKBONES + ("convnext_lite",)


def chroma_channels(rgb: torch.Tensor) -> torch.Tensor:
    """RGB -> log-RGB/chromaticity plus local row-normalized log-RGB (8 channels)."""
    base = make_chromatic_channels(rgb)
    log_rgb = base[:, :3]
    local = log_rgb - log_rgb.mean(dim=3, keepdim=True)
    return torch.cat([base, local], dim=1)


class ResNetTrunk(nn.Module):
    out_channels = 256

    def __init__(self, name: str = "resnet34", pretrained: bool = True) -> None:
        super().__init__()
        if name not in RESNET_BACKBONES:
            raise ValueError(f"backbone must be one of {RESNET_BACKBONES}, got {name!r}")
        try:
            import torchvision
        except ImportError as exc:
            raise RuntimeError("ResNet V2 backbones require the optional 'resnet' dependency (torchvision)") from exc
        weights = "IMAGENET1K_V1" if pretrained else None
        net = getattr(torchvision.models, name)(weights=weights)
        for layer in (net.layer2, net.layer3):
            block = layer[0]
            block.conv1.stride = (2, 1)
            if block.downsample is not None:
                block.downsample[0].stride = (2, 1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        self.register_buffer("mean", torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.mean) / self.std
        return self.layer3(self.layer2(self.layer1(self.stem(x))))


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0) -> None:
        super().__init__()
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        keep = 1.0 - self.p
        mask = (torch.rand(x.shape[0], 1, 1, device=x.device, dtype=x.dtype) < keep).to(x.dtype) / keep
        return x * mask


class HeightAttentionPool(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, 1)
        self.mix = nn.Conv1d(channels * 2, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.score(x), dim=2)
        attended = (x * weights).sum(dim=2)
        return self.mix(torch.cat([attended, x.mean(dim=2)], dim=1))


class ConvTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        heads: int,
        dropout: float,
        drop_path: float,
        kernel: int = 7,
        ff_mult: int = 4,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel // 2, groups=d_model)
        self.pw = nn.Conv1d(d_model, d_model, 1)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.drop = nn.Dropout(dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.norm1(x)
        y = self.attn(y, y, y, need_weights=False)[0]
        x = x + self.drop_path(self.drop(y))
        y = self.norm2(x).transpose(1, 2)
        y = self.pw(F.gelu(self.dw(y))).transpose(1, 2)
        x = x + self.drop_path(self.drop(y))
        return x + self.drop_path(self.drop(self.ff(self.norm3(x))))


class ResistorBandNetV2(nn.Module):
    def __init__(
        self,
        *,
        num_colors: int = 12,
        max_bands: int = 6,
        sequence_bins: int = 192,
        backbone: str = "convnext_lite",
        pretrained: bool = False,
        base_channels: int = 48,
        d_model: int = 256,
        transformer_layers: int = 3,
        transformer_heads: int = 8,
        slot_decoder_layers: int = 2,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        conv_kernel: int = 7,
        use_chromatic_branch: bool = True,
    ) -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"backbone must be one of {BACKBONES}, got {backbone!r}")
        if d_model % transformer_heads:
            raise ValueError("d_model must be divisible by transformer_heads")
        if conv_kernel < 1 or conv_kernel % 2 == 0:
            raise ValueError("conv_kernel must be a positive odd integer")
        self.num_colors = int(num_colors)
        self.max_bands = int(max_bands)
        self.sequence_bins = int(sequence_bins)
        self.use_chromatic_branch = bool(use_chromatic_branch)
        self.pretrained_backbone = bool(pretrained) and backbone in RESNET_BACKBONES

        if backbone == "convnext_lite":
            self.trunk = _VisualEncoder(3, base_channels, d_model)
            trunk_channels = d_model
        else:
            self.trunk = ResNetTrunk(backbone, pretrained=pretrained)
            trunk_channels = ResNetTrunk.out_channels
        self.proj = nn.Sequential(
            nn.Conv2d(trunk_channels, d_model, 1),
            nn.GroupNorm(1, d_model),
            nn.GELU(),
        )

        if use_chromatic_branch:
            chroma_base = max(base_channels // 2, 8)
            chroma_out = max(d_model // 2, 16)
            self.chroma = _VisualEncoder(8, chroma_base, chroma_out)
            self.fuse = nn.Sequential(
                nn.Conv2d(d_model + chroma_out, d_model, 1),
                nn.GroupNorm(1, d_model),
                nn.GELU(),
            )
        else:
            self.chroma = None
            self.fuse = nn.Identity()

        self.height_pool = HeightAttentionPool(d_model)
        rates = torch.linspace(0.0, float(drop_path), max(transformer_layers, 1)).tolist()
        self.blocks = nn.ModuleList(
            [
                ConvTransformerBlock(d_model, transformer_heads, dropout, rates[i], conv_kernel)
                for i in range(transformer_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=transformer_heads,
            dim_feedforward=d_model * 3,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=slot_decoder_layers,
            norm=nn.LayerNorm(d_model),
        )
        prior = sine_position_encoding(max_bands, d_model, torch.device("cpu"), torch.float32)
        self.slot_queries = nn.Parameter(prior * 0.5 + torch.randn(max_bands, d_model) * 0.02)

        self.dense_head = nn.Conv1d(d_model, num_colors + 1, 1)
        self.slot_exist = nn.Linear(d_model, 1)
        self.slot_color = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_colors),
        )
        self.slot_center_head = nn.Linear(d_model, 1)
        self.slot_width_head = nn.Linear(d_model, 1)
        self.count_head = nn.Linear(d_model, max_bands + 1)

    def backbone_parameters(self) -> list[nn.Parameter]:
        return list(self.trunk.parameters()) if self.pretrained_backbone else []

    def encode(self, rgb: torch.Tensor) -> torch.Tensor:
        feat = self.proj(self.trunk(rgb))
        if self.chroma is not None:
            chroma = self.chroma(chroma_channels(rgb))
            if chroma.shape[-2:] != feat.shape[-2:]:
                chroma = F.adaptive_avg_pool2d(chroma, feat.shape[-2:])
            feat = self.fuse(torch.cat([feat, chroma], dim=1))
        seq = self.height_pool(feat).transpose(1, 2)
        seq = seq + sine_position_encoding(seq.shape[1], seq.shape[2], seq.device, seq.dtype).unsqueeze(0)
        for block in self.blocks:
            seq = block(seq)
        return self.norm(seq)

    def forward(self, rgb: torch.Tensor) -> dict[str, torch.Tensor]:
        memory = self.encode(rgb)
        dense = self.dense_head(memory.transpose(1, 2))
        if dense.shape[-1] != self.sequence_bins:
            dense = F.interpolate(dense, size=self.sequence_bins, mode="linear", align_corners=False)
        queries = self.slot_queries.unsqueeze(0).expand(rgb.shape[0], -1, -1)
        slots = self.slot_decoder(queries, memory)
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
