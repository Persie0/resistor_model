from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossWeights:
    dense: float = 1.0; color: float = 2.0; exist: float = 0.5; center: float = 1.0; width: float = 0.5; order: float = 0.2; count: float = 0.2; consistency: float = 0.0

@dataclass
class LossResult:
    total: torch.Tensor
    parts: dict[str, torch.Tensor]


def monotonic_order_loss(center: torch.Tensor, exists: torch.Tensor, margin: float = 0.005) -> torch.Tensor:
    if center.shape[1] < 2: return center.sum() * 0.0
    pair_mask = exists[:, :-1] * exists[:, 1:]; penalties = F.relu(center[:, :-1] + margin - center[:, 1:]) * pair_mask
    return penalties.sum() / pair_mask.sum().clamp_min(1.0)


def _masked_smooth_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.sum().item() == 0: return pred.sum() * 0.0
    return (F.smooth_l1_loss(pred, target, reduction="none") * mask).sum() / mask.sum().clamp_min(1.0)


def compute_loss(outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], weights: LossWeights, *, second_view: dict[str, torch.Tensor] | None = None, dense_class_weights: torch.Tensor | None = None) -> LossResult:
    dense = F.cross_entropy(outputs["dense_logits"], targets["dense_target"], weight=dense_class_weights); exist = F.binary_cross_entropy_with_logits(outputs["slot_exist_logits"], targets["slot_exists"])
    valid_color = targets["slot_colors"] != -100
    color = F.cross_entropy(outputs["slot_color_logits"].reshape(-1, outputs["slot_color_logits"].shape[-1]), targets["slot_colors"].reshape(-1), ignore_index=-100) if valid_color.any() else outputs["slot_color_logits"].sum() * 0.0
    mask = targets["slot_exists"].float(); center = _masked_smooth_l1(outputs["slot_center"], targets["slot_centers"], mask); width = _masked_smooth_l1(outputs["slot_width"], targets["slot_widths"], mask)
    order = monotonic_order_loss(outputs["slot_center"], mask); count = F.cross_entropy(outputs["count_logits"], targets["count"])
    parts = {"dense": dense, "color": color, "exist": exist, "center": center, "width": width, "order": order, "count": count}
    total = weights.dense*dense + weights.color*color + weights.exist*exist + weights.center*center + weights.width*width + weights.order*order + weights.count*count
    if weights.consistency > 0 and second_view is not None:
        consistency = (1.0 - F.cosine_similarity(outputs["embedding"], second_view["embedding"], dim=-1)).mean(); parts["consistency"] = consistency; total = total + weights.consistency * consistency
    return LossResult(total=total, parts=parts)
