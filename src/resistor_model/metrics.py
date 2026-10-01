from __future__ import annotations

import torch

from resistor_model.constants import INDEX_TO_COLOR
from resistor_model.decoder import decode_resistor


class MetricAccumulator:
    def __init__(self, num_colors: int = 12, max_bands: int = 6) -> None:
        self.num_colors = num_colors
        self.max_bands = max_bands
        self.confusion = torch.zeros((num_colors, num_colors), dtype=torch.long)
        self.samples = 0
        self.exact_sequence = 0
        self.exact_value = 0
        self.value_den = 0
        self.count_correct = 0
        self.pos_abs = 0.0
        self.pos_n = 0
        self.dense_correct = 0
        self.dense_n = 0

    @torch.no_grad()
    def update(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> None:
        pred_colors = outputs["slot_color_logits"].argmax(-1).detach().cpu()
        pred_count = outputs["count_logits"].argmax(-1).detach().cpu()
        pred_center = outputs["slot_center"].detach().cpu()
        gt_colors = targets["slot_colors"].detach().cpu()
        gt_count = targets["count"].detach().cpu()
        gt_center = targets["slot_centers"].detach().cpu()
        exists = targets["slot_exists"].detach().cpu().bool()
        dense_pred = outputs["dense_logits"].argmax(1).detach().cpu()
        dense_gt = targets["dense_target"].detach().cpu()

        valid = gt_colors >= 0
        for g, p in zip(gt_colors[valid].tolist(), pred_colors[valid].tolist()):
            if 0 <= g < self.num_colors and 0 <= p < self.num_colors:
                self.confusion[g, p] += 1
        self.pos_abs += float((pred_center[exists] - gt_center[exists]).abs().sum())
        self.pos_n += int(exists.sum())
        self.dense_correct += int((dense_pred == dense_gt).sum())
        self.dense_n += dense_gt.numel()

        bsz = gt_count.shape[0]
        self.samples += bsz
        self.count_correct += int((pred_count == gt_count).sum())
        for b in range(bsz):
            gc = int(gt_count[b])
            pc = int(pred_count[b])
            gt_seq = [int(x) for x in gt_colors[b, :gc].tolist()]
            pred_seq = [int(x) for x in pred_colors[b, :pc].tolist()]
            if pc == gc and pred_seq == gt_seq:
                self.exact_sequence += 1
            gt_names = [INDEX_TO_COLOR[x] for x in gt_seq if x in INDEX_TO_COLOR]
            pred_names = [INDEX_TO_COLOR[x] for x in pred_seq if x in INDEX_TO_COLOR]
            gt_dec = decode_resistor(gt_names)
            if gt_dec.valid:
                self.value_den += 1
                pred_dec = decode_resistor(pred_names)
                if pred_dec.valid and pred_dec.ohms == gt_dec.ohms and pred_dec.tolerance_percent == gt_dec.tolerance_percent and pred_dec.tempco_ppm == gt_dec.tempco_ppm:
                    self.exact_value += 1

    def compute(self) -> dict[str, float]:
        f1s: list[float] = []
        per_color: dict[str, float] = {}
        for c in range(self.num_colors):
            tp = int(self.confusion[c, c])
            fp = int(self.confusion[:, c].sum()) - tp
            fn = int(self.confusion[c, :].sum()) - tp
            denom = 2 * tp + fp + fn
            f1 = 2 * tp / denom if denom > 0 else 0.0
            name = INDEX_TO_COLOR.get(c, f"class_{c}")
            per_color[f"f1_{name}"] = float(f1)
            if denom > 0:
                f1s.append(float(f1))
        metrics = {
            "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "exact_sequence_accuracy": self.exact_sequence / max(self.samples, 1),
            "exact_value_accuracy": self.exact_value / max(self.value_den, 1),
            "count_accuracy": self.count_correct / max(self.samples, 1),
            "position_mae": self.pos_abs / max(self.pos_n, 1),
            "dense_accuracy": self.dense_correct / max(self.dense_n, 1),
        }
        metrics.update(per_color)
        return metrics
