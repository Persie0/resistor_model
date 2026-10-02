from __future__ import annotations

import torch

from resistor_model.constants import INDEX_TO_COLOR
from resistor_model.decoder import DecodeResult, decode_resistor
from resistor_model.decoding import (
    DecodedBands,
    count_log_probs,
    decode_dense_constrained,
    decode_dense_plain,
    decode_slots,
    dense_segments,
)


EXTRA_DECODERS = {
    "constrained": "slot head with resistor grammar and count marginalization",
    "dense": "plain run-length dense-map decoder",
    "dense_constrained": "grammar-constrained dense-map decoder",
}


def _same_value(pred: DecodeResult, gt: DecodeResult) -> bool:
    return (
        pred.valid
        and not pred.ambiguous
        and pred.ohms == gt.ohms
        and pred.tolerance_percent == gt.tolerance_percent
        and pred.tempco_ppm == gt.tempco_ppm
    )


class MetricAccumulator:
    def __init__(
        self,
        num_colors: int = 12,
        max_bands: int = 6,
        *,
        extra_decoders: bool = False,
        series_bonus: float = 0.0,
    ) -> None:
        self.num_colors = num_colors
        self.max_bands = max_bands
        self.no_band_index = num_colors
        self.confusion = torch.zeros((num_colors + 1, num_colors + 1), dtype=torch.long)
        self.samples = 0
        self.exact_sequence = 0
        self.exact_value = 0
        self.value_den = 0
        self.value_ambiguous = 0
        self.count_correct = 0
        self.pos_abs = 0.0
        self.pos_n = 0
        self.dense_correct = 0
        self.dense_n = 0
        self.extra_decoders = bool(extra_decoders)
        self.series_bonus = float(series_bonus)
        self._extra = (
            {name: {"exact_sequence": 0, "exact_value": 0, "count": 0} for name in EXTRA_DECODERS}
            if self.extra_decoders
            else {}
        )

    def _extra_predictions(
        self,
        outputs: dict[str, torch.Tensor],
        batch_size: int,
    ) -> dict[str, list[DecodedBands]]:
        slot_probs = torch.softmax(outputs["slot_color_logits"].detach().float(), dim=-1).cpu().numpy()
        count_lp = count_log_probs(outputs["count_logits"], outputs.get("slot_exist_logits"))
        dense = outputs["dense_logits"].detach().cpu()
        constrained: list[DecodedBands] = []
        plain: list[DecodedBands] = []
        dense_constrained: list[DecodedBands] = []
        for i in range(batch_size):
            constrained.append(
                decode_slots(slot_probs[i], count_lp[i], series_bonus=self.series_bonus)
            )
            segments = dense_segments(dense[i])
            plain.append(decode_dense_plain(segments))
            dense_constrained.append(
                decode_dense_constrained(segments, series_bonus=self.series_bonus)
            )
        return {
            "constrained": constrained,
            "dense": plain,
            "dense_constrained": dense_constrained,
        }

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

        self.pos_abs += float((pred_center[exists] - gt_center[exists]).abs().sum())
        self.pos_n += int(exists.sum())
        self.dense_correct += int((dense_pred == dense_gt).sum())
        self.dense_n += dense_gt.numel()

        batch_size = gt_count.shape[0]
        self.samples += batch_size
        self.count_correct += int((pred_count == gt_count).sum())
        extra = self._extra_predictions(outputs, batch_size) if self.extra_decoders else {}
        for batch_index in range(batch_size):
            gt_n = max(0, min(int(gt_count[batch_index]), self.max_bands))
            pred_n = max(0, min(int(pred_count[batch_index]), self.max_bands))
            gt_seq = [int(x) for x in gt_colors[batch_index, :gt_n].tolist()]
            pred_seq = [int(x) for x in pred_colors[batch_index, :pred_n].tolist()]

            for i in range(max(gt_n, pred_n)):
                gt_color = gt_seq[i] if i < gt_n else self.no_band_index
                pred_color = pred_seq[i] if i < pred_n else self.no_band_index
                if 0 <= gt_color <= self.no_band_index and 0 <= pred_color <= self.no_band_index:
                    self.confusion[gt_color, pred_color] += 1

            if pred_n == gt_n and pred_seq == gt_seq:
                self.exact_sequence += 1

            gt_names = [INDEX_TO_COLOR[x] for x in gt_seq if x in INDEX_TO_COLOR]
            pred_names = [INDEX_TO_COLOR[x] for x in pred_seq if x in INDEX_TO_COLOR]
            gt_decoded = decode_resistor(gt_names)
            if gt_decoded.ambiguous:
                self.value_ambiguous += 1
            elif gt_decoded.valid:
                self.value_den += 1
                if _same_value(decode_resistor(pred_names), gt_decoded):
                    self.exact_value += 1

            for name, stats in self._extra.items():
                prediction = extra[name][batch_index]
                if prediction.colors == gt_names:
                    stats["exact_sequence"] += 1
                if len(prediction.colors) == gt_n:
                    stats["count"] += 1
                if not gt_decoded.ambiguous and gt_decoded.valid and _same_value(prediction.decode, gt_decoded):
                    stats["exact_value"] += 1

    def compute(self) -> dict[str, float]:
        f1s: list[float] = []
        per_color: dict[str, float] = {}
        for color in range(self.num_colors):
            true_positive = int(self.confusion[color, color])
            false_positive = int(self.confusion[:, color].sum()) - true_positive
            false_negative = int(self.confusion[color, :].sum()) - true_positive
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = 2 * true_positive / denominator if denominator > 0 else 0.0
            name = INDEX_TO_COLOR.get(color, f"class_{color}")
            per_color[f"f1_{name}"] = float(f1)
            if denominator > 0:
                f1s.append(float(f1))

        metrics = {
            "macro_f1": float(sum(f1s) / len(f1s)) if f1s else 0.0,
            "exact_sequence_accuracy": self.exact_sequence / max(self.samples, 1),
            "exact_value_accuracy": self.exact_value / max(self.value_den, 1),
            "value_decode_coverage": self.value_den / max(self.samples, 1),
            "value_ambiguity_rate": self.value_ambiguous / max(self.samples, 1),
            "count_accuracy": self.count_correct / max(self.samples, 1),
            "position_mae": self.pos_abs / max(self.pos_n, 1),
            "dense_accuracy": self.dense_correct / max(self.dense_n, 1),
        }
        metrics.update(per_color)
        for name, stats in self._extra.items():
            metrics[f"{name}_sequence_accuracy"] = stats["exact_sequence"] / max(self.samples, 1)
            metrics[f"{name}_value_accuracy"] = stats["exact_value"] / max(self.value_den, 1)
            metrics[f"{name}_count_accuracy"] = stats["count"] / max(self.samples, 1)
        return metrics
