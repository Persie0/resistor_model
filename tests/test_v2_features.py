from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from resistor_model.config import DEFAULT_CONFIG
from resistor_model.constants import COLOR_TO_INDEX
from resistor_model.decoding import best_valid_sequence, count_log_probs
from resistor_model.losses import LossResult, LossWeights, compute_loss, ctc_color_loss
from resistor_model.metrics import MetricAccumulator
from resistor_model.models.bandnet import ResistorBandNet
from resistor_model.models.bandnet_v2 import ResistorBandNetV2, chroma_channels
from resistor_model.runtime import build_model
from resistor_model.train import evaluate_loader


def _targets(batch: int = 1, bins: int = 24):
    colors = torch.tensor([[COLOR_TO_INDEX["brown"], COLOR_TO_INDEX["black"], COLOR_TO_INDEX["red"], COLOR_TO_INDEX["gold"], -100, -100]] * batch)
    return {
        "dense_target": torch.full((batch, bins), 12, dtype=torch.long),
        "slot_exists": torch.tensor([[1, 1, 1, 1, 0, 0]] * batch, dtype=torch.float32),
        "slot_colors": colors,
        "slot_centers": torch.tensor([[.15, .35, .55, .80, 0, 0]] * batch),
        "slot_widths": torch.tensor([[.05, .05, .05, .05, 0, 0]] * batch),
        "count": torch.tensor([4] * batch, dtype=torch.long),
    }


def test_ctc_color_loss_is_finite_with_repeated_band_colors():
    logits = torch.randn(2, 13, 32, requires_grad=True)
    colors = torch.tensor([
        [1, 1, 2, 10, -100, -100],
        [2, 0, 2, 10, -100, -100],
    ])
    loss = ctc_color_loss(logits, colors)
    assert torch.isfinite(loss)
    loss.backward()
    assert logits.grad is not None


def test_compute_loss_adds_ctc_and_prediction_kl_for_second_view():
    model = ResistorBandNet(num_colors=12, max_bands=6, sequence_bins=24, base_channels=8, d_model=32, transformer_layers=1, transformer_heads=4, slot_decoder_layers=1)
    out1 = model(torch.rand(1, 3, 32, 96))
    out2 = model(torch.rand(1, 3, 32, 96))
    result = compute_loss(
        out1,
        _targets(),
        LossWeights(consistency=0.1, ctc=0.2, kl=0.05, label_smoothing=0.05),
        second_view=out2,
    )
    assert torch.isfinite(result.total)
    assert {"ctc", "consistency", "kl_slot", "kl_dense"}.issubset(result.parts)


def test_constrained_decoder_can_choose_valid_second_choice_over_invalid_argmax():
    n = 4
    probs = np.full((n, 12), 1e-4, dtype=np.float64)
    # Argmax sequence gold-black-red-gold is invalid because gold cannot be a significant digit.
    # Brown is the second choice in slot 0, yielding brown-black-red-gold = 1 kOhm ±5%.
    sequence = ["gold", "black", "red", "gold"]
    for i, name in enumerate(sequence):
        probs[i, COLOR_TO_INDEX[name]] = 0.60
    probs[0, COLOR_TO_INDEX["brown"]] = 0.39
    probs /= probs.sum(axis=1, keepdims=True)
    decoded = best_valid_sequence(probs, top_k=2, series_bonus=0.0)
    assert decoded is not None
    assert decoded.decode.valid
    assert decoded.decode.ohms == 1000.0
    assert decoded.colors[0] == "brown"


def test_count_log_probs_returns_distribution_over_zero_to_max_bands():
    count_logits = torch.zeros(1, 7)
    exist_logits = torch.zeros(1, 6)
    out = count_log_probs(count_logits, exist_logits)
    assert out.shape == (1, 7)
    assert torch.isfinite(out).all()
    assert torch.allclose(torch.logsumexp(out, dim=-1), torch.zeros(1), atol=1e-5)


def test_metric_accumulator_can_report_optional_constrained_decoders():
    acc = MetricAccumulator(num_colors=12, max_bands=6, extra_decoders=True)
    outputs = {
        "slot_color_logits": torch.randn(1, 6, 12),
        "slot_exist_logits": torch.randn(1, 6),
        "slot_center": torch.rand(1, 6),
        "count_logits": torch.randn(1, 7),
        "dense_logits": torch.randn(1, 13, 24),
    }
    targets = _targets()
    acc.update(outputs, targets)
    metrics = acc.compute()
    assert {"constrained_sequence_accuracy", "dense_sequence_accuracy", "dense_constrained_sequence_accuracy"}.issubset(metrics)


def test_v2_chromatic_channels_are_finite_and_eight_channel():
    out = chroma_channels(torch.zeros(2, 3, 16, 32))
    assert out.shape == (2, 8, 16, 32)
    assert torch.isfinite(out).all()


def test_v2_model_has_same_output_contract_as_v1():
    model = ResistorBandNetV2(
        num_colors=12,
        max_bands=6,
        sequence_bins=48,
        backbone="convnext_lite",
        base_channels=8,
        d_model=32,
        transformer_layers=1,
        transformer_heads=4,
        slot_decoder_layers=1,
        use_chromatic_branch=True,
    )
    out = model(torch.rand(2, 3, 32, 128))
    assert out["dense_logits"].shape == (2, 13, 48)
    assert out["slot_color_logits"].shape == (2, 6, 12)
    assert out["count_logits"].shape == (2, 7)
    assert out["embedding"].shape == (2, 32)


def test_runtime_keeps_v1_default_and_builds_v2_when_requested():
    default_cfg = deepcopy(DEFAULT_CONFIG)
    assert isinstance(build_model(default_cfg), ResistorBandNet)

    v2_cfg = deepcopy(DEFAULT_CONFIG)
    v2_cfg["model"].update({
        "architecture": "v2",
        "backbone": "convnext_lite",
        "pretrained": False,
        "base_channels": 8,
        "d_model": 32,
        "transformer_layers": 1,
        "transformer_heads": 4,
        "slot_decoder_layers": 1,
    })
    assert isinstance(build_model(v2_cfg), ResistorBandNetV2)


class _EvalDataset(Dataset):
    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> dict:
        del index
        sample = {key: value[0] for key, value in _targets().items()}
        sample["image"] = torch.zeros(3, 16, 32)
        return sample


class _EvalModel(torch.nn.Module):
    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = image.shape[0]
        return {
            "slot_color_logits": torch.zeros(batch, 6, 12),
            "slot_exist_logits": torch.zeros(batch, 6),
            "slot_center": torch.zeros(batch, 6),
            "slot_width": torch.zeros(batch, 6),
            "count_logits": torch.zeros(batch, 7),
            "dense_logits": torch.zeros(batch, 13, 24),
            "embedding": torch.zeros(batch, 8),
        }


def test_evaluate_loader_uses_supplied_training_color_weights(monkeypatch):
    expected = torch.linspace(0.5, 1.5, 12)
    captured: list[torch.Tensor | None] = []

    def fake_compute_loss(outputs, targets, weights, **kwargs):
        del targets, weights
        captured.append(kwargs.get("color_class_weights"))
        zero = outputs["dense_logits"].sum() * 0.0
        return LossResult(total=zero, parts={})

    monkeypatch.setattr("resistor_model.train.compute_loss", fake_compute_loss)
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["data"]["num_workers"] = 0
    loader = DataLoader(_EvalDataset(), batch_size=1)
    evaluate_loader(
        _EvalModel(),
        loader,
        torch.device("cpu"),
        cfg,
        color_class_weights=expected,
        show_progress=False,
    )
    assert len(captured) == 1
    assert captured[0] is expected
