from copy import deepcopy
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import torch
from torch import nn

from resistor_model.config import DEFAULT_CONFIG, load_config
from resistor_model.models.bandnet_mobile import ResistorBandNetMobile1D
from resistor_model.runtime import build_model

SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "train_rres_mobile1d_colab.py"
spec = spec_from_file_location("train_rres_mobile1d_colab", SCRIPT)
assert spec is not None and spec.loader is not None
colab = module_from_spec(spec)
spec.loader.exec_module(colab)


def test_mobile1d_output_contract_at_64x384():
    model = ResistorBandNetMobile1D(
        num_colors=12, max_bands=6, sequence_bins=192,
        base_channels=24, d_model=128, transformer_layers=3,
        transformer_heads=4, slot_decoder_layers=2,
    )
    out = model(torch.rand(2, 3, 64, 384))
    assert out["dense_logits"].shape == (2, 13, 192)
    assert out["slot_exist_logits"].shape == (2, 6)
    assert out["slot_color_logits"].shape == (2, 6, 12)
    assert out["slot_center"].shape == (2, 6)
    assert out["slot_width"].shape == (2, 6)
    assert out["count_logits"].shape == (2, 7)
    assert out["embedding"].shape == (2, 128)
    assert torch.all((out["slot_center"] >= 0) & (out["slot_center"] <= 1))


def test_mobile1d_is_tflite_safe_no_heavy_ops():
    model = ResistorBandNetMobile1D()
    banned = (nn.MultiheadAttention, nn.TransformerDecoder, nn.TransformerEncoder,
              nn.TransformerDecoderLayer, nn.TransformerEncoderLayer,
              nn.GELU, nn.GroupNorm, nn.LayerNorm)
    found = [type(m).__name__ for m in model.modules() if isinstance(m, banned)]
    assert found == []


def test_mobile1d_chroma_toggle_changes_stem():
    with_chroma = ResistorBandNetMobile1D(use_chromatic_branch=True)
    without = ResistorBandNetMobile1D(use_chromatic_branch=False)
    assert with_chroma.stem[0].in_channels == 5
    assert without.stem[0].in_channels == 3
    out = without(torch.rand(1, 3, 64, 384))
    assert out["dense_logits"].shape == (1, 13, 192)


def test_runtime_builds_mobile1d():
    cfg = deepcopy(DEFAULT_CONFIG)
    cfg["model"].update({
        "architecture": "mobile1d", "backbone": "mobile1d",
        "base_channels": 16, "d_model": 64, "transformer_layers": 2,
        "transformer_heads": 4, "slot_decoder_layers": 1,
    })
    cfg["data"]["sequence_bins"] = 96
    model = build_model(cfg)
    assert isinstance(model, ResistorBandNetMobile1D)


def test_config_accepts_mobile1d(tmp_path: Path):
    p = tmp_path / "config.yaml"
    p.write_text("model:\n  architecture: mobile1d\n", encoding="utf-8")
    assert load_config(p)["model"]["architecture"] == "mobile1d"


def test_mobile1d_colab_uses_same_three_assets():
    assert colab.DATASET_ASSETS == (
        "rres.v4i.coco.zip",
        "resistor.value.training.v8i.coco.zip",
        "Deteksi.Nilai.Resistor.v1i.coco.zip",
    )
    assert colab.RUN_DIR.name == "r1-mobile1d-colab"


def test_mobile1d_colab_config(tmp_path: Path):
    cfg = colab.build_config(tmp_path / "run", None, None)
    assert cfg["model"]["architecture"] == "mobile1d"
    assert cfg["data"]["output_size"] == [64, 384]
    assert cfg["data"]["sequence_bins"] == 192
    assert cfg["data"]["geometric_augment"] is True
    assert cfg["data"]["hflip_prob"] == 0.5
    assert cfg["loss"]["ctc"] > 0
    assert cfg["loss"]["order"] < 0.2  # direction-agnostic: must stay small
    assert cfg["loss"]["consistency"] == 0.0
    assert cfg["loss"]["kl"] == 0.0
    assert cfg["distill"]["teacher_checkpoint"] is None
    assert cfg["distill"]["weight"] == 0.0

    teacher = tmp_path / "teacher.pt"
    teacher.write_bytes(b"fake")
    cfg_t = colab.build_config(tmp_path / "run", None, teacher)
    assert cfg_t["distill"]["teacher_checkpoint"] == str(teacher)
    assert cfg_t["distill"]["weight"] > 0


def test_distill_kl_is_finite():
    from resistor_model.train import _distill_kl

    student = torch.randn(2, 6, 12, requires_grad=True)
    teacher = torch.randn(2, 6, 12)
    loss = _distill_kl(student, teacher, 3.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert student.grad is not None
