from pathlib import Path

import torch

from resistor_model.export_segmentation import load_checkpoint_model, metadata_for_checkpoint
from resistor_model.segmentation import build_lraspp_model


def test_export_loader_restores_model_and_image_size(tmp_path: Path):
    model = build_lraspp_model(num_classes=2, pretrained_backbone=False)
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "architecture": "lraspp_mobilenet_v3_large",
            "num_classes": 2,
            "model": model.state_dict(),
            "image_size": 320,
            "category_names": ["resistor"],
        },
        checkpoint,
    )
    restored, payload = load_checkpoint_model(checkpoint)
    assert payload["image_size"] == 320
    with torch.no_grad():
        out = restored(torch.zeros(1, 3, 64, 64))["out"]
    assert out.shape == (1, 2, 64, 64)


def test_export_metadata_documents_normalization_and_output():
    payload = {"image_size": 384, "category_names": None}
    meta = metadata_for_checkpoint(payload)
    assert meta["input"]["shape"] == [1, 3, 384, 384]
    assert meta["output"]["shape"] == [1, 1, 384, 384]
    assert meta["output"]["meaning"] == "resistor foreground probability"
    assert meta["threshold"] == 0.5
