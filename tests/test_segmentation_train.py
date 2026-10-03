from pathlib import Path

import torch
from torch import nn

from resistor_model.train_segmentation import (
    build_parser,
    evaluate_model,
    foreground_metrics,
    format_training_progress,
    should_save_periodic_checkpoint,
)


class FixedModel(nn.Module):
    def forward(self, x):
        logits = torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)
        logits[:, 1] = 5.0
        return {"out": logits}


def test_foreground_metrics_reports_perfect_mask():
    logits = torch.full((1, 2, 4, 4), -5.0)
    logits[:, 1] = 5.0
    target = torch.ones(1, 4, 4, dtype=torch.long)
    metrics = foreground_metrics(logits, target)
    assert metrics["intersection"] == 16
    assert metrics["predicted"] == 16
    assert metrics["target"] == 16


def test_evaluate_model_aggregates_iou_and_dice():
    images = torch.zeros(2, 3, 8, 8)
    masks = torch.ones(2, 8, 8, dtype=torch.long)
    loader = [(images, masks)]
    metrics = evaluate_model(FixedModel(), loader, torch.device("cpu"))
    assert metrics["iou"] == 1.0
    assert metrics["dice"] == 1.0


def test_training_defaults_to_no_pretrained_weights_for_license_cleanliness(tmp_path: Path):
    args = build_parser().parse_args(["--dataset-root", str(tmp_path)])
    assert args.pretrained_backbone is False


def test_training_defaults_enable_progress_and_periodic_checkpoints(tmp_path: Path):
    args = build_parser().parse_args(["--dataset-root", str(tmp_path)])
    assert args.progress_every == 10
    assert args.checkpoint_every == 5


def test_periodic_checkpoint_saves_interval_and_final_epoch():
    assert should_save_periodic_checkpoint(5, 60, 5) is True
    assert should_save_periodic_checkpoint(10, 60, 5) is True
    assert should_save_periodic_checkpoint(7, 60, 5) is False
    assert should_save_periodic_checkpoint(60, 60, 5) is True
    assert should_save_periodic_checkpoint(7, 7, 5) is True
    assert should_save_periodic_checkpoint(5, 60, 0) is False


def test_progress_message_contains_epoch_batch_loss_lr_and_percent():
    message = format_training_progress(
        epoch=2,
        epochs=60,
        batch=10,
        batches=40,
        loss=0.123456,
        lr=3e-4,
        elapsed_s=12.5,
    )
    assert "epoch 2/60" in message
    assert "batch 10/40" in message
    assert "25.0%" in message
    assert "loss 0.1235" in message
    assert "lr 3.00e-04" in message
    assert "12.5s" in message
