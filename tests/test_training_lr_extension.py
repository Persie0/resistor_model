"""Regression tests: extend a completed 60-epoch Mobile1D run to 120 safely."""

import math

import pytest
import torch

from resistor_model.train import (
    _align_resumed_extension_learning_rate,
    _learning_rate_multiplier,
)


def factor(epoch: int, *, total: int = 120) -> float:
    return _learning_rate_multiplier(
        epoch, epochs=total, warmup=3,
        initial_cycle_epochs=60, extension_peak_factor=0.2,
    )


def test_first_60_epochs_keep_exact_original_cosine_learning_rate():
    for epoch in range(60):
        if epoch < 3:
            old = (epoch + 1) / 3
        else:
            old = 0.5 * (1 + math.cos(math.pi * (epoch - 3) / 57))
        assert factor(epoch) == pytest.approx(old, abs=1e-14)


def test_120_epoch_low_lr_extension_is_bounded_and_cosine_decays():
    assert factor(0) == pytest.approx(1 / 3)
    assert factor(2) == pytest.approx(1.0)
    assert factor(59) < 0.001
    assert factor(60) == pytest.approx(0.2)
    assert factor(90) == pytest.approx(0.1)
    assert factor(119) < 0.001
    assert factor(120) == pytest.approx(0.0, abs=1e-12)
    assert all(0 <= factor(e) <= 0.2 for e in range(60, 121))


def test_existing_training_configs_keep_original_single_cycle_schedule():
    for e in range(60):
        expected = ((e + 1) / 3 if e < 3
                    else 0.5 * (1 + math.cos(math.pi * (e - 3) / 57)))
        actual = _learning_rate_multiplier(e, epochs=60, warmup=3)
        assert actual == pytest.approx(expected)


def test_resume_after_finished_60_epochs_enters_phase_2_without_zero_lr():
    param = torch.nn.Parameter(torch.tensor(1.0))
    old_opt = torch.optim.SGD([param], lr=5e-4)
    old_sched = torch.optim.lr_scheduler.LambdaLR(
        old_opt, lambda e: _learning_rate_multiplier(e, epochs=60, warmup=3)
    )
    for _ in range(60):
        old_opt.step()
        old_sched.step()
    assert old_sched.last_epoch == 60
    assert old_opt.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)

    resumed_opt = torch.optim.SGD([param], lr=5e-4)
    resumed_sched = torch.optim.lr_scheduler.LambdaLR(resumed_opt, factor)
    resumed_opt.load_state_dict(old_opt.state_dict())
    resumed_sched.load_state_dict(old_sched.state_dict())
    cfg = {"train": {"epochs": 120, "warmup_epochs": 3,
                     "lr_initial_cycle_epochs": 60,
                     "lr_extension_peak_factor": 0.2}}
    old_checkpoint = {"config": {"train": {"epochs": 60}}}
    assert _align_resumed_extension_learning_rate(
        optimizer=resumed_opt, scheduler=resumed_sched,
        checkpoint=old_checkpoint, start_epoch=60, config=cfg,
    )
    assert resumed_opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    resumed_opt.step()
    resumed_sched.step()
    assert resumed_opt.param_groups[0]["lr"] == pytest.approx(5e-4 * factor(61))


def test_normal_resume_before_epoch_60_is_not_modified():
    param = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([param], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, factor)
    cfg = {"train": {"epochs": 120, "warmup_epochs": 3,
                     "lr_initial_cycle_epochs": 60,
                     "lr_extension_peak_factor": 0.2}}
    checkpoint = {"config": {"train": {"epochs": 60}}}
    lr = optimizer.param_groups[0]["lr"]
    assert not _align_resumed_extension_learning_rate(
        optimizer=optimizer, scheduler=scheduler, checkpoint=checkpoint,
        start_epoch=45, config=cfg,
    )
    assert optimizer.param_groups[0]["lr"] == lr


@pytest.mark.parametrize("initial,peak", [(0, 0.2), (120, 0.2), (60, 0.0), (60, 1.5)])
def test_invalid_schedule_config_rejected(initial, peak):
    with pytest.raises(ValueError):
        _learning_rate_multiplier(
            1 if initial != 60 else 60,
            epochs=120, warmup=3,
            initial_cycle_epochs=initial,
            extension_peak_factor=peak,
        )
