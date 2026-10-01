from resistor_model import train


def test_progress_interval_targets_about_ten_updates():
    assert train._progress_interval(1) == 1
    assert train._progress_interval(8) == 1
    assert train._progress_interval(95) == 10
    assert train._progress_interval(100) == 10
    assert train._progress_interval(101) == 11


def test_progress_line_contains_phase_percent_loss_lr_elapsed_and_eta():
    line = train._format_progress(
        phase="train",
        epoch=3,
        epochs=100,
        step=25,
        total_steps=100,
        running_loss=1.23456,
        lr=3e-4,
        elapsed=30.0,
    )
    assert "train" in line
    assert "epoch 3/100" in line
    assert "25/100" in line
    assert "25.0%" in line
    assert "loss 1.2346" in line
    assert "lr 3.000e-04" in line
    assert "elapsed 00:30" in line
    assert "eta 01:30" in line


def test_validation_progress_omits_learning_rate():
    line = train._format_progress(
        phase="val",
        epoch=2,
        epochs=10,
        step=5,
        total_steps=10,
        running_loss=0.75,
        lr=None,
        elapsed=5.0,
    )
    assert "val" in line
    assert "loss 0.7500" in line
    assert "lr " not in line
