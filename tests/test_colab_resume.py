from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "colab" / "train_rres_v4_colab.py"
spec = spec_from_file_location("train_rres_v4_colab", SCRIPT)
assert spec is not None and spec.loader is not None
colab = module_from_spec(spec)
spec.loader.exec_module(colab)


def test_find_resume_checkpoint_prefers_drive_last_checkpoint(tmp_path: Path):
    run_dir = tmp_path / "drive-run"
    run_dir.mkdir()
    assert colab.find_resume_checkpoint(run_dir) is None

    checkpoint = run_dir / "last.pt"
    checkpoint.write_bytes(b"checkpoint")
    assert colab.find_resume_checkpoint(run_dir) == checkpoint


def test_build_config_persists_outputs_and_resume_to_drive(tmp_path: Path):
    run_dir = tmp_path / "drive-run"
    checkpoint = run_dir / "last.pt"
    cfg = colab.build_config(run_dir, checkpoint)

    assert cfg["train"]["output_dir"] == str(run_dir)
    assert cfg["train"]["resume"] == str(checkpoint)
    assert cfg["train"]["epochs"] == colab.EPOCHS
    assert cfg["train"]["amp"] is True


def test_build_config_starts_fresh_when_drive_has_no_checkpoint(tmp_path: Path):
    cfg = colab.build_config(tmp_path / "drive-run", None)
    assert cfg["train"]["resume"] is None
