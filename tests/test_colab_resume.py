from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import subprocess


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


def test_run_streams_child_stdout_line_by_line(monkeypatch, capsys):
    class FakeProcess:
        def __init__(self):
            self.stdout = iter(["first line\n", "second line\n"])
            self.returncode = 0

        def wait(self):
            return self.returncode

    calls = []

    def fake_popen(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return FakeProcess()

    def forbidden_run(*args, **kwargs):
        raise AssertionError("Colab wrapper must stream with Popen, not subprocess.run")

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "run", forbidden_run)

    output = colab.run(["python", "-m", "example"], capture=True)

    visible = capsys.readouterr().out
    assert "+ python -m example" in visible
    assert "first line" in visible
    assert "second line" in visible
    assert output == "first line\nsecond line\n"
    assert calls
    _, kwargs = calls[0]
    assert kwargs["stdout"] is subprocess.PIPE
    assert kwargs["stderr"] is subprocess.STDOUT
    assert kwargs["text"] is True
    assert kwargs["bufsize"] == 1
