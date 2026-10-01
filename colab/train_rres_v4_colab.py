"""Google Colab GPU training script for Persie0/resistor_model.

Usage in Colab:
1. Runtime -> Change runtime type -> GPU.
2. Run this script/notebook.
3. Authorize Google Drive when prompted.
4. Paste a GitHub token with read access to Persie0/resistor_scanner.

Training outputs are stored in Google Drive under
``MyDrive/resistor_model/rres-v4-colab``. If ``last.pt`` already exists there,
training automatically resumes from the next epoch. The GitHub token is
requested with getpass and is never written to disk or printed.
"""

from __future__ import annotations

import getpass
import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


# -------------------------
# Settings to tune in Colab
# -------------------------
EPOCHS = 100
BATCH_SIZE = 8              # safe default for a T4; try 12/16 on larger GPUs
OUTPUT_HEIGHT = 128
OUTPUT_WIDTH = 768
SEQUENCE_BINS = 256
BASE_CHANNELS = 48
D_MODEL = 256
TRANSFORMER_LAYERS = 4
TRANSFORMER_HEADS = 8
SLOT_DECODER_LAYERS = 2
CONSISTENCY_WEIGHT = 0.05
SEED = 42

WORK = Path("/content/resistor_training")
MODEL_REPO = WORK / "resistor_model"
DATA_ROOT = WORK / "rres_v4"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "rres-v4-colab"
CONFIG_PATH = WORK / "rres-v4-colab.yaml"
ZIP_PATH = WORK / "resistor-bandnet-rres-v4-colab.zip"


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
    )
    if capture:
        assert result.stdout is not None
        print(result.stdout, flush=True)
        return result.stdout
    return ""


def require_gpu() -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            "No NVIDIA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU."
        ) from exc
    print("GPU:", out, flush=True)


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("Google Drive persistence requires running this script in Google Colab.") from exc

    drive.mount(str(DRIVE_MOUNT), force_remount=False)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Persistent run directory: {RUN_DIR}", flush=True)


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    """Return a usable latest checkpoint from persistent storage, if present."""
    checkpoint = Path(run_dir) / "last.pt"
    if checkpoint.is_file() and checkpoint.stat().st_size > 0:
        return checkpoint
    return None


def clone_training_repo() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if MODEL_REPO.exists():
        shutil.rmtree(MODEL_REPO)
    run([
        "git", "clone", "--depth", "1", "--branch", "main",
        "https://github.com/Persie0/resistor_model.git", str(MODEL_REPO),
    ])


def install_dependencies() -> None:
    # Colab already provides CUDA-enabled PyTorch. Do not replace it with a CPU wheel.
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{MODEL_REPO}[export]"])
    run([sys.executable, "-m", "pip", "install", "-q", "pytest"])


def download_private_dataset(token: str) -> None:
    # Keep the image corpus on local Colab storage for training throughput.
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True)
    zip_path = WORK / "rres.v4i.yolov8.zip"
    url = "https://api.github.com/repos/Persie0/resistor_scanner/contents/rres.v4i.yolov8.zip?ref=tflite"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "resistor-model-colab",
        },
    )
    print("Downloading private annotated dataset...", flush=True)
    with urllib.request.urlopen(request) as response, zip_path.open("wb") as dst:
        shutil.copyfileobj(response, dst)
    print(f"Downloaded {zip_path.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)
    shutil.unpack_archive(zip_path, DATA_ROOT)


def prepare_dataset() -> None:
    manifest = DATA_ROOT / "manifest.jsonl"
    splits = DATA_ROOT / "splits.json"
    run([
        sys.executable, "-m", "resistor_model.tools.import_roboflow_yolo",
        "--root", str(DATA_ROOT),
        "--output", str(manifest),
        "--min-bands", "3",
        "--max-bands", "6",
    ])
    run([
        sys.executable, "-m", "resistor_model.tools.make_splits",
        "--manifest", str(manifest),
        "--output", str(splits),
        "--ratios", "0.70", "0.15", "0.15",
        "--seed", str(SEED),
        "--group-session",
    ])


def build_config(run_dir: Path, resume: Path | None) -> dict:
    """Build the training config with persistent output/resume paths."""
    return {
        "seed": SEED,
        "data": {
            "manifest": str(DATA_ROOT / "manifest.jsonl"),
            "image_root": str(DATA_ROOT),
            "splits_file": str(DATA_ROOT / "splits.json"),
            "output_size": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
            "sequence_bins": SEQUENCE_BINS,
            "num_workers": 2,
            "group_session": True,
            "split_ratios": [0.70, 0.15, 0.15],
        },
        "model": {
            "num_colors": 12,
            "max_bands": 6,
            "base_channels": BASE_CHANNELS,
            "d_model": D_MODEL,
            "transformer_layers": TRANSFORMER_LAYERS,
            "transformer_heads": TRANSFORMER_HEADS,
            "slot_decoder_layers": SLOT_DECODER_LAYERS,
            "dropout": 0.10,
            "use_chromatic_branch": True,
        },
        "train": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": 3e-4,
            "weight_decay": 0.05,
            "warmup_epochs": 5,
            "grad_clip": 1.0,
            "amp": True,
            "ema_decay": 0.999,
            "output_dir": str(run_dir),
            "resume": str(resume) if resume is not None else None,
        },
        "loss": {
            "dense": 1.0,
            "color": 2.0,
            "exist": 0.5,
            "center": 1.0,
            "width": 0.5,
            "order": 0.2,
            "count": 0.2,
            "consistency": CONSISTENCY_WEIGHT,
            "body_class_weight": 0.25,
        },
    }


def write_config() -> None:
    import yaml

    resume = find_resume_checkpoint(RUN_DIR)
    if resume is None:
        print("No Drive checkpoint found; starting a fresh training run.", flush=True)
    else:
        print(f"Resuming from persistent checkpoint: {resume}", flush=True)

    cfg = build_config(RUN_DIR, resume)
    text = yaml.safe_dump(cfg, sort_keys=False)
    CONFIG_PATH.write_text(text, encoding="utf-8")
    # Keep the resolved recipe next to the checkpoints for reproducibility.
    (RUN_DIR / "resolved_config.yaml").write_text(text, encoding="utf-8")
    print(text, flush=True)


def train_and_evaluate() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    # Tests are short and catch accidental branch/API drift before a long GPU run.
    run([sys.executable, "-m", "pytest", "-q"], cwd=MODEL_REPO)
    # Training stdout is inherited, so flushed batch/epoch progress appears live in Colab.
    run([sys.executable, "-u", "-m", "resistor_model.train", "--config", str(CONFIG_PATH)])

    test_output = run([
        sys.executable, "-m", "resistor_model.evaluate",
        "--checkpoint", str(RUN_DIR / "best.pt"),
        "--manifest", str(DATA_ROOT / "manifest.jsonl"),
        "--image-root", str(DATA_ROOT),
        "--split", "test",
    ], capture=True)
    (RUN_DIR / "test_metrics.txt").write_text(test_output, encoding="utf-8")

    run([
        "resistor-export",
        "--checkpoint", str(RUN_DIR / "best.pt"),
        "--output", str(RUN_DIR / "resistor_bandnet.onnx"),
    ])


def summarize() -> None:
    metrics_path = RUN_DIR / "metrics.jsonl"
    raw_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    # If a user manually reruns from an older checkpoint, keep the newest record per epoch.
    rows_by_epoch = {int(row["epoch"]): row for row in raw_rows}
    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    if not rows:
        raise RuntimeError(f"No training metrics found in {metrics_path}")
    best = max(rows, key=lambda r: (r["val"]["exact_sequence_accuracy"], r["val"]["macro_f1"]))
    summary = {
        "epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_validation": best["val"],
        "last_validation": rows[-1]["val"],
        "persistent_run_dir": str(RUN_DIR),
    }
    (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def package_and_download() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    archive_base = ZIP_PATH.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", root_dir=RUN_DIR)
    print(f"Created {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
    try:
        from google.colab import files
        files.download(str(ZIP_PATH))
    except Exception:
        print("Output remains persistently available at", RUN_DIR, flush=True)


def main() -> None:
    require_gpu()
    mount_drive()
    token = getpass.getpass("GitHub token with READ access to Persie0/resistor_scanner: ").strip()
    if not token:
        raise RuntimeError("A GitHub token is required because resistor_scanner is private.")

    clone_training_repo()
    install_dependencies()
    download_private_dataset(token)
    # Avoid retaining the credential longer than needed.
    token = ""
    prepare_dataset()
    write_config()
    train_and_evaluate()
    summarize()
    package_and_download()


if __name__ == "__main__":
    main()
