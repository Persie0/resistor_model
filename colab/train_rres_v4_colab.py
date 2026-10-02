"""Google Colab GPU training script for the r1 multi-dataset ResistorBandNetV2 run.

Usage in Colab:
1. Runtime -> Change runtime type -> GPU.
2. Run this script/notebook.
3. Authorize Google Drive when prompted.

The three public COCO datasets from the ``r1`` GitHub release are downloaded to
local Colab storage for training throughput. Checkpoints and reports persist in
``MyDrive/resistor_model/r1-v2-colab`` and ``last.pt`` is resumed automatically.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request


EPOCHS = 100
BATCH_SIZE = 8
OUTPUT_HEIGHT = 128
OUTPUT_WIDTH = 768
SEQUENCE_BINS = 256
BASE_CHANNELS = 48
D_MODEL = 256
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 8
SLOT_DECODER_LAYERS = 2
CONSISTENCY_WEIGHT = 0.05
# CTC starts at a much larger raw scale than the primary CE losses; 0.10 keeps it auxiliary.
CTC_WEIGHT = 0.10
KL_WEIGHT = 0.03
LABEL_SMOOTHING = 0.05
SEED = 42

RELEASE_BASE = "https://github.com/Persie0/resistor_model/releases/download/r1"
DATASET_ASSETS = (
    "rres.v4i.coco.zip",
    "resistor.value.training.v8i.coco.zip",
    "Deteksi.Nilai.Resistor.v1i.coco.zip",
)

WORK = Path("/content/resistor_training")
MODEL_REPO = WORK / "resistor_model"
DATA_ROOT = WORK / "r1_datasets"
DOWNLOAD_ROOT = WORK / "downloads"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "r1-v2-colab"
CONFIG_PATH = WORK / "r1-v2-colab.yaml"
ZIP_PATH = WORK / "resistor-bandnet-r1-v2-colab.zip"


def run(cmd: list[str], *, cwd: Path | None = None, capture: bool = False) -> str:
    print("+", " ".join(cmd), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    captured: list[str] = []
    for line in process.stdout:
        print(line, end="", flush=True)
        if capture:
            captured.append(line)
    return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)
    return "".join(captured)


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
    # Colab already provides CUDA-enabled PyTorch; the V2 default backbone is internal.
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{MODEL_REPO}[export]"])
    run([sys.executable, "-m", "pip", "install", "-q", "pytest"])


def dataset_roots() -> list[Path]:
    return [DATA_ROOT / asset.removesuffix(".zip") for asset in DATASET_ASSETS]


def download_release_datasets() -> None:
    """Download and extract all three public r1 COCO datasets to local Colab storage."""
    if DATA_ROOT.exists():
        shutil.rmtree(DATA_ROOT)
    DATA_ROOT.mkdir(parents=True)
    DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    for asset, root in zip(DATASET_ASSETS, dataset_roots(), strict=True):
        archive = DOWNLOAD_ROOT / asset
        url = f"{RELEASE_BASE}/{asset}"
        print(f"Downloading {asset} ...", flush=True)
        request = urllib.request.Request(url, headers={"User-Agent": "resistor-model-colab"})
        with urllib.request.urlopen(request) as response, archive.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        print(f"Downloaded {asset}: {archive.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)
        root.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(archive, root)


def prepare_dataset() -> None:
    manifest = DATA_ROOT / "manifest.jsonl"
    splits = DATA_ROOT / "splits.json"
    report = DATA_ROOT / "import-report.json"
    command = [sys.executable, "-m", "resistor_model.tools.import_roboflow_coco"]
    for root in dataset_roots():
        command.extend(["--root", str(root)])
    command.extend([
        "--output", str(manifest),
        "--report", str(report),
        "--min-bands", "3",
        "--max-bands", "6",
    ])
    run(command)
    run([
        sys.executable, "-m", "resistor_model.tools.make_splits",
        "--manifest", str(manifest),
        "--output", str(splits),
        "--ratios", "0.70", "0.15", "0.15",
        "--seed", str(SEED),
        "--group-session",
    ])


def build_config(run_dir: Path, resume: Path | None) -> dict:
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
            "geometric_augment": True,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "jitter_strength": 1.0,
        },
        "model": {
            "architecture": "v2",
            "backbone": "convnext_lite",
            "pretrained": False,
            "num_colors": 12,
            "max_bands": 6,
            "base_channels": BASE_CHANNELS,
            "d_model": D_MODEL,
            "transformer_layers": TRANSFORMER_LAYERS,
            "transformer_heads": TRANSFORMER_HEADS,
            "slot_decoder_layers": SLOT_DECODER_LAYERS,
            "dropout": 0.10,
            "drop_path": 0.10,
            "conv_kernel": 7,
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
            "ctc": CTC_WEIGHT,
            "kl": KL_WEIGHT,
            "label_smoothing": LABEL_SMOOTHING,
            "body_class_weight": 0.25,
            "color_balance": "sqrt_inverse",
            "max_color_weight": 4.0,
        },
        "eval": {
            "extra_decoders": True,
            # E-series preference is useful for analysis but intentionally not a model-selection bias.
            "series_bonus": 0.0,
        },
    }


def write_config() -> None:
    import yaml

    resume = find_resume_checkpoint(RUN_DIR)
    if resume is None:
        print("No r1 V2 Drive checkpoint found; starting a fresh training run.", flush=True)
    else:
        print(f"Resuming from persistent checkpoint: {resume}", flush=True)
    cfg = build_config(RUN_DIR, resume)
    text = yaml.safe_dump(cfg, sort_keys=False)
    CONFIG_PATH.write_text(text, encoding="utf-8")
    (RUN_DIR / "resolved_config.yaml").write_text(text, encoding="utf-8")
    print(text, flush=True)


def train_and_evaluate() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    run([sys.executable, "-m", "pytest", "-q"], cwd=MODEL_REPO)
    run([sys.executable, "-u", "-m", "resistor_model.train", "--config", str(CONFIG_PATH)])

    test_output = run([
        sys.executable, "-m", "resistor_model.evaluate",
        "--checkpoint", str(RUN_DIR / "best.pt"),
        "--manifest", str(DATA_ROOT / "manifest.jsonl"),
        "--image-root", str(DATA_ROOT),
        "--split", "test",
    ], capture=True)
    (RUN_DIR / "test_metrics.txt").write_text(test_output, encoding="utf-8")

    try:
        run([
            "resistor-export",
            "--checkpoint", str(RUN_DIR / "best.pt"),
            "--output", str(RUN_DIR / "resistor_bandnet.onnx"),
        ])
    except subprocess.CalledProcessError as exc:
        # A converter issue must not discard a completed GPU training run.
        (RUN_DIR / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print("ONNX export failed; checkpoints and metrics remain saved in Drive.", flush=True)


def summarize() -> None:
    metrics_path = RUN_DIR / "metrics.jsonl"
    raw_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    rows_by_epoch = {int(row["epoch"]): row for row in raw_rows}
    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    if not rows:
        raise RuntimeError(f"No training metrics found in {metrics_path}")
    best = max(rows, key=lambda row: (row["val"]["exact_sequence_accuracy"], row["val"]["macro_f1"]))
    summary = {
        "epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_validation": best["val"],
        "last_validation": rows[-1]["val"],
        "persistent_run_dir": str(RUN_DIR),
        "datasets": list(DATASET_ASSETS),
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
    clone_training_repo()
    install_dependencies()
    download_release_datasets()
    prepare_dataset()
    write_config()
    train_and_evaluate()
    summarize()
    package_and_download()


if __name__ == "__main__":
    main()
