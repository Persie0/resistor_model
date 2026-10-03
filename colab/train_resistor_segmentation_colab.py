"""Google Colab recipe for binary whole-resistor segmentation.

Uses torchvision LR-ASPP + MobileNetV3-Large and the public m2
COCO-segmentation release asset. No Ultralytics package is used.

The default recipe trains the architecture from scratch so it does not pull in
ImageNet-derived pretrained weights. Set USE_PRETRAINED_BACKBONE=True only if
you have reviewed the applicable pretrained-weight terms for your use case.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request

MODEL_NAME = "lraspp_mobilenet_v3_large"
DATASET_URL = "https://github.com/Persie0/resistor_model/releases/download/m2/detection_res.v1i.coco-segmentation.zip"
IMAGE_SIZE = 384
EPOCHS = 60
BATCH_SIZE = 16
USE_PRETRAINED_BACKBONE = False
# The source Roboflow project uses both names for resistor instances. Collapse
# them into the single foreground class required by the Android pipeline.
FOREGROUND_CATEGORIES = ("resistor", "res")

WORK = Path("/content/resistor_segmentation")
REPO = WORK / "resistor_model"
ARCHIVE = WORK / "detection_res.v1i.coco-segmentation.zip"
DATASET_ROOT = WORK / "dataset"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "segmentation-m2-lraspp"
ONNX_PATH = RUN_DIR / "resistor_segmenter_lraspp.onnx"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def require_gpu() -> None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            "No NVIDIA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU."
        ) from exc
    print("GPU:", output, flush=True)


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This recipe is intended for Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)
    RUN_DIR.mkdir(parents=True, exist_ok=True)


def clone_repo() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/Persie0/resistor_model.git",
        str(REPO),
    ])


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[segmentation,export]"])


def download_dataset() -> None:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "resistor-model-colab"})
    with urllib.request.urlopen(request) as response, ARCHIVE.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    print(f"Downloaded {ARCHIVE.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)
    shutil.unpack_archive(ARCHIVE, DATASET_ROOT)


def find_dataset_root(root: Path) -> Path:
    if (root / "train" / "_annotations.coco.json").is_file():
        return root
    candidates = [path.parent.parent for path in root.rglob("train/_annotations.coco.json")]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            f"Could not uniquely locate extracted COCO dataset root under {root}: {unique}"
        )
    return unique[0]


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "last.pt"
    return checkpoint if checkpoint.is_file() and checkpoint.stat().st_size > 0 else None


def training_command(dataset_root: Path, run_dir: Path, resume: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "resistor_model.train_segmentation",
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(run_dir),
        "--image-size",
        str(IMAGE_SIZE),
        "--epochs",
        str(EPOCHS),
        "--batch-size",
        str(BATCH_SIZE),
        "--num-workers",
        "2",
    ]
    for category in FOREGROUND_CATEGORIES:
        command.extend(["--category", category])
    if USE_PRETRAINED_BACKBONE:
        command.append("--pretrained-backbone")
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def train_and_export() -> None:
    dataset_root = find_dataset_root(DATASET_ROOT)
    resume = find_resume_checkpoint(RUN_DIR)
    if resume is not None:
        print(f"Resuming from {resume}", flush=True)
    run(training_command(dataset_root, RUN_DIR, resume), cwd=REPO)
    run([
        sys.executable,
        "-m",
        "resistor_model.export_segmentation",
        "--checkpoint",
        str(RUN_DIR / "best.pt"),
        "--output",
        str(ONNX_PATH),
    ], cwd=REPO)
    print(f"Best checkpoint: {RUN_DIR / 'best.pt'}", flush=True)
    print(f"ONNX model: {ONNX_PATH}", flush=True)


def main() -> None:
    require_gpu()
    mount_drive()
    clone_repo()
    install_dependencies()
    download_dataset()
    train_and_export()


if __name__ == "__main__":
    main()
