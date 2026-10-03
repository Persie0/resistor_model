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
import time
import urllib.request

MODEL_NAME = "lraspp_mobilenet_v3_large"
DATASET_URL = "https://github.com/Persie0/resistor_model/releases/download/m2/detection_res.v1i.coco-segmentation.zip"
IMAGE_SIZE = 384
EPOCHS = 60
BATCH_SIZE = 16
PROGRESS_EVERY = 1
CHECKPOINT_EVERY = 5
HEARTBEAT_SECONDS = 5
USE_PRETRAINED_BACKBONE = False
# The source Roboflow project uses both names for resistor instances. Collapse
# them into the single foreground class required by the Android pipeline.
FOREGROUND_CATEGORIES = ("resistor", "res")

WORK = Path("/content/resistor_segmentation")
REPO = WORK / "resistor_model"
ARCHIVE = WORK / "detection_res.v1i.coco-segmentation.zip"
DATASET_ROOT = WORK / "dataset"
LOCAL_RESUME = WORK / "resume-last.pt"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "segmentation-m2-lraspp"
ONNX_PATH = RUN_DIR / "resistor_segmenter_lraspp.onnx"
ZIP_PATH = WORK / "resistor-segmentation-m2-lraspp.zip"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def run_with_heartbeat(
    command: list[str],
    *,
    cwd: Path | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
    """Run a subprocess while guaranteeing visible Colab output during silent startup."""
    print("+", " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=str(cwd) if cwd else None)
    next_heartbeat = started + max(1.0, heartbeat_seconds)
    while True:
        return_code = process.poll()
        if return_code is not None:
            if return_code != 0:
                raise subprocess.CalledProcessError(return_code, command)
            return
        now = time.monotonic()
        if now >= next_heartbeat:
            print(
                f"[trainer] process alive | waiting for next trainer log line | elapsed {now - started:.0f}s",
                flush=True,
            )
            next_heartbeat = now + max(1.0, heartbeat_seconds)
        time.sleep(0.5)


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
    print(f"Persistent run directory: {RUN_DIR}", flush=True)


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
    print("Extracting dataset...", flush=True)
    shutil.unpack_archive(ARCHIVE, DATASET_ROOT)
    print("Dataset extracted.", flush=True)


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


def stage_resume_checkpoint(source: Path | None, destination: Path = LOCAL_RESUME) -> Path | None:
    """Copy a Drive checkpoint to local Colab storage with visible byte progress."""
    if source is None:
        if destination.exists():
            destination.unlink()
        return None

    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    total = source.stat().st_size
    print(
        f"[resume-copy] staging {source} -> {destination} ({total / 1024 / 1024:.1f} MiB)",
        flush=True,
    )
    copied = 0
    report_step = max(8 * 1024 * 1024, total // 10 if total else 1)
    next_report = report_step
    with source.open("rb") as src, destination.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if copied >= next_report or copied == total:
                percent = 100.0 * copied / max(1, total)
                print(
                    f"[resume-copy] {copied / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB ({percent:.0f}%)",
                    flush=True,
                )
                next_report = copied + report_step
    print("[resume-copy] local checkpoint ready.", flush=True)
    return destination


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
        "--progress-every",
        str(PROGRESS_EVERY),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
    ]
    for category in FOREGROUND_CATEGORIES:
        command.extend(["--category", category])
    if USE_PRETRAINED_BACKBONE:
        command.append("--pretrained-backbone")
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def print_run_configuration(
    dataset_root: Path,
    drive_resume: Path | None,
    local_resume: Path | None,
) -> None:
    print("\nSegmentation training configuration", flush=True)
    print(f"  model: {MODEL_NAME}", flush=True)
    print(f"  dataset: {dataset_root}", flush=True)
    print(f"  input: {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"  epochs: {EPOCHS}", flush=True)
    print(f"  batch size: {BATCH_SIZE}", flush=True)
    print(f"  progress: every {PROGRESS_EVERY} batch", flush=True)
    print(f"  heartbeat: every {HEARTBEAT_SECONDS}s during silent trainer startup", flush=True)
    print(f"  numbered checkpoints: every {CHECKPOINT_EVERY} epochs", flush=True)
    print(f"  persistent outputs: {RUN_DIR}", flush=True)
    if drive_resume is None:
        print("  resume: no checkpoint found; starting fresh", flush=True)
    else:
        print(f"  resume source: {drive_resume}", flush=True)
        print(f"  resume local: {local_resume}", flush=True)
    print("", flush=True)


def train_and_export() -> None:
    dataset_root = find_dataset_root(DATASET_ROOT)
    drive_resume = find_resume_checkpoint(RUN_DIR)
    local_resume = stage_resume_checkpoint(drive_resume)
    print_run_configuration(dataset_root, drive_resume, local_resume)
    run_with_heartbeat(training_command(dataset_root, RUN_DIR, local_resume), cwd=REPO)
    try:
        run([
            sys.executable,
            "-m",
            "resistor_model.export_segmentation",
            "--checkpoint",
            str(RUN_DIR / "best.pt"),
            "--output",
            str(ONNX_PATH),
        ], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print("ONNX export failed; checkpoints and metrics remain safely in Drive.", flush=True)
        return
    print(f"Best checkpoint: {RUN_DIR / 'best.pt'}", flush=True)
    print(f"Latest checkpoint: {RUN_DIR / 'last.pt'}", flush=True)
    print(f"Numbered checkpoints: {RUN_DIR / 'checkpoints'}", flush=True)
    print(f"ONNX model: {ONNX_PATH}", flush=True)


def package_and_download() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    archive_base = ZIP_PATH.with_suffix("")
    shutil.make_archive(str(archive_base), "zip", root_dir=RUN_DIR)
    print(
        f"Packaged run outputs: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)",
        flush=True,
    )
    try:
        from google.colab import files

        files.download(str(ZIP_PATH))
    except Exception:
        print(f"Automatic download unavailable. Outputs remain in Drive at {RUN_DIR}", flush=True)


def main() -> None:
    require_gpu()
    mount_drive()
    clone_repo()
    install_dependencies()
    download_dataset()
    train_and_export()
    package_and_download()


if __name__ == "__main__":
    main()
