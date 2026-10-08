"""Google Colab recipe for the v4 one-class resistor box detector.

Trains torchvision SSDLite320 + MobileNetV3-Large directly from the COCO bounding
boxes in the public v4 resistor_sam3_merged.zip release asset. No Ultralytics
package is used.

The default recipe does not download pretrained ImageNet/COCO weights. Set
USE_PRETRAINED_BACKBONE=True only after reviewing the applicable pretrained-weight
terms for your use case.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request

DETECTION_COLAB_VERSION = "2026-10-09-v1"
MODEL_NAME = "ssdlite320_mobilenet_v3_large"
DATASET_URL = (
    "https://github.com/Persie0/resistor_model/releases/download/v4/"
    "resistor_sam3_merged.zip"
)
DATASET_SHA256 = "be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f"
IMAGE_SIZE = 320
EPOCHS = 80
BATCH_SIZE = 16
PROGRESS_EVERY = 10
CHECKPOINT_EVERY = 5
HEARTBEAT_SECONDS = 5
USE_PRETRAINED_BACKBONE = False
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024

WORK = Path("/content/resistor_detector")
REPO = WORK / "resistor_model"
ARCHIVE = WORK / "resistor_sam3_merged.zip"
DATASET_ROOT = WORK / "dataset"
LOCAL_RESUME = WORK / "resume-last.pt"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "detector-v4-ssdlite320"
ONNX_PATH = RUN_DIR / "resistor_detector_ssdlite320.onnx"
ZIP_PATH = WORK / "resistor-detector-v4-ssdlite320.zip"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=str(cwd) if cwd else None, check=True)


def run_with_heartbeat(
    command: list[str],
    *,
    cwd: Path | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
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
                f"[trainer] process alive | waiting for next trainer log line | "
                f"elapsed {now - started:.0f}s",
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
    run([sys.executable, "-m", "pip", "install", "-q", "-e", f"{REPO}[detection,export]"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset() -> None:
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    if ARCHIVE.exists():
        ARCHIVE.unlink()

    request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "resistor-model-colab"})
    started = time.monotonic()
    with urllib.request.urlopen(request) as response, ARCHIVE.open("wb") as destination:
        total = int(response.headers.get("Content-Length") or 0)
        copied = 0
        report_step = max(DOWNLOAD_CHUNK_BYTES, total // 20 if total else DOWNLOAD_CHUNK_BYTES)
        next_report = report_step
        while True:
            chunk = response.read(DOWNLOAD_CHUNK_BYTES)
            if not chunk:
                break
            destination.write(chunk)
            copied += len(chunk)
            if copied >= next_report or (total and copied >= total):
                elapsed = max(0.001, time.monotonic() - started)
                speed = copied / elapsed / 1024 / 1024
                if total:
                    print(
                        f"[download] {copied / 1024 / 1024:.1f}/"
                        f"{total / 1024 / 1024:.1f} MiB "
                        f"({100.0 * copied / total:.0f}%) | {speed:.1f} MiB/s",
                        flush=True,
                    )
                else:
                    print(
                        f"[download] {copied / 1024 / 1024:.1f} MiB | {speed:.1f} MiB/s",
                        flush=True,
                    )
                next_report = copied + report_step

    actual_sha = sha256_file(ARCHIVE)
    if actual_sha.lower() != DATASET_SHA256.lower():
        raise RuntimeError(
            f"v4 dataset SHA-256 mismatch: expected {DATASET_SHA256}, got {actual_sha}"
        )
    print(
        f"[download] complete | {ARCHIVE.stat().st_size / 1024 / 1024:.1f} MiB | checksum OK",
        flush=True,
    )
    print("[dataset] extracting v4 COCO dataset...", flush=True)
    shutil.unpack_archive(ARCHIVE, DATASET_ROOT)
    print("[dataset] extraction complete.", flush=True)


def find_dataset_root(root: Path) -> Path:
    if (root / "train" / "_annotations.coco.json").is_file():
        return root
    candidates = [path.parent.parent for path in root.rglob("train/_annotations.coco.json")]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(
            f"Could not uniquely locate extracted v4 COCO root under {root}: {unique}"
        )
    return unique[0]


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "last.pt"
    return checkpoint if checkpoint.is_file() and checkpoint.stat().st_size > 0 else None


def stage_resume_checkpoint(source: Path | None, destination: Path = LOCAL_RESUME) -> Path | None:
    if source is None:
        if destination.exists():
            destination.unlink()
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        destination.unlink()

    total = source.stat().st_size
    print(
        f"[resume-copy] staging {source} -> {destination} "
        f"({total / 1024 / 1024:.1f} MiB)",
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
                print(
                    f"[resume-copy] {100.0 * copied / max(1, total):.0f}% "
                    f"({copied / 1024 / 1024:.1f} MiB)",
                    flush=True,
                )
                next_report = copied + report_step
    return destination


def training_command(dataset_root: Path, run_dir: Path, resume: Path | None) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "-m",
        "resistor_model.train_detection",
        "--dataset-root",
        str(dataset_root),
        "--output-dir",
        str(run_dir),
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
    print("\nDetector training configuration", flush=True)
    print(f"  model: {MODEL_NAME}", flush=True)
    print("  dataset release: v4 / resistor_sam3_merged.zip", flush=True)
    print(f"  dataset root: {dataset_root}", flush=True)
    print(f"  input: {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print("  classes: background + resistor", flush=True)
    print(f"  epochs: {EPOCHS}", flush=True)
    print(f"  batch size: {BATCH_SIZE}", flush=True)
    print(f"  pretrained backbone: {USE_PRETRAINED_BACKBONE}", flush=True)
    print(f"  progress: every {PROGRESS_EVERY} batches", flush=True)
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
            "resistor_model.export_detection",
            "--checkpoint",
            str(RUN_DIR / "best.pt"),
            "--output",
            str(ONNX_PATH),
        ], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print(
            "ONNX export failed; checkpoints and metrics remain safely in Drive.",
            flush=True,
        )
        return

    print(f"Best checkpoint: {RUN_DIR / 'best.pt'}", flush=True)
    print(f"Latest checkpoint: {RUN_DIR / 'last.pt'}", flush=True)
    print(f"Numbered checkpoints: {RUN_DIR / 'checkpoints'}", flush=True)
    print(f"ONNX model: {ONNX_PATH}", flush=True)
    print(f"ONNX metadata: {ONNX_PATH.with_suffix(ONNX_PATH.suffix + '.json')}", flush=True)


def package_and_download() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(
        f"Packaged detector outputs: {ZIP_PATH} "
        f"({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)",
        flush=True,
    )
    try:
        from google.colab import files

        files.download(str(ZIP_PATH))
    except Exception:
        print(f"Automatic download unavailable. Outputs remain in Drive at {RUN_DIR}", flush=True)


def main() -> None:
    print(f"Detector Colab version: {DETECTION_COLAB_VERSION}", flush=True)
    require_gpu()
    mount_drive()
    clone_repo()
    install_dependencies()
    download_dataset()
    train_and_export()
    package_and_download()


if __name__ == "__main__":
    main()
