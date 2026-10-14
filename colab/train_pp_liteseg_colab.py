"""Google Colab recipe: PP-LiteSeg-T whole-resistor segmentation (Apache-2.0).

Why this exists alongside ``train_resistor_segmentation_colab.py`` (torchvision
LR-ASPP): PP-LiteSeg-T (STDC1 backbone + detail/attention decoder) is the best
published speed/accuracy trade-off in the license-clean set, and its
boundary-aware design suits thin resistor bodies. Single resistor per frame,
so plain binary semantic segmentation is all the phone pipeline needs.

Everything is Apache-2.0 / BSD: PaddleSeg + PaddlePaddle (Apache-2.0),
STDC/ImageNet-derived pretrained weights via PaddleSeg's own hosting
(review their terms for your use case, same as any pretrained weight).
No Ultralytics code, models or weights anywhere in this path.

Data: same public v4 SAM 3.1 archive as the LR-ASPP recipe, plus offline
body-color synthesis (blue/green/random recolors of exact SAM masks, train
split only) so ANY body color is detected, not just the dominant brown.
Strong online color jitter is label-preserving for body-vs-background.

Usage in Colab:
1. Runtime -> Change runtime type -> GPU.
2. Run all cells. 3. Authorize Google Drive when prompted.

Outputs in ``MyDrive/resistor_model/segmentation-pp-liteseg``: PaddleSeg
output dir (best_model by val mIoU), exported inference model, ONNX,
best-effort TFLite, hue-bucketed test report (IoU per body color), packaging.
Resume: PaddleSeg ``iter_*`` checkpoints under the Drive output dir are picked
up automatically via ``--resume_model``.
"""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile

PADDLESEG_COLAB_VERSION = "2026-10-14-v14-py312-conversion-preview-fix"
PADDLESEG_REPO = "https://github.com/PaddlePaddle/PaddleSeg.git"
PADDLESEG_BRANCH = "release/2.10"
# NOTE: `paddlepaddle-gpu` on default PyPI stops at 2.6.x (no Python 3.13
# wheels), so the GPU build MUST come from Paddle's official index below.
# cu126 matches Colab's CUDA 12.x runtimes; cu118 is the fallback.
PADDLE_GPU_INDEXES = (
    "https://www.paddlepaddle.org.cn/packages/stable/cu126/",
    "https://www.paddlepaddle.org.cn/packages/stable/cu118/",
)
PADDLE_GPU_PIN = "3.1.1"  # known-present on the stable indexes; fallback is unpinned latest
# Same v4 SAM-mask source as the LR-ASPP recipe.
DATASET_URL = "https://github.com/Persie0/resistor_model/releases/download/v4/resistor_sam3_merged.zip"
DATASET_SHA256 = "be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f"
# Real blue-resistor bodies (SAM 3.1, same layout); merged as blue_* into the v4 tree.
BLUE_DATASET_URL = "https://github.com/Persie0/resistor_model/releases/download/ee/blue_resistors_sam3.zip"
BLUE_DATASET_SHA256 = "800bd3aef16f1fbd2e98adbc2ba3c99551cbaf0f9b4c772614fa01909ceb733e"
MODEL_TYPE_NEEDLE = "liteseg"  # parsed stock config must be a PP-LiteSeg model
IMAGE_SIZE = 512
# ~20 epochs at bs48 (19352 train pairs -> ~403 iters/epoch).
# Stop at 8k and retain whichever validation checkpoint has the best mIoU.
ITERS = 8000
# Sized for a 15 GB T4: activations scale ~linearly, bs16 ~= 3.4 GB reserved,
# so bs48 lands ~9-10 GB fp32 (~6-7 GB with AMP below). If you ever hit OOM
# (e.g. smaller GPU), drop to 32. LR stays at the stock 0.01: larger batches
# at the same LR are safe (cleaner gradients); scaling LR up risks spikes.
BATCH_SIZE = 48
# PaddleSeg AMP (fp16/O1): the actual speedup on a compute-bound T4 (~1.5x).
# Eval stays fp32. If loss ever goes NaN under AMP, drop this to fp32.
PRECISION = "fp16"
AMP_LEVEL = "O1"
# Data loading was ~45% of batch_cost at bs48: 6 workers feed the GPU.
NUM_WORKERS = 6
# ResizeStepScaling up to 2.0x dominates CPU aug cost (upscale-then-crop per
# sample). Our frames already vary in distance/size, so cap at 1.5x: keeps
# scale invariance, cuts worst-case aug pixels ~2x. Transforms are runtime
# config, safe to change across a resume.
AUG_MIN_SCALE = 0.5
AUG_MAX_SCALE = 1.5
SAVE_INTERVAL = 1000
# PaddleSeg only writes checkpoints locally; mirror them to Drive on this
# cadence so a Colab disconnect loses minutes, not hours (like last.pt/best.pt
# in the other recipes). Final sync still happens after training.
CHECKPOINT_SYNC_SECONDS = 600
HEARTBEAT_SECONDS = 30  # nudge printed when the trainer is silent this long
NUM_CLASSES = 2  # background + resistor body
# Train-only synthesis; valid/test stay real so metrics measure reality.
SYNTH_VARIANTS = ("blue", "green", "random")
SYNTH_SEED = 7

WORK = Path("/content/pp_liteseg")
RESISTOR_REPO = WORK / "resistor_model"
SYNTH_SCRIPT = RESISTOR_REPO / "src" / "resistor_model" / "tools" / "synthesize_body_colors.py"
MERGE_SCRIPT = RESISTOR_REPO / "src" / "resistor_model" / "tools" / "merge_extra_masks.py"
EVAL_SCRIPT = RESISTOR_REPO / "src" / "resistor_model" / "tools" / "eval_color_buckets.py"
PADDLESEG_DIR = WORK / "PaddleSeg"
ARCHIVE = WORK / "resistor_sam3_merged.zip"
BLUE_ARCHIVE = WORK / "blue_resistors_sam3.zip"
DATASET_ROOT = WORK / "dataset"
OUTPUT_LOCAL = WORK / "output"
CUSTOM_CONFIG = WORK / "pp_liteseg_resistor_512.yml"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "segmentation-pp-liteseg"
OUTPUT_DRIVE = RUN_DIR / "output"
# Legacy (v11) per-file directory cache, left untouched for recovery.
DRIVE_DATASET = RUN_DIR / "dataset"
# The new cache is a single archive on Drive, extracted on Colab's local SSD.
DRIVE_PREPARED_ARCHIVE = RUN_DIR / "prepared_dataset.zip"
DRIVE_PREPARED_MANIFEST = RUN_DIR / "prepared_dataset.manifest.json"
DRIVE_ARCHIVE_V4 = RUN_DIR / "archives" / "resistor_sam3_merged.zip"
DRIVE_ARCHIVE_BLUE = RUN_DIR / "archives" / "blue_resistors_sam3.zip"
DATASET_FINGERPRINT = {
    "recipe": "segmentation-pp-liteseg-v1",
    "v4_sha256": DATASET_SHA256,
    "blue_sha256": BLUE_DATASET_SHA256,
    "synth_variants": list(SYNTH_VARIANTS),
    "synth_seed": SYNTH_SEED,
    "masks01": 1,
}
ZIP_PATH = WORK / "resistor-segmentation-pp-liteseg.zip"
CONVERTER_VENV = WORK / "onnx-converter-py312"
DRIVE_INFERENCE = RUN_DIR / "paddle_inference"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    """Run a short command, streaming its output line by line."""
    print("+", " ".join(map(str, cmd)), flush=True)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="" if line.endswith("\n") else "\n", flush=True)
    if process.wait():
        raise subprocess.CalledProcessError(process.returncode, cmd)


def run_with_heartbeat(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
    extra_env: dict[str, str] | None = None,
) -> None:
    """Stream a long command (training); heartbeat while the child is silent.

    A subprocess inheriting the kernel's native file descriptors can train for
    hours while *none* of its prints reach the notebook cell. This forwards
    every line through the parent Python process and prints a keep-alive plus
    process health whenever the child goes quiet.
    """
    print("+", " ".join(map(str, cmd)), flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update(extra_env)
    process = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None

    events: queue.Queue[str | None] = queue.Queue()

    def pump_output() -> None:
        try:
            for line in process.stdout:
                events.put(line)
        finally:
            events.put(None)

    reader = threading.Thread(target=pump_output, name="pplite-trainer-log-forwarder", daemon=True)
    reader.start()
    started = time.monotonic()
    heartbeat = max(0.1, float(heartbeat_seconds))
    next_heartbeat = started + heartbeat
    next_health = started + 300.0

    try:
        while True:
            now = time.monotonic()
            try:
                item = events.get(timeout=max(0.01, min(0.25, next_heartbeat - now)))
            except queue.Empty:
                item = ""
            if item is None:
                break
            if item:
                print(item, end="" if item.endswith("\n") else "\n", flush=True)
                next_heartbeat = time.monotonic() + heartbeat
            now = time.monotonic()
            if now >= next_heartbeat:
                print(
                    f"[trainer] process pid={process.pid} alive | "
                    f"no child output for {heartbeat:.0f}s | elapsed {now - started:.0f}s",
                    flush=True,
                )
                next_heartbeat = now + heartbeat
            if now >= next_health:
                try:
                    status = Path(f"/proc/{process.pid}/status").read_text(encoding="utf-8")
                    selected = [
                        line.strip() for line in status.splitlines()
                        if line.startswith(("State:", "VmRSS:", "Threads:"))
                    ]
                    print(f"[trainer] process health: {' | '.join(selected)}", flush=True)
                except OSError:
                    pass
                next_health = now + 300.0
        if process.wait():
            raise subprocess.CalledProcessError(process.returncode, cmd)
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        process.stdout.close()
        reader.join(timeout=1)


def require_gpu() -> None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError("No NVIDIA GPU detected. In Colab choose Runtime -> Change runtime type -> GPU.") from exc
    print("GPU:", out, flush=True)


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This recipe is intended for Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Persistent run directory: {RUN_DIR}", flush=True)


def clone_repos() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if RESISTOR_REPO.exists():
        shutil.rmtree(RESISTOR_REPO)
    run(["git", "clone", "--depth", "1", "https://github.com/Persie0/resistor_model.git", str(RESISTOR_REPO)])
    if PADDLESEG_DIR.exists():
        shutil.rmtree(PADDLESEG_DIR)
    run(["git", "clone", "--depth", "1", "--branch", PADDLESEG_BRANCH, PADDLESEG_REPO, str(PADDLESEG_DIR)])


def _pip_install(args: list[str], *, description: str) -> None:
    """pip install with the failing tail printed (pip -q hides the reason)."""
    print("+", sys.executable, "-m pip install", " ".join(args), flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", *args],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        tail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()[-25:]
        print(f"[pip] {description} failed:", flush=True)
        print("\n".join(tail), flush=True)
        raise subprocess.CalledProcessError(proc.returncode, args)


def install_paddle() -> None:
    print(f"[paddle] python {sys.version.split()[0]}", flush=True)
    if sys.version_info >= (3, 14):
        raise RuntimeError(
            "PaddlePaddle publishes no wheels for Python >= 3.14; "
            "use a Colab runtime with Python 3.11-3.13."
        )
    attempts = [
        [f"paddlepaddle-gpu=={PADDLE_GPU_PIN}", "-i", PADDLE_GPU_INDEXES[0]],
        ["paddlepaddle-gpu", "-i", PADDLE_GPU_INDEXES[0]],
        ["paddlepaddle-gpu", "-i", PADDLE_GPU_INDEXES[1]],
    ]
    last_error: Exception | None = None
    print("[paddle] downloading the ~1 GB GPU wheel; this takes a few minutes, progress below", flush=True)
    for attempt in attempts:
        started = time.monotonic()
        try:
            # No -q here and streamed (not captured): pip's download progress
            # bar is the only sign of life during this multi-minute step.
            # Generous timeout/retries: the official index can be slow but
            # alive; pip's defaults abort too eagerly on it.
            run_with_heartbeat(
                [sys.executable, "-m", "pip", "install",
                 "--timeout", "120", "--retries", "10", *attempt],
                heartbeat_seconds=120,
            )
            print(f"[paddle] installed in {time.monotonic() - started:.0f}s", flush=True)
            break
        except subprocess.CalledProcessError as exc:
            last_error = exc
            print("[paddle] attempt failed, trying next index/version...", flush=True)
    else:
        raise RuntimeError("Could not install paddlepaddle-gpu from any official index") from last_error
    import paddle  # noqa: E402  (verify install before the long training run)

    print(f"paddle {paddle.__version__} | cuda: {paddle.device.cuda.device_count()} device(s)", flush=True)


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    install_paddle()
    # PaddleSeg requirements pinned `opencv-python <= 4.6.0`, which has no
    # Python 3.13 wheel: skip it and use Colab's preinstalled cv2 instead.
    req_lines = (PADDLESEG_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
    filtered = [line for line in req_lines if line.strip() and "opencv" not in line.lower()]
    filtered_req = WORK / "paddleseg-requirements-filtered.txt"
    filtered_req.write_text("\n".join(filtered) + "\n", encoding="utf-8")
    _pip_install(["-q", "-r", str(filtered_req)], description="paddleseg requirements")
    try:
        import cv2  # noqa: E402  (must exist: preinstalled or via requirements)

        print(f"[deps] cv2 {cv2.__version__} OK", flush=True)
    except ImportError as exc:
        raise RuntimeError("No OpenCV available and the pinned one has no wheel for this Python") from exc


def pip_best_effort(args: list[str], *, description: str) -> bool:
    """Install optional conversion deps without ever failing the training run."""
    try:
        _pip_install(["-q", *args], description=description)
        return True
    except subprocess.CalledProcessError:
        print(f"[deps] optional install failed, continuing without it: {description}", flush=True)
        return False


def archive_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_dataset_root(root: Path) -> Path:
    if (root / "train" / "_annotations.coco.json").is_file():
        return root
    candidates = [p.parent.parent for p in root.rglob("train/_annotations.coco.json")]
    unique = sorted(set(candidates))
    if len(unique) != 1:
        raise RuntimeError(f"Could not uniquely locate dataset root under {root}: {unique}")
    return unique[0]


def download_with_progress(url: str, destination: Path, *, label: str) -> None:
    """Download with MiB progress lines (multi-hundred-MiB archives, like other colabs)."""
    request = urllib.request.Request(url, headers={"User-Agent": "resistor-model-colab"})
    hasher = hashlib.sha256()
    downloaded = 0
    print(f"[{label}] downloading...", flush=True)
    with urllib.request.urlopen(request) as response, destination.open("wb") as out:
        total = int(response.headers.get("Content-Length") or 0)
        last_report = 0
        while True:
            chunk = response.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            hasher.update(chunk)
            downloaded += len(chunk)
            if downloaded - last_report >= 32 * 1024 * 1024 or (total and downloaded >= total):
                progress = f" ({100 * downloaded / total:.0f}%)" if total else ""
                print(f"[{label}] downloaded {downloaded / 1024 / 1024:.1f} MiB{progress}", flush=True)
                last_report = downloaded
    print(f"[{label}] done | {downloaded / 1024 / 1024:.1f} MiB | sha256 {hasher.hexdigest()[:16]}...", flush=True)


def download_dataset() -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    dataset_cache.fetch_cached_archive(
        local=ARCHIVE, drive_copy=DRIVE_ARCHIVE_V4,
        url=DATASET_URL, sha256=DATASET_SHA256, label="dataset",
    )
    root = find_dataset_root(DATASET_ROOT) if (DATASET_ROOT / "train").exists() else None
    if root is None:
        if DATASET_ROOT.exists():
            shutil.rmtree(DATASET_ROOT)
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(ARCHIVE, DATASET_ROOT)
        root = find_dataset_root(DATASET_ROOT)
    print(f"[dataset] ready: {root}", flush=True)
    return root


def merge_blue_dataset(dataset_root: Path) -> None:
    """Merge real blue-resistor pairs (archives cached in Drive). Standalone script: stdlib+PIL only."""
    WORK.mkdir(parents=True, exist_ok=True)
    dataset_cache.fetch_cached_archive(
        local=BLUE_ARCHIVE, drive_copy=DRIVE_ARCHIVE_BLUE,
        url=BLUE_DATASET_URL, sha256=BLUE_DATASET_SHA256, label="blue",
    )
    run([
        sys.executable, str(MERGE_SCRIPT),
        "--dataset-root", str(dataset_root),
        "--extra-archive", str(BLUE_ARCHIVE),
        "--extra-sha256", BLUE_DATASET_SHA256,
        "--splits", "train", "valid", "test",
        "--prefix", "blue",
        "--progress-every", "200",
    ])


def synthesize_body_colors(dataset_root: Path) -> None:
    """Recolor train bodies (masks unchanged). Standalone script: cv2+numpy only."""
    if not SYNTH_VARIANTS:
        print("[synth] disabled", flush=True)
        return
    run([
        sys.executable, str(SYNTH_SCRIPT),
        "--dataset-root", str(dataset_root),
        "--splits", "train",
        "--variants", *SYNTH_VARIANTS,
        "--seed", str(SYNTH_SEED),
        "--progress-every", "500",
    ])


def normalize_masks_01(dataset_root: Path) -> None:
    """Rewrite masks to 0/1 class indices in place (idempotent).

    SAM PNGs use 0/255, but PaddleSeg reads labels as class indices where 255
    means *ignore*: without this the foreground would train as background.
    The PyTorch loader and all repo tools accept 0/1 masks unchanged.
    """
    import cv2

    for split in ("train", "valid", "test"):
        masks = sorted((dataset_root / split / "masks_semantic").glob("*.png"))
        converted = 0
        for done, path in enumerate(masks, start=1):
            if done % 500 == 0 or done == len(masks):
                print(f"[masks01] {split}: {done}/{len(masks)}...", flush=True)
            arr = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if arr is None:
                raise RuntimeError(f"unreadable mask: {path}")
            if int(arr.max()) > 1:
                cv2.imwrite(str(path), (arr > 0).astype("uint8"))
                converted += 1
        print(f"[masks01] {split}: {converted} rewritten to 0/1, {len(masks) - converted} already 0/1", flush=True)


def write_paddleseg_lists(dataset_root: Path) -> None:
    """PaddleSeg Dataset lists: '<image> <mask>' paths relative to dataset root."""
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    for split in ("train", "valid"):
        images, masks = dataset_root / split / "images", dataset_root / split / "masks_semantic"
        lines = []
        for mask in sorted(masks.glob("*.png")):
            img = next((f"images/{mask.stem}{e}" for e in exts if (images / f"{mask.stem}{e}").is_file()), None)
            if img is not None:
                lines.append(f"{split}/{img} {split}/masks_semantic/{mask.name}")
        if not lines:
            raise RuntimeError(f"no image/mask pairs for PaddleSeg list in {split}")
        (dataset_root / f"{split}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"[lists] {split}.txt: {len(lines)} pairs (incl. synthesized train bodies)", flush=True)


def find_base_config() -> Path:
    """Prefer the 512px PP-LiteSeg quick-start; fall back to a Cityscapes STDC1 config."""
    quick = PADDLESEG_DIR / "configs" / "quick_start" / "pp_liteseg_optic_disc_512x512_1k.yml"
    if quick.is_file():
        return quick
    candidates = sorted((PADDLESEG_DIR / "configs" / "pp_liteseg").glob("*.yml"))
    stdc1 = [p for p in candidates if "stdc1" in p.name.lower() and "1024x512" in p.name]
    if stdc1:
        return stdc1[0]
    raise RuntimeError(
        f"No PP-LiteSeg base config found; contents: {[p.name for p in candidates]}. "
        "PaddleSeg layout may have changed; pick a configs/pp_liteseg/*.yml manually."
    )


def _cap_train_scale_aug(cfg: dict) -> None:
    """Cap ResizeStepScaling range (pure-dict helper, unit-tested)."""
    transforms = cfg.get("train_dataset", {}).get("transforms", [])
    for tr in transforms:
        if isinstance(tr, dict) and tr.get("type") == "ResizeStepScaling":
            tr["min_scale_factor"] = AUG_MIN_SCALE
            tr["max_scale_factor"] = AUG_MAX_SCALE


def build_custom_config(dataset_root: Path) -> Path:
    import yaml

    base = find_base_config()
    cfg = yaml.safe_load(base.read_text(encoding="utf-8"))
    model_type = str(cfg.get("model", {}).get("type", ""))
    if MODEL_TYPE_NEEDLE not in model_type.lower():
        raise RuntimeError(f"Base config {base.name} is not PP-LiteSeg (model.type={model_type!r})")
    # Binary head; KEEP the stock `pretrained` URL so Cityscapes/optic-disc
    # PP-LiteSeg weights transfer (backbone+decoder init, head replaced).
    cfg["model"]["num_classes"] = NUM_CLASSES
    for key in ("train_dataset", "val_dataset"):
        section = cfg.get(key, {})
        section["dataset_root"] = str(dataset_root)
        section["num_classes"] = NUM_CLASSES
    cfg["train_dataset"]["train_path"] = str(dataset_root / "train.txt")
    cfg["val_dataset"]["val_path"] = str(dataset_root / "valid.txt")
    cfg["batch_size"] = BATCH_SIZE
    cfg["iters"] = ITERS
    _cap_train_scale_aug(cfg)
    CUSTOM_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[config] {base.name} -> {CUSTOM_CONFIG} | iters={ITERS} batch={BATCH_SIZE}", flush=True)
    backbone_pretrained = cfg["model"].get("backbone", {}).get("pretrained")
    print(f"[config] transfer init: model.pretrained={cfg['model'].get('pretrained')} "
          f"backbone.pretrained={backbone_pretrained}", flush=True)
    if not cfg["model"].get("pretrained") and not backbone_pretrained:
        print("[config] WARNING: no pretrained weights in base config; backbone trains from scratch", flush=True)
    return CUSTOM_CONFIG


def latest_resume_checkpoint() -> str | None:
    """Newest PaddleSeg iter checkpoint, preferring the Drive copy (cross-session resume)."""
    roots = [OUTPUT_DRIVE, OUTPUT_LOCAL]
    best: tuple[int, str] | None = None
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.glob("iter_*"):
            pdparams = child / "model.pdparams"
            if pdparams.is_file():
                try:
                    step = int(child.name.split("_")[1])
                except (IndexError, ValueError):
                    continue
                if best is None or step > best[0]:
                    best = (step, str(child))
    if best is None:
        print("[resume] no PaddleSeg checkpoint; fresh run", flush=True)
        return None
    print(f"[resume] resuming from {best[1]}", flush=True)
    return best[1]


def print_run_configuration(dataset_root: Path, resume: str | None, dataset_source: str) -> None:
    print("\nPP-LiteSeg training configuration", flush=True)
    print("  model: PP-LiteSeg (STDC backbone from base config), binary background-vs-body", flush=True)
    print("  release: v4 / resistor_sam3_merged.zip + ee / blue_resistors_sam3.zip (blue_* stems)", flush=True)
    print("  mask source: train/valid/test/masks_semantic/*.png", flush=True)
    print(f"  dataset: {dataset_root} ({dataset_source})", flush=True)
    for split in ("train", "valid", "test"):
        pairs = len(list((dataset_root / split / "masks_semantic").glob("*.png")))
        print(f"  pairs {split}: {pairs}", flush=True)
    print(f"  input: {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"  iters: {ITERS} | batch size: {BATCH_SIZE} | eval+save every {SAVE_INTERVAL}", flush=True)
    print(f"  precision: {PRECISION} (AMP {AMP_LEVEL}) | workers: {NUM_WORKERS}", flush=True)
    print(f"  aug scale range: {AUG_MIN_SCALE}-{AUG_MAX_SCALE} (capped for loader speed)", flush=True)
    print(f"  body-color synth (train only): {', '.join(SYNTH_VARIANTS)}", flush=True)
    print(f"  heartbeat: every {HEARTBEAT_SECONDS}s during silent training", flush=True)
    print(f"  drive checkpoint sync: every {CHECKPOINT_SYNC_SECONDS}s", flush=True)
    print(f"  persistent outputs: {RUN_DIR}", flush=True)
    if resume is None:
        print("  resume: no checkpoint found; starting fresh", flush=True)
    else:
        print(f"  resume: {resume}", flush=True)
    print("", flush=True)


def _sync_dir_to_drive() -> None:
    """Incremental local -> Drive mirror (rsync when available, else copytree)."""
    OUTPUT_DRIVE.mkdir(parents=True, exist_ok=True)
    rsync = shutil.which("rsync")
    if rsync:
        subprocess.run(
            [rsync, "-a", "--prune-empty-dirs", f"{OUTPUT_LOCAL}/", f"{OUTPUT_DRIVE}/"],
            check=True,
        )
    else:
        shutil.copytree(OUTPUT_LOCAL, OUTPUT_DRIVE, dirs_exist_ok=True)


def _start_drive_sync() -> threading.Event:
    """Background mirror of PaddleSeg checkpoints to Drive; returns stop event."""
    stop = threading.Event()

    def loop() -> None:
        while not stop.wait(CHECKPOINT_SYNC_SECONDS):
            try:
                print("[drive-sync] syncing checkpoints to Drive...", flush=True)
                _sync_dir_to_drive()
                print("[drive-sync] checkpoints safe in Drive.", flush=True)
            except Exception as exc:
                print(f"[drive-sync] failed, training continues: {exc}", flush=True)

    threading.Thread(target=loop, name="drive-checkpoint-sync", daemon=True).start()
    return stop


def prepare_dataset() -> tuple[Path, str]:
    """Restore one ZIP from Drive, or rebuild once and persist the new format.

    Old per-file caches stay intact but are deliberately not copied from Drive:
    copying tens of thousands of FUSE-mounted files can take 40+ minutes.
    Original source ZIPs already cached in Drive are used for the one-time
    rebuild; subsequent sessions only transfer the packed prepared ZIP.
    """
    if packed_dataset_cache.prepared_usable_archive(
        DRIVE_PREPARED_ARCHIVE, DRIVE_PREPARED_MANIFEST, DATASET_FINGERPRINT,
    ):
        if packed_dataset_cache.restore_prepared_archive(
            DRIVE_PREPARED_ARCHIVE, DRIVE_PREPARED_MANIFEST,
            DATASET_ROOT, DATASET_FINGERPRINT,
        ):
            return find_dataset_root(DATASET_ROOT), "Drive ZIP cache"
    if dataset_cache.prepared_usable(DRIVE_DATASET, DATASET_FINGERPRINT):
        print("[dataset-cache] old directory-format cache detected. Migrating by "
              "rebuilding locally from cached source archives instead of slowly "
              "copying thousands of files from Drive.", flush=True)
    # A previous interrupted directory rsync may have copied the marker before
    # all images. Never treat that partial local tree as a prepared dataset.
    if DATASET_ROOT.exists():
        shutil.rmtree(DATASET_ROOT)
    dataset_root = download_dataset()
    merge_blue_dataset(dataset_root)
    synthesize_body_colors(dataset_root)
    normalize_masks_01(dataset_root)
    packed_dataset_cache.persist_prepared_archive(
        DATASET_ROOT, DRIVE_PREPARED_ARCHIVE, DRIVE_PREPARED_MANIFEST,
        DATASET_FINGERPRINT,
    )
    return dataset_root, "freshly built (ZIP cached)"



def _checkpoint_miou(best_dir: Path) -> float | None:
    """mIoU stored by PaddleSeg in best_model/model.pdstates."""
    states_path = best_dir / "model.pdstates"
    if not states_path.is_file():
        return None
    try:
        import paddle

        states = paddle.load(str(states_path))
        return float(states["mIoU"])
    except (OSError, ValueError, TypeError, KeyError) as exc:
        print(f"[best] could not read mIoU at {states_path}: {exc}", flush=True)
        return None


def _keep_stronger_best_model(backup_dir: Path, previous_miou: float | None) -> None:
    """Resume can overwrite best_model even if new validation quality is worse."""
    if previous_miou is None or not backup_dir.is_dir():
        return
    best_dir = OUTPUT_LOCAL / "best_model"
    new_miou = _checkpoint_miou(best_dir)
    if new_miou is None or new_miou < previous_miou:
        print(
            f"[best] restoring pre-resume best model: mIoU={previous_miou:.4f} "
            f"(current={new_miou})", flush=True,
        )
        if best_dir.exists():
            shutil.rmtree(best_dir)
        shutil.copytree(backup_dir, best_dir)
    else:
        print(f"[best] retained new best: mIoU={new_miou:.4f} "
              f"(previous={previous_miou:.4f})", flush=True)



def train(config: Path, dataset_source: str) -> None:
    OUTPUT_LOCAL.mkdir(parents=True, exist_ok=True)
    if OUTPUT_DRIVE.is_dir():
        print("[resume] syncing Drive output dir locally...", flush=True)
        shutil.copytree(OUTPUT_DRIVE, OUTPUT_LOCAL, dirs_exist_ok=True)
    command = [
        sys.executable, "-u", str(PADDLESEG_DIR / "tools" / "train.py"),
        "--config", str(config),
        "--do_eval",
        "--save_interval", str(SAVE_INTERVAL),
        "--save_dir", str(OUTPUT_LOCAL),
        "--num_workers", str(NUM_WORKERS),
        "--precision", PRECISION,
        "--amp_level", AMP_LEVEL,
    ]
    resume = latest_resume_checkpoint()
    # latest_resume_checkpoint may live under OUTPUT_DRIVE; it was just synced.
    if resume is not None and resume.startswith(str(OUTPUT_DRIVE)):
        resume = resume.replace(str(OUTPUT_DRIVE), str(OUTPUT_LOCAL), 1)
        if not Path(resume).is_dir():
            resume = None
    if resume is not None:
        command.extend(["--resume_model", resume])
    print_run_configuration(find_dataset_root(DATASET_ROOT), resume, dataset_source)
    # PaddleSeg starts best_mean_iou=-1 on EVERY invocation, even after resume.
    # Save the previous winner outside the synced output so later evaluations
    # cannot silently replace a superior best_model.
    backup_dir = WORK / "best_model_before_resume"
    if backup_dir.exists():
        shutil.rmtree(backup_dir)
    old_miou = None
    if resume is not None:
        previous_best = OUTPUT_LOCAL / "best_model"
        old_miou = _checkpoint_miou(previous_best)
        if old_miou is not None and (previous_best / "model.pdparams").is_file():
            shutil.copytree(previous_best, backup_dir)
            print(f"[best] previous best mIoU={old_miou:.4f}, safe backup created", flush=True)
        # A finished 8k run may be re-opened only to export or preview.
        try:
            completed_iter = int(Path(resume).name.rsplit("_", 1)[1])
        except (IndexError, ValueError):
            completed_iter = -1
        if completed_iter >= ITERS:
            print(f"[resume] iter_{completed_iter} already reaches target {ITERS}; "
                  "skipping re-training", flush=True)
            _sync_dir_to_drive()
            return
    stop_sync = _start_drive_sync()
    # tools/train.py does `from paddleseg...`: the script's own dir (tools/)
    # lands on sys.path, not the repo root, so point PYTHONPATH at PaddleSeg.
    paddle_env = {"PYTHONPATH": str(PADDLESEG_DIR)}
    try:
        run_with_heartbeat(command, cwd=PADDLESEG_DIR, extra_env=paddle_env)
    finally:
        stop_sync.set()
    _keep_stronger_best_model(backup_dir, old_miou)
    print("[train] final sync of outputs to Drive...", flush=True)
    _sync_dir_to_drive()


def _converter_executable() -> Path | None:
    """Use isolated Python 3.12 wheels: Paddle2ONNX/ONNX lack cp313 wheels.

    Keep the working Colab Python 3.13 + CUDA Paddle installation untouched.
    uv installs its own managed Python 3.12 when the runtime lacks one.
    """
    binary = CONVERTER_VENV / "bin" / "paddle2onnx"
    if binary.is_file():
        return binary
    print("[onnx] Paddle2ONNX 2.1.0 + ONNX 1.17 have Python 3.12 wheels, "
          "but not Python 3.13 wheels; creating separate 3.12 converter.", flush=True)
    try:
        # uv itself has cp313 wheels and can install a standalone CPython 3.12.
        _pip_install(["-q", "uv"], description="uv for isolated ONNX converter")
        run_with_heartbeat([
            sys.executable, "-m", "uv", "venv", "--python", "3.12", str(CONVERTER_VENV),
        ], heartbeat_seconds=60)
        run_with_heartbeat([
            sys.executable, "-m", "uv", "pip", "install", "--python",
            str(CONVERTER_VENV / "bin" / "python"),
            "paddlepaddle==3.1.1", "paddle2onnx==2.1.0",
        ], heartbeat_seconds=60)
        if not binary.is_file():
            raise RuntimeError(f"Converter executable missing after installation: {binary}")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        (RUN_DIR / "onnx_setup_error.txt").write_text(
            f"Python 3.12 converter install failed: {exc}\n", encoding="utf-8"
        )
        print(f"[onnx] converter setup failed: {exc}. PaddleSeg inference saved.", flush=True)
        return None
    return binary


def _inference_model_filename(inference_dir: Path) -> str:
    """Paddle 3.x may export PIR model.json instead of legacy model.pdmodel."""
    for name in ("model.json", "model.pdmodel"):
        if (inference_dir / name).is_file():
            return name
    raise FileNotFoundError(
        f"Export completed but neither model.json nor model.pdmodel exists "
        f"in {inference_dir}; found: {[p.name for p in inference_dir.iterdir()]}"
    )


def export_and_convert() -> Path | None:
    best = OUTPUT_LOCAL / "best_model" / "model.pdparams"
    if not best.is_file():
        print(f"[export] no best_model at {best}; skipping export", flush=True)
        return None
    inference_dir = WORK / "inference"
    try:
        run_with_heartbeat([
            sys.executable, str(PADDLESEG_DIR / "tools" / "export.py"),
            "--config", str(CUSTOM_CONFIG),
            "--model_path", str(best),
            "--save_dir", str(inference_dir),
        ], cwd=PADDLESEG_DIR, extra_env={"PYTHONPATH": str(PADDLESEG_DIR)})
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "export_error.txt").write_text(str(exc), encoding="utf-8")
        print("[export] PaddleSeg export failed; best checkpoint remains safe in Drive.", flush=True)
        return None

    # Persist the native Paddle model even when ONNX conversion is not possible.
    shutil.copytree(inference_dir, DRIVE_INFERENCE, dirs_exist_ok=True)
    print(f"[export] native Paddle inference model saved to {DRIVE_INFERENCE}", flush=True)
    try:
        model_filename = _inference_model_filename(inference_dir)
        converter = _converter_executable()
        if converter is None:
            return None
        onnx_path = RUN_DIR / "resistor_segmenter_ppliteseg.onnx"
        # Remove any stale model before converting, to avoid false success.
        onnx_path.unlink(missing_ok=True)
        run_with_heartbeat([
            str(converter),
            "--model_dir", str(inference_dir),
            "--model_filename", model_filename,
            "--params_filename", "model.pdiparams",
            "--save_file", str(onnx_path),
            "--opset_version", "12",
        ], heartbeat_seconds=60)
        if not onnx_path.is_file() or onnx_path.stat().st_size == 0:
            raise RuntimeError("Paddle2ONNX returned without creating an ONNX file")
        print(f"[export] ONNX bytes: {onnx_path.stat().st_size / 1024:.1f} KiB", flush=True)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        (RUN_DIR / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"[export] ONNX conversion failed: {exc}. Native Paddle model remains usable.", flush=True)
        return None

    if pip_best_effort(["onnx2tf", "onnxsim"], description="onnx2tf for TFLite conversion"):
        try:
            run_with_heartbeat([
                sys.executable, "-m", "onnx2tf", "-i", str(onnx_path),
                "-o", str(RUN_DIR / "tflite_tmp"),
            ], heartbeat_seconds=60)
            candidates = sorted((RUN_DIR / "tflite_tmp").rglob("*.tflite"))
            if candidates:
                tflite_path = RUN_DIR / "resistor_segmenter_ppliteseg.tflite"
                shutil.copy(candidates[0], tflite_path)
                print(f"[export] TFLite bytes: {tflite_path.stat().st_size / 1024:.1f} KiB", flush=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            (RUN_DIR / "tflite_export_error.txt").write_text(str(exc), encoding="utf-8")
            print(f"[export] TFLite conversion failed: {exc}; ONNX remains available.", flush=True)
    return onnx_path


def evaluate_test(onnx_path: Path | None, dataset_root: Path) -> None:
    if onnx_path is None or not onnx_path.is_file():
        print("[eval] no ONNX model; skipping hue-bucketed test eval", flush=True)
        return
    pip_best_effort(["onnxruntime"], description="onnxruntime for hue-bucketed eval")
    try:
        run([
            sys.executable, str(EVAL_SCRIPT),
            "--dataset-root", str(dataset_root),
            "--split", "test",
            "--onnx", str(onnx_path),
            "--image-size", str(IMAGE_SIZE),
            "--output", str(RUN_DIR / "color_buckets_test.json"),
        ])
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "color_eval_error.txt").write_text(str(exc), encoding="utf-8")
        print("[eval] hue-bucketed eval failed; checkpoints remain saved.", flush=True)


def summarize() -> None:
    summary = {
        "model": "PP-LiteSeg-T (STDC1), binary, Apache-2.0 path",
        "input": IMAGE_SIZE,
        "iters": ITERS,
        "synth_variants_train_only": list(SYNTH_VARIANTS),
        "persistent_run_dir": str(RUN_DIR),
    }
    buckets = RUN_DIR / "color_buckets_test.json"
    if buckets.is_file():
        summary["color_buckets_test"] = json.loads(buckets.read_text(encoding="utf-8"))
    (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


def package_and_download() -> None:
    """Package model outputs only, never the multi-GB Drive dataset caches.

    Older runs can still contain the v11 per-file dataset directory and raw
    source archives. Zipping RUN_DIR recursively would copy all of them again.
    """
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    reports = [
        p for p in RUN_DIR.iterdir()
        if p.is_file() and (p.suffix in (".onnx", ".tflite", ".json")
                            or p.name.endswith("_error.txt"))
        and p.name != DRIVE_PREPARED_MANIFEST.name
    ]
    outputs = sorted(p for p in OUTPUT_DRIVE.rglob("*") if p.is_file()) if OUTPUT_DRIVE.exists() else []
    inference = sorted(p for p in DRIVE_INFERENCE.rglob("*") if p.is_file()) if DRIVE_INFERENCE.exists() else []
    selected = sorted(reports) + outputs + inference
    print(f"[package] packing {len(selected)} result files (excluding dataset caches)...", flush=True)
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        for i, path in enumerate(selected, 1):
            output.write(path, arcname=path.relative_to(RUN_DIR).as_posix())
            if i % 20 == 0 or i == len(selected):
                print(f"[package] {i}/{len(selected)} files", flush=True)
    print(f"Created {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
    try:
        from google.colab import files
        files.download(str(ZIP_PATH))
    except Exception:
        print("Output remains persistently available at", RUN_DIR, flush=True)



def _foreground_mask_from_output(output: object) -> "np.ndarray":
    """Convert an ONNX or PaddleSeg batch=1 output to a 2-D foreground mask."""
    import numpy as np

    arr = np.asarray(output)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got model output {arr.shape}")
        arr = arr[0]
    if arr.ndim == 3:
        if arr.shape[0] == NUM_CLASSES:
            return arr.argmax(axis=0) == 1
        if arr.shape[-1] == NUM_CLASSES:
            return arr.argmax(axis=-1) == 1
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unknown output channel layout: {arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D mask, got model output {arr.shape}")
    if arr.dtype.kind in ("b", "i", "u"):
        return arr > 0
    return arr >= 0.5


def _letterbox_preview_input(image: "Image.Image") -> "np.ndarray":
    """Match PaddleSeg Normalize() defaults and the ONNX test evaluator."""
    import numpy as np
    from PIL import Image

    width, height = image.size
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    scaled = image.resize((new_width, new_height), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), (128, 128, 128))
    canvas.paste(scaled, ((IMAGE_SIZE - new_width) // 2, (IMAGE_SIZE - new_height) // 2))
    rgb = np.asarray(canvas, dtype=np.float32) / 255.0
    # Stock PaddleSeg quick-start uses Normalize() with 0.5 mean/std, not ImageNet.
    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    return ((rgb - mean) / std).transpose(2, 0, 1)[None].astype(np.float32)


def _unletterbox_preview_mask(
    foreground: "np.ndarray", original_size: tuple[int, int],
    display_size: tuple[int, int],
) -> "np.ndarray":
    """Undo padding before resizing to the preview (not full-size image) dimensions."""
    import numpy as np
    from PIL import Image

    width, height = original_size
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    x0, y0 = (IMAGE_SIZE - new_width) // 2, (IMAGE_SIZE - new_height) // 2
    mask = Image.fromarray(np.asarray(foreground, dtype=np.uint8) * 255)
    if mask.size != (IMAGE_SIZE, IMAGE_SIZE):
        mask = mask.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST)
    cropped = mask.crop((x0, y0, x0 + new_width, y0 + new_height))
    return np.asarray(cropped.resize(display_size, Image.Resampling.NEAREST)) > 0


def _load_preview_predictor(onnx_path: Path | None):
    """Prefer the exported mobile model, fall back to the real PaddleSeg best checkpoint."""
    onnx = onnx_path if onnx_path and onnx_path.is_file() else (
        RUN_DIR / "resistor_segmenter_ppliteseg.onnx"
    )
    if onnx.is_file():
        try:
            try:
                import onnxruntime as ort
            except ImportError:
                if not pip_best_effort(["onnxruntime"], description="ONNX image preview"):
                    raise RuntimeError("Could not install onnxruntime")
                import onnxruntime as ort

            session = ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
            model_input = session.get_inputs()[0].name
            print(f"[preview] testing exported ONNX: {onnx}", flush=True)

            def predict_onnx(tensor):
                return session.run(None, {model_input: tensor})[0]

            return predict_onnx
        except Exception as exc:
            print(f"[preview] ONNX failed to load: {exc}; trying Paddle checkpoint", flush=True)

    checkpoint = OUTPUT_LOCAL / "best_model" / "model.pdparams"
    if not checkpoint.is_file():
        checkpoint = OUTPUT_DRIVE / "best_model" / "model.pdparams"
    if not checkpoint.is_file() or not CUSTOM_CONFIG.is_file() or not PADDLESEG_DIR.is_dir():
        print("[preview] No usable ONNX or local Paddle checkpoint. "
              "Model download was retained; upload preview is unavailable.", flush=True)
        return None

    # PaddleSeg lives in a checked-out source tree, not site-packages.
    # Subprocess exports set PYTHONPATH; direct Colab inference also needs it.
    if str(PADDLESEG_DIR) not in sys.path:
        sys.path.insert(0, str(PADDLESEG_DIR))
    import paddle
    from paddleseg.cvlibs import Config
    from paddleseg.utils import utils

    config = Config(str(CUSTOM_CONFIG))
    model = config.model
    utils.load_entire_model(model, str(checkpoint))
    model.eval()
    print(f"[preview] testing PaddleSeg checkpoint: {checkpoint}", flush=True)

    def predict_paddle(tensor):
        with paddle.no_grad():
            result = model(paddle.to_tensor(tensor))
        if isinstance(result, (tuple, list)):
            result = result[0]
        return result.numpy()

    return predict_paddle


def preview_uploaded_images(onnx_path: Path | None = None) -> None:
    """Optional visual smoke test AFTER the training ZIP download.

    Google Colab's native uploader accepts several images in one operation.
    Preview images stay local to the Colab VM, not in the Drive training cache.
    The result is qualitative; images without GT masks cannot produce IoU.
    """
    predictor = _load_preview_predictor(onnx_path)
    if predictor is None:
        return

    import io
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image, ImageOps, UnidentifiedImageError
    from google.colab import files

    print("\n[preview] Training/export/download stage is complete.", flush=True)
    print("[preview] Upload one or multiple resistor photos (JPG/PNG/WEBP/BMP).", flush=True)
    print("[preview] The output shows original / predicted body mask / red overlay.", flush=True)
    print("[preview] This is a visual test, NOT a ground-truth accuracy/IoU score.", flush=True)
    uploaded = files.upload()
    if not uploaded:
        print("[preview] No images uploaded; model files remain in Drive.", flush=True)
        return

    preview_dir = WORK / "manual_previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    for index, (filename, data) in enumerate(uploaded.items(), start=1):
        try:
            with Image.open(io.BytesIO(data)) as source:
                photo = ImageOps.exif_transpose(source).convert("RGB")
            if min(photo.size) < 1:
                raise ValueError("Empty image")
            original_size = photo.size
            tensor = _letterbox_preview_input(photo)
            foreground = _foreground_mask_from_output(predictor(tensor))

            photo.thumbnail((900, 900), Image.Resampling.LANCZOS)
            mask = _unletterbox_preview_mask(foreground, original_size, photo.size)
            overlay = np.asarray(photo).copy()
            tint = np.array([255, 40, 40], dtype=np.float32)
            overlay[mask] = (
                overlay[mask].astype(np.float32) * 0.40 + tint * 0.60
            ).astype(np.uint8)

            safe_stem = f"upload_{index:03d}"
            Image.fromarray(mask.astype(np.uint8) * 255).save(preview_dir / f"{safe_stem}_mask.png")
            Image.fromarray(overlay).save(preview_dir / f"{safe_stem}_overlay.png")

            figure, axes = plt.subplots(1, 3, figsize=(14, 4.5))
            axes[0].imshow(photo)
            axes[0].set_title("Original")
            axes[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
            axes[1].set_title("Resistor body mask")
            axes[2].imshow(overlay)
            axes[2].set_title("Overlay (red = detected)")
            for ax in axes:
                ax.axis("off")
            figure.suptitle(f"{index}. {filename}  |  mask coverage: {100 * mask.mean():.1f}%")
            figure.tight_layout()
            plt.show()
            plt.close(figure)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            print(f"[preview] Skipped {filename!r}: {exc}", flush=True)
        except Exception as exc:
            print(f"[preview] Inference failed for {filename!r}: {exc}", flush=True)
    print(f"[preview] Masks and overlays saved locally: {preview_dir}", flush=True)
    print("[preview] To test more photos later, run a new Colab cell with:", flush=True)
    print("  import runpy", flush=True)
    print("  p = runpy.run_path('/content/train_pp_liteseg_colab.py')", flush=True)
    print("  p['preview_uploaded_images']()", flush=True)


def main() -> None:
    print(f"PaddleSeg Colab version: {PADDLESEG_COLAB_VERSION}", flush=True)
    if os.environ.get("PP_LITESEG_PREVIEW_ONLY") == "1":
        mount_drive()
        preview_uploaded_images()
        return
    if os.environ.get("PP_LITESEG_EXPORT_ONLY") == "1":
        require_gpu()
        mount_drive()
        clone_repos()
        install_dependencies()
        OUTPUT_LOCAL.mkdir(parents=True, exist_ok=True)
        if OUTPUT_DRIVE.is_dir():
            shutil.copytree(OUTPUT_DRIVE, OUTPUT_LOCAL, dirs_exist_ok=True)
        build_custom_config(DATASET_ROOT)  # only model architecture is used for export
        onnx_path = export_and_convert()
        summarize()
        package_and_download()
        preview_uploaded_images(onnx_path)
        return
    require_gpu()
    mount_drive()
    clone_repos()
    sys.path.insert(0, str(RESISTOR_REPO / "src"))
    global dataset_cache, packed_dataset_cache
    from resistor_model.tools import dataset_cache, packed_dataset_cache
    install_dependencies()
    dataset_root, dataset_source = prepare_dataset()
    write_paddleseg_lists(dataset_root)
    config = build_custom_config(dataset_root)
    train(config, dataset_source)
    onnx_path = export_and_convert()
    evaluate_test(onnx_path, dataset_root)
    summarize()
    package_and_download()
    try:
        preview_uploaded_images(onnx_path)
    except Exception as exc:
        print(f"[preview] Optional upload test failed: {exc}", flush=True)
        print(f"[preview] Training ZIP is already saved at {ZIP_PATH}", flush=True)


if __name__ == "__main__":
    main()
