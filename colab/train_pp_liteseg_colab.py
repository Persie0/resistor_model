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
import shutil
import subprocess
import sys
import urllib.request

PADDLESEG_COLAB_VERSION = "2026-10-11-v1-pplite"
PADDLESEG_REPO = "https://github.com/PaddlePaddle/PaddleSeg.git"
PADDLESEG_BRANCH = "release/2.10"
# Same v4 SAM-mask source as the LR-ASPP recipe.
DATASET_URL = "https://github.com/Persie0/resistor_model/releases/download/v4/resistor_sam3_merged.zip"
DATASET_SHA256 = "be3a1bb3b952f07556906decf6393fa7a8c228665867f721914a4e8e64376c6f"
MODEL_TYPE_NEEDLE = "liteseg"  # parsed stock config must be a PP-LiteSeg model
IMAGE_SIZE = 512
ITERS = 20000
BATCH_SIZE = 16
SAVE_INTERVAL = 1000
NUM_CLASSES = 2  # background + resistor body
# Train-only synthesis; valid/test stay real so metrics measure reality.
SYNTH_VARIANTS = ("blue", "green", "random")
SYNTH_SEED = 7

WORK = Path("/content/pp_liteseg")
RESISTOR_REPO = WORK / "resistor_model"
SYNTH_SCRIPT = RESISTOR_REPO / "src" / "resistor_model" / "tools" / "synthesize_body_colors.py"
EVAL_SCRIPT = RESISTOR_REPO / "src" / "resistor_model" / "tools" / "eval_color_buckets.py"
PADDLESEG_DIR = WORK / "PaddleSeg"
ARCHIVE = WORK / "resistor_sam3_merged.zip"
DATASET_ROOT = WORK / "dataset"
OUTPUT_LOCAL = WORK / "output"
CUSTOM_CONFIG = WORK / "pp_liteseg_resistor_512.yml"
DRIVE_MOUNT = Path("/content/drive")
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "segmentation-pp-liteseg"
OUTPUT_DRIVE = RUN_DIR / "output"
ZIP_PATH = WORK / "resistor-segmentation-pp-liteseg.zip"


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


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


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    # PaddlePaddle GPU wheel + PaddleSeg deps (yaml, visualdl, ... come along).
    run([sys.executable, "-m", "pip", "install", "-q", "paddlepaddle-gpu"])
    run([sys.executable, "-m", "pip", "install", "-q", "-r", str(PADDLESEG_DIR / "requirements.txt")])
    run([sys.executable, "-m", "pip", "install", "-q", "paddle2onnx", "onnx", "onnxruntime", "opencv-python", "pillow"])
    import paddle  # noqa: E402  (verify install before the long training run)

    print(f"paddle {paddle.__version__} | cuda: {paddle.device.cuda.device_count()} device(s)", flush=True)


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


def download_dataset() -> Path:
    WORK.mkdir(parents=True, exist_ok=True)
    if not (ARCHIVE.is_file() and archive_sha256(ARCHIVE) == DATASET_SHA256):
        if ARCHIVE.is_file():
            ARCHIVE.unlink()
        print("[dataset] downloading v4 SAM-mask dataset...", flush=True)
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "resistor-model-colab"})
        with urllib.request.urlopen(request) as response, ARCHIVE.open("wb") as destination:
            shutil.copyfileobj(response, destination)
        if archive_sha256(ARCHIVE) != DATASET_SHA256:
            ARCHIVE.unlink(missing_ok=True)
            raise RuntimeError("v4 archive checksum mismatch")
        print(f"[dataset] checksum OK | {ARCHIVE.stat().st_size / 1024 / 1024:.1f} MiB", flush=True)
    else:
        print("[dataset] verified archive cached locally", flush=True)
    root = find_dataset_root(DATASET_ROOT) if (DATASET_ROOT / "train").exists() else None
    if root is None:
        if DATASET_ROOT.exists():
            shutil.rmtree(DATASET_ROOT)
        DATASET_ROOT.mkdir(parents=True, exist_ok=True)
        shutil.unpack_archive(ARCHIVE, DATASET_ROOT)
        root = find_dataset_root(DATASET_ROOT)
    print(f"[dataset] ready: {root}", flush=True)
    return root


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
    ])


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
    CUSTOM_CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(f"[config] {base.name} -> {CUSTOM_CONFIG} | iters={ITERS} batch={BATCH_SIZE}", flush=True)
    print(f"[config] transfer from stock pretrained: {cfg['model'].get('pretrained')}", flush=True)
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


def train(config: Path) -> None:
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
        "--num_workers", "2",
    ]
    resume = latest_resume_checkpoint()
    # latest_resume_checkpoint may live under OUTPUT_DRIVE; it was just synced.
    if resume is not None and resume.startswith(str(OUTPUT_DRIVE)):
        resume = resume.replace(str(OUTPUT_DRIVE), str(OUTPUT_LOCAL), 1)
        if not Path(resume).is_dir():
            resume = None
    if resume is not None:
        command.extend(["--resume_model", resume])
    run(command, cwd=PADDLESEG_DIR)
    print("[train] syncing outputs to Drive...", flush=True)
    shutil.copytree(OUTPUT_LOCAL, OUTPUT_DRIVE, dirs_exist_ok=True)


def export_and_convert() -> Path | None:
    best = OUTPUT_LOCAL / "best_model" / "model.pdparams"
    if not best.is_file():
        print(f"[export] no best_model at {best}; skipping export", flush=True)
        return None
    inference_dir = WORK / "inference"
    try:
        run([
            sys.executable, str(PADDLESEG_DIR / "tools" / "export.py"),
            "--config", str(CUSTOM_CONFIG),
            "--model_path", str(best),
            "--save_dir", str(inference_dir),
        ], cwd=PADDLESEG_DIR)
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "export_error.txt").write_text(str(exc), encoding="utf-8")
        print("[export] PaddleSeg export failed; run help: tools/export.py --help", flush=True)
        return None
    onnx_path = RUN_DIR / "resistor_segmenter_ppliteseg.onnx"
    try:
        run([
            sys.executable, "-m", "paddle2onnx",
            "--model_dir", str(inference_dir),
            "--model_filename", "model.pdmodel",
            "--params_filename", "model.pdiparams",
            "--save_file", str(onnx_path),
            "--opset_version", "12",
            "--enable_onnx_checker", "True",
        ])
        print(f"[export] ONNX bytes: {onnx_path.stat().st_size / 1024:.1f} KiB", flush=True)
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print("[export] paddle2onnx failed; inference model remains in Drive output.", flush=True)
        return None
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "onnx2tf", "onnxsim"])
        run([sys.executable, "-m", "onnx2tf", "-i", str(onnx_path), "-o", str(RUN_DIR / "tflite_tmp")])
        candidates = sorted((RUN_DIR / "tflite_tmp").rglob("*.tflite"))
        if candidates:
            tflite_path = RUN_DIR / "resistor_segmenter_ppliteseg.tflite"
            shutil.copy(candidates[0], tflite_path)
            print(f"[export] TFLite bytes: {tflite_path.stat().st_size / 1024:.1f} KiB", flush=True)
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "tflite_export_error.txt").write_text(str(exc), encoding="utf-8")
        print("[export] TFLite conversion failed; ONNX remains the mobile artifact.", flush=True)
    return onnx_path


def evaluate_test(onnx_path: Path | None, dataset_root: Path) -> None:
    if onnx_path is None or not onnx_path.is_file():
        print("[eval] no ONNX model; skipping hue-bucketed test eval", flush=True)
        return
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
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", root_dir=RUN_DIR)
    print(f"Created {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
    try:
        from google.colab import files
        files.download(str(ZIP_PATH))
    except Exception:
        print("Output remains persistently available at", RUN_DIR, flush=True)


def main() -> None:
    print(f"PaddleSeg Colab version: {PADDLESEG_COLAB_VERSION}", flush=True)
    require_gpu()
    mount_drive()
    clone_repos()
    install_dependencies()
    dataset_root = download_dataset()
    synthesize_body_colors(dataset_root)
    write_paddleseg_lists(dataset_root)
    config = build_custom_config(dataset_root)
    train(config)
    onnx_path = export_and_convert()
    evaluate_test(onnx_path, dataset_root)
    summarize()
    package_and_download()


if __name__ == "__main__":
    main()
