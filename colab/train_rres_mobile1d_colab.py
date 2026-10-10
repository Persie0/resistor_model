"""Google Colab GPU training script for the mobile1d body-only reader run.

Usage in Colab:
1. Runtime -> Change runtime type -> GPU.
2. Run this script/notebook.
3. Authorize Google Drive when prompted.

Pipeline context: a segmentation model runs BEFORE this model and hands it a
resistor-body-only crop that is perfectly horizontal but of unknown direction
(0 degrees or 180 degrees flipped). This script therefore trains
``ResistorBandNetMobile1D`` (``architecture: mobile1d``), a TFLite-safe 1D
reader, instead of the heavier V2 transformer:

* input 64x384 (was 128x768), dense bins 192 (was 256),
* single depthwise-separable CNN trunk with early height collapse,
* cheap 2-channel log-ratio chroma prepend instead of a 2nd encoder,
* 1D depthwise TCN instead of multi-head self-attention,
* tiny single-head cross-attention slot reader instead of TransformerDecoder,
* BatchNorm + HardSwish only (no GroupNorm/LayerNorm/GELU/MultiheadAttention).

The same three public COCO datasets as the r1 V2 run are downloaded to local
Colab storage. Checkpoints persist in ``MyDrive/resistor_model/r1-mobile1d-colab``
and ``last.pt`` is resumed automatically. If a trained V2 teacher exists at
``MyDrive/resistor_model/r1-v2-colab/best.pt`` it is used for KL distillation
(dense + slot colors); otherwise the student trains from scratch.

Direction handling: hflip p=0.5 keeps the model direction-equivariant during
training; reading direction is resolved at inference by grammar decoding both
ways + tolerance-gap geometry + flip-TTA (see ``evaluate_flip_tta`` below),
never by silently preferring left-to-right.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import urllib.request


EPOCHS = 60
BATCH_SIZE = 32
OUTPUT_HEIGHT = 64
OUTPUT_WIDTH = 384
SEQUENCE_BINS = 192
BASE_CHANNELS = 24
D_MODEL = 128
TCN_LAYERS = 3
TCN_HEADS = 4  # accepted for config compat; mobile1d uses single-head attention
SLOT_DECODER_LAYERS = 2
DROPOUT = 0.10
CONV_KERNEL = 7
# Left-to-right order is arbitrary under 180-degree ambiguity: keep small.
ORDER_WEIGHT = 0.05
COUNT_WEIGHT = 0.3
CTC_WEIGHT = 0.20
LABEL_SMOOTHING = 0.05
JITTER_STRENGTH = 0.7
DISTILL_WEIGHT = 0.5
DISTILL_TEMPERATURE = 3.0
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
RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "r1-mobile1d-colab"
TEACHER_RUN_DIR = DRIVE_MOUNT / "MyDrive" / "resistor_model" / "r1-v2-colab"
CONFIG_PATH = WORK / "r1-mobile1d-colab.yaml"
ZIP_PATH = WORK / "resistor-bandnet-mobile1d-colab.zip"


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


def find_teacher_checkpoint() -> Path | None:
    candidate = TEACHER_RUN_DIR / "best.pt"
    if candidate.is_file() and candidate.stat().st_size > 0:
        return candidate
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
    # Colab already provides CUDA-enabled PyTorch; mobile1d needs no extra deps.
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


def build_config(run_dir: Path, resume: Path | None, teacher: Path | None) -> dict:
    distill_cfg: dict = {"teacher_checkpoint": None, "weight": 0.0, "temperature": DISTILL_TEMPERATURE}
    if teacher is not None:
        distill_cfg = {
            "teacher_checkpoint": str(teacher),
            "weight": DISTILL_WEIGHT,
            "temperature": DISTILL_TEMPERATURE,
        }
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
            # Segmentation already rectifies to horizontal: keep direction flips,
            # shrink rotation jitter so bands stay inside body-only crops.
            "geometric_augment": True,
            "hflip_prob": 0.5,
            "vflip_prob": 0.5,
            "jitter_strength": JITTER_STRENGTH,
        },
        "model": {
            "architecture": "mobile1d",
            "backbone": "mobile1d",
            "pretrained": False,
            "num_colors": 12,
            "max_bands": 6,
            "base_channels": BASE_CHANNELS,
            "d_model": D_MODEL,
            "transformer_layers": TCN_LAYERS,
            "transformer_heads": TCN_HEADS,
            "slot_decoder_layers": SLOT_DECODER_LAYERS,
            "dropout": DROPOUT,
            "drop_path": 0.0,
            "conv_kernel": CONV_KERNEL,
            # Cheap 2ch log-ratio prepend, not a second encoder.
            "use_chromatic_branch": True,
        },
        "train": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": 5e-4,
            "weight_decay": 0.04,
            "warmup_epochs": 3,
            "grad_clip": 1.0,
            "amp": True,
            "ema_decay": 0.995,
            "output_dir": str(run_dir),
            "resume": str(resume) if resume is not None else None,
        },
        "loss": {
            "dense": 1.0,
            "color": 2.0,
            "exist": 0.5,
            "center": 1.0,
            "width": 0.5,
            "order": ORDER_WEIGHT,
            "count": COUNT_WEIGHT,
            "consistency": 0.0,
            "ctc": CTC_WEIGHT,
            "kl": 0.0,
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
        "distill": distill_cfg,
    }


def write_config() -> Path | None:
    import yaml

    resume = find_resume_checkpoint(RUN_DIR)
    if resume is None:
        print("No mobile1d Drive checkpoint found; starting a fresh training run.", flush=True)
    else:
        print(f"Resuming from persistent checkpoint: {resume}", flush=True)
    teacher = find_teacher_checkpoint()
    if teacher is None:
        print("No V2 teacher found at r1-v2-colab/best.pt; training student from scratch.", flush=True)
    else:
        print(f"Distilling from V2 teacher: {teacher}", flush=True)
    cfg = build_config(RUN_DIR, resume, teacher)
    text = yaml.safe_dump(cfg, sort_keys=False)
    CONFIG_PATH.write_text(text, encoding="utf-8")
    (RUN_DIR / "resolved_config.yaml").write_text(text, encoding="utf-8")
    print(text, flush=True)
    return teacher


def benchmark_student() -> dict:
    """Report params + CPU/GPU latency of the trained student at 64x384."""
    import torch

    from resistor_model.runtime import build_model

    ckpt_path = RUN_DIR / "best.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["config"]).eval()
    model.load_state_dict(ckpt["model_state"])
    params = sum(p.numel() for p in model.parameters())
    dummy = torch.zeros(1, 3, OUTPUT_HEIGHT, OUTPUT_WIDTH)
    with torch.no_grad():
        for _ in range(10):
            model(dummy)
        t0 = time.perf_counter()
        iters = 50
        for _ in range(iters):
            model(dummy)
        cpu_ms = (time.perf_counter() - t0) / iters * 1000.0
    report: dict = {
        "params_m": round(params / 1e6, 3),
        "cpu_ms_per_image_64x384": round(cpu_ms, 2),
        "input": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
    }
    if torch.cuda.is_available():
        model = model.cuda().eval()
        gpu_dummy = dummy.cuda()
        with torch.no_grad():
            for _ in range(20):
                model(gpu_dummy)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(100):
                model(gpu_dummy)
            torch.cuda.synchronize()
            report["gpu_ms_per_image_64x384"] = round((time.perf_counter() - t0) / 100 * 1000.0, 2)
    (RUN_DIR / "mobile_benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Mobile benchmark: " + json.dumps(report), flush=True)
    return report


def evaluate_flip_tta(max_samples: int = 200) -> dict:
    """Flip-consistency check for the unknown 180-degree direction.

    Runs the student on each test crop and its horizontal flip, decodes both
    with the grammar-constrained slot decoder, and reports how often the flip
    prediction equals the reversed base prediction (direction-equivariance) and
    how often at least one of the two directions is electrically valid.
    """
    import torch
    from torch.utils.data import DataLoader

    from resistor_model.data.dataset import ResistorBandDataset
    from resistor_model.decoding import decode_slots, count_log_probs
    from resistor_model.runtime import build_model

    ckpt = torch.load(RUN_DIR / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    ids = set(ckpt["splits"]["test"])
    ds = ResistorBandDataset(
        cfg["data"]["manifest"], cfg["data"]["image_root"],
        allowed_resistor_ids=ids,
        output_size=tuple(cfg["data"]["output_size"]),
        sequence_bins=cfg["data"]["sequence_bins"],
        max_bands=cfg["model"]["max_bands"], augment=False,
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    model = build_model(cfg).eval()
    model.load_state_dict(ckpt["model_state"])

    checked = consistent = valid_either = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"]
            flipped = images.flip(dims=[3])  # horizontal flip = 180° ambiguity
            out = model(images)
            out_f = model(flipped)
            probs = torch.softmax(out["slot_color_logits"].float(), dim=-1).cpu().numpy()
            probs_f = torch.softmax(out_f["slot_color_logits"].float(), dim=-1).cpu().numpy()
            count_lp = count_log_probs(out["count_logits"], out.get("slot_exist_logits")).cpu().numpy()
            count_lp_f = count_log_probs(out_f["count_logits"], out_f.get("slot_exist_logits")).cpu().numpy()
            for i in range(images.shape[0]):
                if checked >= max_samples:
                    break
                base = decode_slots(probs[i], count_lp[i])
                flip = decode_slots(probs_f[i], count_lp_f[i])
                # Direction-equivariance: flip prediction should be the reverse
                # of the base prediction (same physical resistor, turned around).
                if list(reversed(flip.colors)) == base.colors:
                    consistent += 1
                if base.decode.valid or flip.decode.valid:
                    valid_either += 1
                checked += 1
            if checked >= max_samples:
                break
    report = {
        "samples": checked,
        "flip_consistency_rate": (consistent / checked) if checked else 0.0,
        "valid_either_direction_rate": (valid_either / checked) if checked else 0.0,
    }
    (RUN_DIR / "flip_tta_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Flip-TTA report: " + json.dumps(report, indent=2), flush=True)
    return report


def export_mobile() -> None:
    onnx_path = RUN_DIR / "resistor_bandnet_mobile1d.onnx"
    try:
        run([
            "resistor-export",
            "--checkpoint", str(RUN_DIR / "best.pt"),
            "--output", str(onnx_path),
        ])
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print("ONNX export failed; checkpoints and metrics remain saved in Drive.", flush=True)
        return
    print(f"ONNX bytes: {onnx_path.stat().st_size / 1024:.1f} KiB", flush=True)
    # Best-effort TFLite conversion; must not fail the training run.
    try:
        run([sys.executable, "-m", "pip", "install", "-q", "onnx2tf", "onnxsim"])
        tflite_path = RUN_DIR / "resistor_bandnet_mobile1d.tflite"
        run([
            sys.executable, "-m", "onnx2tf",
            "-i", str(onnx_path),
            "-o", str(RUN_DIR / "tflite_tmp"),
        ])
        candidates = sorted((RUN_DIR / "tflite_tmp").rglob("*.tflite"))
        if candidates:
            shutil.copy(candidates[0], tflite_path)
            print(f"TFLite bytes: {tflite_path.stat().st_size / 1024:.1f} KiB", flush=True)
        else:
            print("onnx2tf produced no .tflite file; ONNX remains the mobile artifact.", flush=True)
    except subprocess.CalledProcessError as exc:
        (RUN_DIR / "tflite_export_error.txt").write_text(str(exc), encoding="utf-8")
        print("TFLite conversion failed; ONNX export remains saved in Drive.", flush=True)


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
        benchmark_student()
    except Exception as exc:  # benchmark must not discard a training run
        (RUN_DIR / "benchmark_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"Benchmark failed: {exc}", flush=True)
    try:
        evaluate_flip_tta()
    except Exception as exc:
        (RUN_DIR / "flip_tta_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"Flip-TTA check failed: {exc}", flush=True)
    export_mobile()


def summarize() -> None:
    metrics_path = RUN_DIR / "metrics.jsonl"
    raw_rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    rows_by_epoch = {int(row["epoch"]): row for row in raw_rows}
    rows = [rows_by_epoch[key] for key in sorted(rows_by_epoch)]
    if not rows:
        raise RuntimeError(f"No training metrics found in {metrics_path}")
    best = max(rows, key=lambda row: (row["val"]["exact_sequence_accuracy"], row["val"]["macro_f1"]))
    summary = {
        "architecture": "mobile1d",
        "input": [OUTPUT_HEIGHT, OUTPUT_WIDTH],
        "sequence_bins": SEQUENCE_BINS,
        "epochs": len(rows),
        "best_epoch": best["epoch"],
        "best_validation": best["val"],
        "last_validation": rows[-1]["val"],
        "persistent_run_dir": str(RUN_DIR),
        "datasets": list(DATASET_ASSETS),
    }
    for extra in ("mobile_benchmark.json", "flip_tta_report.json"):
        path = RUN_DIR / extra
        if path.is_file():
            summary[extra.removesuffix(".json")] = json.loads(path.read_text(encoding="utf-8"))
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
