"""Visible-progress launcher for the SAM 3 resistor dataset refinement Colab.

The underlying workflow stays in ``sam3_merge_3_resistor_datasets.py``. This launcher
adds Colab-friendly progress output around long/silent operations without changing
annotation or export behavior.
"""

from __future__ import annotations

import os
import runpy
import shutil
import threading
import time
import urllib.request
import zipfile
from pathlib import Path

MAIN_URL = "https://raw.githubusercontent.com/Persie0/resistor_model/main/colab/sam3_merge_3_resistor_datasets.py"
MAIN_PATH = Path("/content/sam3_merge_3_resistor_datasets.py")
HEARTBEAT_SECONDS = 15.0
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024
DOWNLOAD_REPORT_FRACTION = 0.10
EXPORT_REPORT_EVERY = 100


def _download_main_script() -> None:
    print("[setup] downloading SAM 3 workflow script...", flush=True)
    request = urllib.request.Request(MAIN_URL, headers={"User-Agent": "resistor-model-colab"})
    with urllib.request.urlopen(request) as response, MAIN_PATH.open("wb") as destination:
        shutil.copyfileobj(response, destination)
    print(f"[setup] workflow script ready: {MAIN_PATH}", flush=True)


def _install_download_progress() -> None:
    def visible_urlretrieve(url, filename=None, reporthook=None, data=None):
        if filename is None:
            filename = os.path.basename(urllib.request.urlparse(url).path) or "/content/download.bin"
        destination = Path(filename)
        destination.parent.mkdir(parents=True, exist_ok=True)

        request = urllib.request.Request(url, headers={"User-Agent": "resistor-model-colab"})
        with urllib.request.urlopen(request, data=data) as response, destination.open("wb") as out:
            raw_total = response.headers.get("Content-Length")
            total = int(raw_total) if raw_total and raw_total.isdigit() else 0
            copied = 0
            next_fraction = DOWNLOAD_REPORT_FRACTION

            print(
                f"[download] {destination.name} | "
                + (f"0/{total / 1024 / 1024:.1f} MiB (0%)" if total else "starting"),
                flush=True,
            )

            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                out.write(chunk)
                copied += len(chunk)

                should_report = False
                if total:
                    fraction = copied / total
                    if fraction >= next_fraction or copied >= total:
                        should_report = True
                        while next_fraction <= fraction:
                            next_fraction += DOWNLOAD_REPORT_FRACTION
                elif copied % (64 * 1024 * 1024) < DOWNLOAD_CHUNK_BYTES:
                    should_report = True

                if should_report:
                    if total:
                        percent = min(100.0, 100.0 * copied / total)
                        print(
                            f"[download] {destination.name} | "
                            f"{copied / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MiB "
                            f"({percent:.0f}%)",
                            flush=True,
                        )
                    else:
                        print(
                            f"[download] {destination.name} | {copied / 1024 / 1024:.1f} MiB",
                            flush=True,
                        )

                if reporthook is not None:
                    reporthook(max(1, copied // DOWNLOAD_CHUNK_BYTES), DOWNLOAD_CHUNK_BYTES, total)

        print(
            f"[download] {destination.name} complete | {copied / 1024 / 1024:.1f} MiB",
            flush=True,
        )
        return str(destination), response.headers

    urllib.request.urlretrieve = visible_urlretrieve


def _install_extraction_progress() -> None:
    original_extractall = zipfile.ZipFile.extractall

    def visible_extractall(self, path=None, members=None, pwd=None):
        target = Path(path or os.getcwd())
        member_count = len(members) if members is not None else len(self.infolist())
        print(
            f"[extract] {Path(self.filename).name} -> {target} | {member_count} files",
            flush=True,
        )
        result = original_extractall(self, path=path, members=members, pwd=pwd)
        print(f"[extract] {Path(self.filename).name} complete", flush=True)
        return result

    zipfile.ZipFile.extractall = visible_extractall


def _install_export_progress() -> None:
    original_copy2 = shutil.copy2
    counter = {"images": 0}

    def visible_copy2(src, dst, *args, **kwargs):
        result = original_copy2(src, dst, *args, **kwargs)
        counter["images"] += 1
        count = counter["images"]
        if count == 1 or count % EXPORT_REPORT_EVERY == 0:
            print(f"[export] copied {count} images", flush=True)
        return result

    shutil.copy2 = visible_copy2


def _heartbeat(stop_event: threading.Event, started: float) -> None:
    while not stop_event.wait(HEARTBEAT_SECONDS):
        elapsed = time.monotonic() - started
        print(
            f"[heartbeat] SAM 3 workflow still running | elapsed {elapsed:.0f}s",
            flush=True,
        )


def main() -> None:
    print("SAM 3 merge Colab: visible progress enabled", flush=True)
    print(f"  heartbeat: every {HEARTBEAT_SECONDS:.0f}s during silent work", flush=True)
    print("  downloads: byte + percent progress", flush=True)
    print(f"  export: report every {EXPORT_REPORT_EVERY} copied images", flush=True)
    print("  SAM 3 refinement: tqdm image progress from the main workflow", flush=True)
    print("", flush=True)

    _download_main_script()
    _install_download_progress()
    _install_extraction_progress()
    _install_export_progress()

    stop_event = threading.Event()
    started = time.monotonic()
    thread = threading.Thread(target=_heartbeat, args=(stop_event, started), daemon=True)
    thread.start()

    try:
        runpy.run_path(str(MAIN_PATH), run_name="__main__")
    finally:
        stop_event.set()
        thread.join(timeout=1.0)

    elapsed = time.monotonic() - started
    print(f"[done] SAM 3 workflow finished | elapsed {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
