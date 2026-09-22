"""Peak memory / GPU sampling helpers for the JEV invariants.

JEV claims a flat memory footprint because nothing is decoded. These helpers
let the evaluation scripts prove it on the running server process.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
import time
from pathlib import Path


def gpu_memory_mib() -> dict[int, int]:
    if not shutil.which("nvidia-smi"):
        return {}
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    result: dict[int, int] = {}
    for line in out.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            result[int(parts[0])] = int(parts[1])
    return result


def read_rss_mib(pid: int | None) -> float:
    if not pid:
        return 0.0
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return 0.0
    for line in status.splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    return 0.0


class PeakSampler:
    """Samples server RSS and GPU memory in a background thread."""

    def __init__(self, pid: int | None, interval: float = 0.1, gpu_indices: list[int] | None = None):
        self.pid = pid
        self.interval = interval
        self.gpu_indices = gpu_indices
        self.peak_rss_mib = 0.0
        self.peak_gpu_mib: dict[int, int] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "PeakSampler":
        self._sample_once()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def _sample_once(self) -> None:
        self.peak_rss_mib = max(self.peak_rss_mib, read_rss_mib(self.pid))
        for idx, used in gpu_memory_mib().items():
            if self.gpu_indices and idx not in self.gpu_indices:
                continue
            self.peak_gpu_mib[idx] = max(self.peak_gpu_mib.get(idx, 0), used)

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        return {
            "peak_server_rss_mib": round(self.peak_rss_mib, 1),
            "peak_gpu_memory_mib": self.peak_gpu_mib,
        }

    def __enter__(self) -> "PeakSampler":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def wait(seconds: float) -> None:
    time.sleep(seconds)
