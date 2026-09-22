"""llama.cpp server process management and device placement.

The engine never talks to the model directly: it drives a `llama-server`
subprocess over HTTP. That keeps the JEV readout (grammar-constrained, single
forward pass, post-sampling probabilities) available on any llama.cpp build,
CPU or CUDA, with no Python bindings to compile.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

from .config import PROJECT_ROOT, Config, physical_cpu_count

log = logging.getLogger("jev.server")

_BIN_CANDIDATES = (
    "third_party/llama.cpp/build/bin/llama-server",
    "build/bin/llama-server",
)


class LlamaServerError(RuntimeError):
    pass


def find_llama_server() -> Path:
    env = os.environ.get("LLAMA_SERVER_BIN")
    if env:
        p = Path(env)
        if p.exists():
            return p
        raise LlamaServerError(f"LLAMA_SERVER_BIN does not exist: {env}")
    for rel in _BIN_CANDIDATES:
        p = PROJECT_ROOT / rel
        if p.exists():
            return p
    which = shutil.which("llama-server")
    if which:
        return Path(which)
    raise LlamaServerError(
        "llama-server not found. Run scripts/01_build_llama_cpp.sh first "
        "(or set LLAMA_SERVER_BIN)."
    )


def list_devices(binary: Path | None = None) -> list[str]:
    """Return device names like ['CUDA0'] as reported by llama.cpp."""
    binary = binary or find_llama_server()
    try:
        out = subprocess.run(
            [str(binary), "--list-devices"],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("could not list devices: %s", exc)
        return []
    devices = re.findall(r"^\s{2}([A-Za-z]+\d+):", out.stdout, re.MULTILINE)
    return devices


def resolve_device(cfg: Config, binary: Path | None = None) -> str:
    """Resolve cfg.device (auto|cpu|cuda|cuda:N) into a concrete device string."""
    device = (cfg.device or "auto").strip().lower()
    if device == "auto":
        devices = list_devices(binary)
        device = "cuda" if any(d.upper().startswith("CUDA") for d in devices) else "cpu"
        log.info("device auto-detected as %r (visible: %s)", device, devices or "none")
    if device in ("cuda", "gpu"):
        device = "cuda:all"
    cfg.resolved_device = device
    return device


class LlamaServer:
    """Owns one llama-server subprocess."""

    def __init__(self, cfg: Config, binary: Path | None = None):
        self.cfg = cfg
        self.binary = binary or find_llama_server()
        self.device = resolve_device(cfg, self.binary)
        self.proc: subprocess.Popen | None = None
        self.log_path = PROJECT_ROOT / "logs" / "llama-server.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def build_args(self) -> list[str]:
        cfg = self.cfg
        model = cfg.model
        args = [
            str(self.binary),
            "-m",
            str(model.gguf_path),
            "-c",
            str(model.ctx_size),
            "-b",
            str(model.batch_size),
            "-ub",
            str(model.ubatch_size),
            "-np",
            str(model.parallel),
            "-ctk",
            model.cache_type_k,
            "-ctv",
            model.cache_type_v,
            "--host",
            cfg.server.host,
            "--port",
            str(cfg.server.port),
            "--jinja",
        ]
        if model.mmproj_path and model.mmproj_path.exists():
            args += ["-mm", str(model.mmproj_path)]
        else:
            log.warning("mmproj not found (%s); image inputs disabled", model.mmproj_path)

        args += self._device_args()
        return args + list(cfg.server.extra_args)

    def _device_args(self) -> list[str]:
        cfg = self.cfg
        device = self.device
        args: list[str] = []

        if device == "cpu":
            args += [
                "--device",
                "none",
                "--no-mmproj-offload",
                "-ngl",
                "0",
                "-t",
                str(cfg.threads or physical_cpu_count()),
            ]
        else:
            indices = self._cuda_indices(device)
            if indices is not None:
                args += ["--device", ",".join(f"CUDA{i}" for i in indices)]
            ngl = cfg.gpu_layers
            if ngl is None or (isinstance(ngl, str) and ngl.strip().lower() == "auto"):
                # Leave n_gpu_layers unset so llama.cpp's --fit (on by default)
                # offloads as many layers as free VRAM allows. This still puts
                # every layer on an empty GPU, but degrades to partial offload
                # instead of failing when another process holds GPU memory.
                pass
            elif isinstance(ngl, str):
                args += ["-ngl", ngl.strip()]
            else:
                args += ["-ngl", str(int(ngl))]
            if cfg.tensor_split:
                args += ["-ts", str(cfg.tensor_split)]
            if cfg.main_gpu is not None and (indices is None or len(indices) > 1):
                args += ["-mg", str(cfg.main_gpu)]

        if cfg.flash_attn and cfg.flash_attn != "auto":
            args += ["-fa", str(cfg.flash_attn)]
        return args

    @staticmethod
    def _cuda_indices(device: str) -> list[int] | None:
        if device in ("cuda", "cuda:all"):
            return None
        if not device.startswith("cuda:"):
            raise LlamaServerError(f"unsupported device: {device!r}")
        spec = device.split(":", 1)[1]
        return [int(x) for x in spec.split(",") if x.strip()]

    @property
    def base_url(self) -> str:
        return self.cfg.server.base_url

    # ------------------------------------------------------------------
    @staticmethod
    def _pdeathsig():
        """Kill the child if its parent dies (Linux PR_SET_PDEATHSIG)."""
        if not sys.platform.startswith("linux"):
            return None

        def _set() -> None:
            try:
                import ctypes

                libc = ctypes.CDLL("libc.so.6", use_errno=True)
                libc.prctl(1, signal.SIGKILL)  # PR_SET_PDEATHSIG
            except Exception:  # noqa: BLE001 - best effort only
                pass

        return _set

    def start(self, wait: bool = True) -> "LlamaServer":
        if self.proc and self.proc.poll() is None:
            return self
        if not self.cfg.model.gguf_path.exists():
            raise LlamaServerError(
                f"model not found: {self.cfg.model.gguf_path}. "
                "Run scripts/02_download_model.sh first."
            )
        if self._server_already_up():
            raise LlamaServerError(
                f"a llama-server already answers at {self.base_url}. Stop it "
                "(or change server.port) before starting another one."
            )
        args = self.build_args()
        log.info("starting llama-server on %s (%s)", self.base_url, self.device)
        log.debug("cmd: %s", " ".join(args))
        self._log_file = open(self.log_path, "w")  # noqa: SIM115 - kept open for process
        env = dict(os.environ)
        self.proc = subprocess.Popen(
            args,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            env=env,
            start_new_session=True,
            preexec_fn=self._pdeathsig(),
        )
        if wait:
            self.wait_ready()
        return self

    def _server_already_up(self) -> bool:
        try:
            r = httpx.get(f"{self.base_url}/health", timeout=1.5)
            return r.status_code in (200, 503)
        except httpx.HTTPError:
            return False

    def wait_ready(self, timeout: float | None = None) -> None:
        timeout = timeout or self.cfg.server.startup_timeout
        deadline = time.time() + timeout
        url = f"{self.base_url}/health"
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                tail = self._tail_log()
                raise LlamaServerError(
                    f"llama-server exited with code {self.proc.returncode}\n{tail}"
                )
            try:
                r = httpx.get(url, timeout=2)
                if r.status_code == 200 and r.json().get("status") in ("ok", "loading"):
                    if r.json().get("status") == "ok":
                        log.info("llama-server ready at %s", self.base_url)
                        return
            except httpx.HTTPError:
                pass
            time.sleep(0.5)
        raise LlamaServerError(f"llama-server not ready after {timeout:.0f}s")

    def stop(self) -> None:
        if not self.proc:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        self.proc = None
        log_file = getattr(self, "_log_file", None)
        if log_file:
            log_file.close()

    def _tail_log(self, lines: int = 30) -> str:
        try:
            content = self.log_path.read_text(errors="replace").splitlines()
            return "\n".join(content[-lines:])
        except OSError:
            return ""

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
