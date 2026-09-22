"""Configuration loading for JEV.

Precedence: keyword overrides > environment (JEV_*) > config.yaml > defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


@dataclass
class ModelConfig:
    gguf: str = "models/gguf/Qwen3.8-4B-Q8_0.gguf"
    mmproj: str | None = "models/gguf/mmproj-Qwen3.5-4B-BF16.gguf"
    ctx_size: int = 8192
    batch_size: int = 2048
    ubatch_size: int = 512
    parallel: int = 4
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"

    @property
    def gguf_path(self) -> Path:
        return _resolve(self.gguf)

    @property
    def mmproj_path(self) -> Path | None:
        if not self.mmproj:
            return None
        return _resolve(self.mmproj)


@dataclass
class ServerConfig:
    host: str = "127.0.0.1"
    port: int = 8080
    startup_timeout: float = 600
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class StrategyConfig:
    """One reasoning-skipping recipe.

    template_kwargs are passed to the chat template (e.g. enable_thinking).
    inject is appended verbatim after the rendered generation prompt.
    system_suffix is appended to the system message.
    """

    name: str = ""
    description: str = ""
    template_kwargs: dict[str, Any] | None = None
    inject: str | None = None
    system_suffix: str | None = None


def default_strategies() -> dict[str, StrategyConfig]:
    # Cues end with a newline: appending an option letter to a prompt ending
    # in ':' can merge into a different token, which the boundary validator
    # rejects (and a merged token would corrupt the readout).
    cue = "Answer with one option letter only.\nOption:\n"
    return {
        "off": StrategyConfig(
            name="off",
            description="chat-template enable_thinking=false only",
            template_kwargs={"enable_thinking": False},
        ),
        "think_skip_cue": StrategyConfig(
            name="think_skip_cue",
            description="closed <think> with a skip instruction + answer cue",
            template_kwargs={"enable_thinking": False},
            inject=f"<think>\nSkip internal reasoning.\n</think>\n\n{cue}",
        ),
        "think_empty_cue": StrategyConfig(
            name="think_empty_cue",
            description="empty closed <think> block + answer cue",
            template_kwargs={"enable_thinking": False},
            inject=f"<think>\n\n</think>\n\n{cue}",
        ),
        "answer_cue": StrategyConfig(
            name="answer_cue",
            description="bare answer cue, no think block",
            template_kwargs={"enable_thinking": False},
            inject="\nFinal answer:\n",
        ),
        "no_think": StrategyConfig(
            name="no_think",
            description="Qwen-style /no_think soft switch",
            template_kwargs={"enable_thinking": False},
            inject="\n/no_think\nOption:\n",
        ),
        "system_no_reasoning": StrategyConfig(
            name="system_no_reasoning",
            description="system prompt forbids reasoning",
            template_kwargs={"enable_thinking": False},
            system_suffix=" Do not reason. Output the option letter immediately.",
        ),
    }


@dataclass
class ThinkingConfig:
    mode: str = "auto"
    auto_order: list[str] = field(
        default_factory=lambda: [
            "system_no_reasoning",
            "think_skip_cue",
            "think_empty_cue",
            "answer_cue",
            "no_think",
            "off",
        ]
    )
    strategies: dict[str, StrategyConfig] = field(default_factory=default_strategies)
    think_markers: list[str] = field(
        default_factory=lambda: ["<think", "<|channel>", "thinking"]
    )

    def get(self, name: str) -> StrategyConfig:
        if name == "bypass":
            return self.strategies.get("think_skip_cue", default_strategies()["think_skip_cue"])
        if name in self.strategies:
            return self.strategies[name]
        raise ValueError(
            f"unknown thinking strategy {name!r}; "
            f"available: {sorted(self.strategies)} plus 'bypass'"
        )


@dataclass
class ImageConfig:
    max_side: int = 512
    max_pixels: int = 262144
    format: str = "png"


@dataclass
class EngineConfig:
    max_options: int = 16
    request_timeout: float = 180
    top_probs: int = 16


@dataclass
class ApiConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class Config:
    device: str = "auto"
    gpu_layers: int | str | None = "auto"
    tensor_split: str | None = None
    main_gpu: int | None = 0
    threads: int | None = None
    flash_attn: str = "auto"

    model: ModelConfig = field(default_factory=ModelConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    thinking: ThinkingConfig = field(default_factory=ThinkingConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    api: ApiConfig = field(default_factory=ApiConfig)

    # Derived, not user-facing.
    resolved_device: str = ""

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        cfg_path = Path(path) if path else DEFAULT_CONFIG
        raw: dict[str, Any] = {}
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text()) or {}
        sectioned = {
            "model": ModelConfig,
            "server": ServerConfig,
            "thinking": ThinkingConfig,
            "image": ImageConfig,
            "engine": EngineConfig,
            "api": ApiConfig,
        }
        kwargs: dict[str, Any] = {}
        for key, value in raw.items():
            if key in sectioned:
                continue
            kwargs[key] = value
        cfg = cls(**kwargs)
        for key, klass in sectioned.items():
            if key in raw and isinstance(raw[key], dict):
                section = dict(raw[key])
                if key == "thinking":
                    cfg.thinking = _parse_thinking(section)
                else:
                    setattr(cfg, key, klass(**section))

        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        env = os.environ
        if device := env.get("JEV_DEVICE"):
            self.device = device
        if host := env.get("JEV_HOST"):
            self.server.host = host
            self.api.host = host
        if port := env.get("JEV_PORT"):
            self.server.port = int(port)
        if api_port := env.get("JEV_API_PORT"):
            self.api.port = int(api_port)
        if gguf := env.get("JEV_GGUF"):
            self.model.gguf = gguf
        if mmproj := env.get("JEV_MMPROJ"):
            self.model.mmproj = mmproj or None
        if ctx := env.get("JEV_CTX_SIZE"):
            self.model.ctx_size = int(ctx)
        if ngl := env.get("JEV_GPU_LAYERS"):
            self.gpu_layers = int(ngl)
        if threads := env.get("JEV_THREADS"):
            self.threads = int(threads)
        if ts := env.get("JEV_TENSOR_SPLIT"):
            self.tensor_split = ts
        if mode := env.get("JEV_THINKING_MODE"):
            self.thinking.mode = mode

    # ------------------------------------------------------------------
    @property
    def is_cpu(self) -> bool:
        return self.resolved_device == "cpu"

    def describe_device(self) -> str:
        return self.resolved_device or self.device


def _yaml_name(name: Any) -> str:
    """YAML 1.1 parses bare off/on as booleans; normalize them back."""
    if name is False:
        return "off"
    if name is True:
        return "on"
    return str(name)


def _parse_thinking(section: dict[str, Any]) -> ThinkingConfig:
    strategies_raw = section.pop("strategies", None)
    if "auto_order" in section and isinstance(section["auto_order"], list):
        section["auto_order"] = [_yaml_name(x) for x in section["auto_order"]]
    cfg = ThinkingConfig(**section)
    if isinstance(strategies_raw, dict):
        merged = default_strategies()
        for raw_name, spec in strategies_raw.items():
            name = _yaml_name(raw_name)
            spec = dict(spec or {})
            spec.pop("name", None)
            merged[name] = StrategyConfig(name=name, **spec)
        cfg.strategies = merged
    for name, spec in cfg.strategies.items():
        if not spec.name:
            spec.name = name
    return cfg


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def physical_cpu_count() -> int:
    """Best-effort physical core count (no psutil dependency)."""
    try:
        cores: set[tuple[str, str]] = set()
        phys_id = core_id = None
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("physical id"):
                phys_id = line.split(":")[1].strip()
            elif line.startswith("core id"):
                core_id = line.split(":")[1].strip()
            elif not line.strip() and phys_id is not None and core_id is not None:
                cores.add((phys_id, core_id))
                phys_id = core_id = None
        if cores:
            return len(cores)
    except OSError:
        pass
    count = os.cpu_count() or 4
    return max(1, count // 2)
