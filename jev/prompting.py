"""JEV prompt construction and reasoning-skipping strategies.

The contract: after rendering, the very next token must be the option letter.
Several named strategies enforce that, because a reasoning-distilled model can
insist on opening <think> even when the template suppresses it. Strategies are
benchmarked against each other (see `jev evaluate --thinking all`) instead of
being hardcoded.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .config import StrategyConfig

LETTERS = "ABCDEFGHIJKLMNOP"

# Compact JSON saves prompt tokens but measurably hurts decision accuracy
# (0.722 vs 0.833 on the authored suite), so the spaced form is kept.
COMPACT_JSON = False

SYSTEM_TEXT = (
    "Apply the criterion to the evidence. Choose exactly one option. "
    "Answer with only its uppercase letter, no explanation or reasoning."
)

SYSTEM_VISION = (
    "Apply the criterion to the image and evidence. Choose exactly one option. "
    "Answer with only its uppercase letter, no explanation or reasoning."
)


@dataclass
class RenderedPrompt:
    prompt: str
    strategy: str
    template_kwargs: dict[str, Any]
    injected: bool
    description: str = ""


@dataclass
class ProbeResult:
    strategy: str
    thinking_detected: bool
    letter_top1: bool
    letter_mass: float
    boundary_ok: bool = True
    note: str = ""
    top_tokens: list[dict] = field(default_factory=list)
    inject: str | None = None

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "thinking_detected": self.thinking_detected,
            "letter_top1": self.letter_top1,
            "letter_mass": round(self.letter_mass, 5),
            "boundary_ok": self.boundary_ok,
            "note": self.note,
            "top_tokens": self.top_tokens,
            "inject": self.inject,
        }


def render_prompt(
    client,
    messages: list[dict],
    spec: StrategyConfig,
    supports_thinking: bool,
) -> RenderedPrompt:
    """Render the chat template and apply one reasoning-skipping strategy."""
    messages = [dict(m) for m in messages]
    if spec.system_suffix:
        for msg in messages:
            if msg.get("role") == "system":
                msg["content"] = f"{msg['content']}{spec.system_suffix}"
                break
        else:
            messages.insert(0, {"role": "system", "content": spec.system_suffix.strip()})

    kwargs = dict(spec.template_kwargs or {})
    if not supports_thinking:
        kwargs.pop("enable_thinking", None)

    prompt = client.apply_template(messages, chat_template_kwargs=kwargs)
    injected = False
    if spec.inject:
        prompt = prompt + spec.inject
        injected = True
    return RenderedPrompt(
        prompt=prompt,
        strategy=spec.name or "custom",
        template_kwargs=kwargs,
        injected=injected,
        description=spec.description,
    )


def build_messages(
    state: Any,
    question: str,
    options: list[dict],
    image_data_url: str | None = None,
    image_data_urls: list[str] | None = None,
) -> list[dict]:
    urls = list(image_data_urls or [])
    if image_data_url:
        urls.insert(0, image_data_url)
    payload = {
        "evidence": state,
        "criterion": question,
        "options": [
            {"letter": LETTERS[i], "description": opt["description"]}
            for i, opt in enumerate(options)
        ],
    }
    # Compact separators save prompt tokens: every token is prefill time.
    text = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":") if COMPACT_JSON else None,
    )
    system = SYSTEM_VISION if urls else SYSTEM_TEXT
    if urls:
        # Images appear in order; option descriptions refer to their index.
        user_content: Any = [
            {"type": "image_url", "image_url": {"url": url}} for url in urls
        ]
        user_content.append({"type": "text", "text": text})
    else:
        user_content = text
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_content},
    ]


def build_grammar(n_options: int) -> str:
    """GBNF grammar that allows exactly the option-letter tokens."""
    choices = " | ".join(f'"{LETTERS[i]}"' for i in range(n_options))
    return f"root ::= {choices}"


def template_supports_thinking(chat_template: str) -> bool:
    return "enable_thinking" in (chat_template or "")


def looks_like_thinking(text: str, markers: list[str]) -> bool:
    low = (text or "").lower()
    return any(marker.lower() in low for marker in markers)
