"""Pydantic schemas for the JEV HTTP API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

ThinkingMode = Literal["auto", "off", "bypass"]


class OptionIn(BaseModel):
    id: str = Field(description="Stable option id echoed back in the result.")
    description: str = Field(description="What this option means.")


class EvidenceIn(BaseModel):
    state: Any = Field(
        default=None,
        description="Text, JSON object or array with the evidence to judge.",
    )
    image_base64: str | None = Field(
        default=None,
        description="Base64 image (raw or data URL) to judge together with the state.",
    )
    image_path: str | None = Field(
        default=None, description="Server-local image path (alternative to base64)."
    )
    images_base64: list[str] | None = Field(
        default=None,
        description=(
            "Multiple base64 images (raw or data URLs). Order is preserved; "
            "option descriptions can refer to image 1, image 2, ..."
        ),
    )
    images_paths: list[str] | None = Field(
        default=None,
        description="Multiple server-local image paths, in order.",
    )


class DecideRequest(EvidenceIn):
    id: str | None = None
    question: str = Field(description="The criterion / question to apply.")
    options: list[OptionIn] = Field(description="2-16 typed options.")
    thinking: ThinkingMode | None = Field(
        default=None,
        description="Override the engine's resolved thinking strategy.",
    )


class BatchItem(BaseModel):
    id: str | None = None
    question: str
    options: list[OptionIn]


class BatchDecideRequest(EvidenceIn):
    items: list[BatchItem] = Field(description="Multiple criteria over the same evidence.")
    thinking: ThinkingMode | None = None
    assignment: Literal["none", "hungarian"] = Field(
        default="none",
        description=(
            "none: independent argmax per item (two items may share an answer). "
            "hungarian: the answers are a one-to-one matching; requires every "
            "item to use the same option ids and solves the item x option "
            "probability matrix with scipy.optimize.linear_sum_assignment."
        ),
    )
    objective: Literal["logprob", "prob"] = "logprob"


class IndexInput(BaseModel):
    type: Literal["image", "text"] | None = Field(
        default=None, description="Inferred from the fields when omitted."
    )
    image_base64: str | None = None
    image_path: str | None = None
    text: str | None = None
    label: str | None = Field(
        default=None,
        description=(
            "Short description shown to the model for this input "
            "(default: 'input N')."
        ),
    )


class IndexRequest(BaseModel):
    inputs: list[IndexInput] = Field(
        description="Ordered mixed inputs (2-16 images and/or texts)."
    )
    items: list[str] = Field(
        description=(
            "Items to assign, one per JEV decision. Strings can be short "
            "labels or long texts (e.g. review bodies for ranking)."
        )
    )
    prompt: str = Field(
        description=(
            "The whole indexing instruction - rules and question in one "
            "string. Must contain the '{item}' placeholder, which is replaced "
            "per item, e.g. 'Each input is a photo. Which input shows a "
            "{item}? Use each input at most once.'"
        )
    )
    assignment: Literal["hungarian", "argmax"] = Field(
        default="hungarian",
        description=(
            "hungarian: solve the one-to-one item/input assignment over the "
            "full probability matrix (SciPy linear_sum_assignment). "
            "argmax: independent per-item argmax (can collide)."
        ),
    )
    objective: Literal["logprob", "prob"] = Field(
        default="logprob",
        description=(
            "Assignment objective: logprob maximises sum(log P) (MAP over "
            "permutations); prob maximises the expected number of matches."
        ),
    )
    thinking: ThinkingMode | None = None


class DiagnoseRequest(BaseModel):
    strategies: list[str] = Field(
        default_factory=lambda: ["off", "bypass", "on"],
        description=(
            "Any configured strategy name (see /v1/info), or 'on' to probe "
            "with reasoning enabled."
        ),
    )


class MazeRequest(BaseModel):
    width: int = Field(default=5, ge=3, le=12)
    height: int = Field(default=5, ge=3, le=12)
    seed: int = 0
    max_steps: int = Field(default=40, ge=1, le=200)
    cell_px: int = Field(default=96, ge=24, le=160)
    thinking: ThinkingMode | None = None
    save_images: bool = True
    legal_moves_only: bool = Field(
        default=True,
        description=(
            "true: only offer open moves (tests planning); "
            "false: offer all four directions (also tests wall perception)"
        ),
    )


class LatencyBenchRequest(BaseModel):
    token_targets: list[int] = Field(default_factory=lambda: [32, 128, 512, 1024])
    repeats: int = Field(default=1, ge=1, le=10)
