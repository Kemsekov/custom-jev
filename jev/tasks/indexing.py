"""Mixed-input indexing evaluation.

The engine has no built-in indexing prompts: the caller supplies the exact
rules and the per-item question. This module owns the fixtures for the test
suite, so it passes its own rule strings explicitly (the same way an API user
would), and never relies on engine-side phrasing.

Two indexings are exercised:
  * photos only - 6 animal photos in shuffled orders;
  * mixed - photos plus short text descriptions in one ordered input list.

Ordering is intentionally not a separate task: it is expressed as an indexing
where the inputs are rank slots and the items are the texts, using exactly the
same generic `index_inputs` call.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from .animals import DEFAULT_DIR, list_animal_images

# Explicit fixture prompts (rules + question in one string). These are what an
# API caller would send; the engine injects no indexing phrasing of its own.
IMAGE_PROMPT = (
    "The inputs are numbered photos in the given order. Assign every requested "
    "item to exactly one input that clearly shows that item; each input is used "
    "at most once. Which input is the photo that shows a {item}?"
)
MIXED_PROMPT = (
    "The inputs are numbered and are either photos or short text descriptions. "
    "Assign every requested item to exactly one input that depicts or describes "
    "that item; each input is used at most once. Which input matches the {item}?"
)
ORDERING_PROMPT = (
    "The inputs are rank slots: input 1 is rank 1 (the most agitated review) "
    "and the last input is the least agitated, most happy review. Assign each "
    "review text to exactly one rank slot so the reviews are ordered from most "
    "agitated to most happy; each rank is used exactly once. Which rank slot "
    "should the following review get? '{item}'"
)


def _image_inputs(images: list[Any]) -> list[dict]:
    return [
        {"type": "image", "image": path, "label": f"photo {i + 1}"}
        for i, path in enumerate(images)
    ]


def run_indexing(
    engine,
    images: list[Any] | None = None,
    trials: int = 3,
    images_per_trial: int = 6,
    seed: int = 7,
    thinking: str | None = None,
    artifacts_key: str = "indexing",
    include_mixed: bool = True,
) -> dict:
    if images is None:
        available = list_animal_images(DEFAULT_DIR)
        images = [path for _, path in available]
    if len(images) < 2:
        return {
            "error": "need at least 2 images; run scripts/05_fetch_assets.sh",
            "trials": [],
            "item_accuracy": 0.0,
        }
    from ..artifacts import TaskArtifacts, contact_sheet

    artifacts = TaskArtifacts(artifacts_key)
    trials_out = []
    totals = {"items": 0, "correct": 0, "argmax_correct": 0, "changed": 0}

    for trial in range(trials):
        rng = random.Random(f"{seed}:trial:{trial}")
        picked = list(images)
        rng.shuffle(picked)
        picked = picked[:images_per_trial]
        labels = [Path(str(p)).stem for p in picked]
        asked = list(labels)
        rng.shuffle(asked)

        inputs = _image_inputs(picked)
        records = engine.index_inputs(
            inputs=inputs,
            items=asked,
            prompt=IMAGE_PROMPT,
            thinking=thinking,
        )
        expected = {label: labels.index(label) + 1 for label in asked}
        _collect(totals, records, expected)
        trials_out.append(
            _trial_record(
                engine, artifacts, trial, "photos", inputs, asked, records, expected
            )
        )

    if include_mixed and len(images) >= 4:
        trial = trials
        rng = random.Random(f"{seed}:mixed")
        picked = list(images)
        rng.shuffle(picked)
        picked = picked[:images_per_trial]
        labels = [Path(str(p)).stem for p in picked]
        # interleave photos with short text descriptions of the next animal
        inputs: list[dict] = []
        expected = {}
        for i, (label, path) in enumerate(zip(labels, picked)):
            if i % 2 == 0:
                inputs.append(
                    {"type": "image", "image": path, "label": f"photo {i + 1}"}
                )
                expected[label] = i + 1
            else:
                inputs.append(
                    {
                        "type": "text",
                        "text": f"a photo of a {label}",
                        "label": f"note {i + 1}",
                    }
                )
                expected[label] = i + 1
        asked = list(expected)
        rng.shuffle(asked)
        records = engine.index_inputs(
            inputs=inputs,
            items=asked,
            prompt=MIXED_PROMPT,
            thinking=thinking,
        )
        _collect(totals, records, expected)
        trials_out.append(
            _trial_record(
                engine,
                artifacts,
                trial,
                "mixed",
                inputs,
                asked,
                records,
                expected,
            )
        )

    report = {
        "trials": trials_out,
        "n": totals["items"],
        "item_accuracy": totals["correct"] / totals["items"] if totals["items"] else 0.0,
        "argmax_item_accuracy": (
            totals["argmax_correct"] / totals["items"] if totals["items"] else 0.0
        ),
        "items_changed_by_assignment": totals["changed"],
        "perfect_trials": sum(1 for t in trials_out if t["perfect"]),
        "text": "\n".join(
            f"{item}:{idx}"
            for t in trials_out
            for item, idx in t["assignments"].items()
        ),
    }
    artifacts.write_json("trials.json", trials_out)
    artifacts.write_text("assignments.txt", report["text"] + "\n")
    artifacts.summary(
        item_accuracy=report["item_accuracy"],
        perfect_trials=report["perfect_trials"],
    )
    report["artifacts"] = str(artifacts.dir)
    return report


def _collect(totals: dict, records: list[dict], expected: dict[str, int]) -> None:
    totals["items"] += len(records)
    totals["correct"] += sum(
        1 for r in records if r["index"] == expected[r["item"]]
    )
    totals["argmax_correct"] += sum(
        1 for r in records if r["argmax_index"] == expected[r["item"]]
    )
    totals["changed"] += sum(1 for r in records if r["index_changed"])


def _trial_record(
    engine,
    artifacts,
    trial: int,
    kind: str,
    inputs: list[dict],
    asked: list[str],
    records: list[dict],
    expected: dict[str, int],
) -> dict:
    from ..artifacts import contact_sheet

    assignments = {r["item"]: r["index"] for r in records}
    argmax_assignments = {r["item"]: r["argmax_index"] for r in records}
    correct = sum(1 for item in asked if assignments.get(item) == expected[item])

    image_entries = []
    for position, spec in enumerate(inputs, start=1):
        if spec["type"] != "image":
            continue
        item = next(
            (label for label, idx in expected.items() if idx == position), f"input {position}"
        )
        got = assignments.get(item)
        image_entries.append(
            {
                "image": spec["image"],
                "title": f"input {position} ({spec.get('label')})",
                "subtitle": (
                    f"truth {item} | got input {got}"
                    if got == expected[item]
                    else f"truth {item} | got input {got}  WRONG"
                ),
                "ok": got == expected[item],
            }
        )
    if image_entries:
        artifacts.save_image(
            f"trial-{trial}-{kind}-order.png", contact_sheet(image_entries, columns=3)
        )

    input_lines = ["inputs:"]
    for position, spec in enumerate(inputs, start=1):
        if spec["type"] == "image":
            detail = str(spec["image"])
        else:
            detail = f"text: {spec['text']}"
        input_lines.append(f"  {position}. [{spec['type']}] {spec.get('label')}: {detail}")
    artifacts.write_text(
        f"trial-{trial}-{kind}-assignments.txt",
        "\n".join(input_lines)
        + "\n\nasked order:\n"
        + "\n".join(f"  {i + 1}. {a}" for i, a in enumerate(asked))
        + "\n\nassignments (item:input):\n"
        + "\n".join(f"{item}:{assignments[item]}" for item in asked)
        + "\n\nexpected (item:input):\n"
        + "\n".join(f"{item}:{expected[item]}" for item in asked)
        + "\n",
    )
    return {
        "trial": trial,
        "kind": kind,
        "inputs": inputs,
        "asked_order": asked,
        "expected": expected,
        "assignments": assignments,
        "argmax_assignments": argmax_assignments,
        "assignment_method": records[0]["assignment_method"] if records else None,
        "correct": correct,
        "argmax_correct": sum(
            1 for item in asked if argmax_assignments.get(item) == expected[item]
        ),
        "changed_by_assignment": sum(1 for r in records if r["index_changed"]),
        "n": len(asked),
        "perfect": correct == len(asked),
        "cached_tokens_per_decision": [r["cached_tokens"] for r in records],
        "confidence": {r["item"]: round(r["confidence"], 4) for r in records},
    }


def run_ordering_as_indexing(
    engine,
    items: list[str],
    thinking: str | None = None,
) -> dict:
    """Demonstrate ordering through the generic indexing call.

    The inputs are rank slots (text inputs), the items are the texts to rank.
    The caller supplies the same kind of explicit rules it would send to
    /v1/index.
    """
    inputs = [
        {
            "type": "text",
            "text": (
                "rank 1 = most agitated"
                if i == 0
                else f"rank {i + 1} = {i + 1}th most agitated"
            ),
            "label": f"rank {i + 1}",
        }
        for i in range(len(items))
    ]
    records = engine.index_inputs(
        inputs=inputs,
        items=items,
        prompt=ORDERING_PROMPT,
        thinking=thinking,
    )
    mapping = {r["item"]: r["index"] for r in records}
    return {
        "order": sorted(items, key=lambda text: mapping[text]),
        "mapping": mapping,
        "confidence": {r["item"]: round(r["confidence"], 4) for r in records},
    }
