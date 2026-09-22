"""Animal image classification as a JEV task.

Images come from Wikimedia Commons (setup-time fetch, one request per label).
Each image becomes a decision with the true label plus distractor labels, so
the engine is scored exactly like a human multiple-choice quiz.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import httpx

from ..config import PROJECT_ROOT

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "jev-multimodal-eval/0.1 (research; contact: local user)"

DEFAULT_LABELS = [
    "cat",
    "dog",
    "horse",
    "cow",
    "sheep",
    "elephant",
    "penguin",
    "owl",
    "frog",
    "butterfly",
    "goldfish",
    "tiger",
]

DEFAULT_DIR = PROJECT_ROOT / "data" / "images" / "animals"


def fetch_animal_images(
    out_dir: Path | None = None,
    labels: list[str] | None = None,
    width: int = 512,
) -> dict[str, str]:
    """Download one public-domain/CC image per label from Wikimedia Commons."""
    out_dir = out_dir or DEFAULT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = labels or DEFAULT_LABELS
    saved: dict[str, str] = {}
    attribution: dict[str, Any] = {}

    with httpx.Client(timeout=40, headers={"User-Agent": USER_AGENT}) as client:
        for label in labels:
            target = out_dir / f"{label}.jpg"
            if target.exists():
                saved[label] = str(target)
                continue
            try:
                thumb, meta = _search_commons_image(client, label, width)
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"[warn] could not fetch image for {label!r}: {exc}")
                continue
            response = client.get(thumb)
            response.raise_for_status()
            target.write_bytes(response.content)
            saved[label] = str(target)
            attribution[label] = meta
            print(f"[ok] {label}: {meta.get('title', thumb)}")

    if attribution:
        attr_file = out_dir / "attribution.json"
        existing = {}
        if attr_file.exists():
            existing = json.loads(attr_file.read_text())
        existing.update(attribution)
        attr_file.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return saved


def _search_commons_image(client: httpx.Client, label: str, width: int) -> tuple[str, dict]:
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": f"filetype:bitmap {label} animal",
        "gsrnamespace": "6",
        "gsrlimit": "5",
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": str(width),
        "format": "json",
    }
    data = client.get(COMMONS_API, params=params).json()
    pages = list((data.get("query") or {}).get("pages", {}).values())
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        thumb = info.get("thumburl") or info.get("url")
        if not thumb:
            continue
        meta = info.get("extmetadata") or {}
        return thumb, {
            "title": page.get("title"),
            "page_url": info.get("descriptionurl"),
            "license": (meta.get("LicenseShortName") or {}).get("value"),
            "artist": (meta.get("Artist") or {}).get("value"),
        }
    raise RuntimeError("no bitmap result found")


def list_animal_images(images_dir: Path | None = None) -> list[tuple[str, Path]]:
    images_dir = images_dir or DEFAULT_DIR
    if not images_dir.exists():
        return []
    out = []
    for path in sorted(images_dir.glob("*")):
        if path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
            out.append((path.stem, path))
    return out


def build_rows(
    images: list[tuple[str, Path]],
    option_count: int = 4,
    seed: int = 7,
) -> list[dict]:
    labels = [label for label, _ in images]
    rows = []
    for label, path in images:
        rng = random.Random(f"{seed}:{label}")
        distractors = [x for x in labels if x != label]
        rng.shuffle(distractors)
        options = [{"id": label, "description": label.replace("_", " ")}]
        for distractor in distractors[: option_count - 1]:
            options.append(
                {"id": distractor, "description": distractor.replace("_", " ")}
            )
        rng.shuffle(options)
        rows.append(
            {
                "id": f"animal-{label}",
                "image": str(path),
                "question": "What animal is shown in the image?",
                "options": options,
                "expected": label,
            }
        )
    return rows


def run_animals(
    engine,
    images_dir: Path | None = None,
    option_count: int = 4,
    thinking: str | None = None,
    limit: int | None = None,
    artifacts_key: str = "animals",
) -> dict:
    images = list_animal_images(images_dir)
    if not images:
        return {
            "error": (
                f"no animal images in {images_dir or DEFAULT_DIR}; run "
                "scripts/05_fetch_assets.sh"
            ),
            "n": 0,
            "accuracy": 0.0,
            "results": [],
        }
    from ..artifacts import TaskArtifacts, contact_sheet

    artifacts = TaskArtifacts(artifacts_key)
    rows = build_rows(images, option_count=option_count)
    if limit:
        rows = rows[:limit]
    results = []
    for row in rows:
        decision = engine.decide(
            question=row["question"],
            options=row["options"],
            state=None,
            image=row["image"],
            thinking=thinking,
            decision_id=row["id"],
        )
        results.append(
            {
                "id": row["id"],
                "image": row["image"],
                "expected": row["expected"],
                "chosen": decision.chosen,
                "correct": decision.chosen == row["expected"],
                "confidence": decision.confidence,
                "probabilities": {o.id: o.probability for o in decision.options},
                "prompt_ms": decision.timings.get("prompt_ms", 0.0),
                "wall_ms": decision.timings.get("wall_ms", 0.0),
                "prompt_tokens": decision.tokens_evaluated,
                "cached_tokens": decision.cached_tokens,
            }
        )
    correct = sum(1 for r in results if r["correct"])
    report = {
        "n": len(results),
        "accuracy": correct / len(results) if results else 0.0,
        "results": results,
    }
    artifacts.write_jsonl("decisions.jsonl", results)
    sheet_entries = [
        {
            "image": row["image"],
            "title": f"{row['id']} (truth: {row['expected']})",
            "subtitle": (
                f"chosen {row['chosen']} ({row['confidence']:.2f})"
                + ("" if row["correct"] else "  WRONG")
            ),
            "ok": row["correct"],
        }
        for row in results
    ]
    artifacts.save_image("contact-sheet.png", contact_sheet(sheet_entries))
    artifacts.summary(accuracy=report["accuracy"], n=report["n"])
    report["artifacts"] = str(artifacts.dir)
    return report
