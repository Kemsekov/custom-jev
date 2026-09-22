"""Decision-shape performance benchmark.

Measures the wall/prefill latency and cache behaviour of every decision shape
the API exposes, so the cost of each pattern is explicit:

  * one text decision (cold and warm)
  * one image decision (cold and warm)
  * decide_many over one text state / one image, sequential vs parallel slots
  * index_inputs with 6 photos / mixed photos+texts / rank-slot texts,
    sequential vs parallel
  * the Hungarian assignment itself (6x6 and 16x16), which is pure CPU

Important reading note: `prompt_tokens` is the number of prompt tokens the
model actually *evaluated*; `reused_tokens` is how many were served from the
prompt cache. On Qwen3.5 hybrid attention partial cache reuse does not happen
for diverging prompts (and multimodal prompts disable it entirely), so
`reused_tokens` is ~0 except for a repeated identical prompt - which is why
parallel slots, not cache branching, are the throughput lever here.

Every measurement is a real model call on the configured device. Results are
written to results/artifacts/perf/ (wiped per run).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..artifacts import TaskArtifacts
from ..assignment import optimal_assignment
from .animals import DEFAULT_DIR, list_animal_images
from .indexing import IMAGE_PROMPT, MIXED_PROMPT, ORDERING_PROMPT

TEXT_STATE = (
    "Checkout API returning 500 for 100% of requests in production for 12 "
    "minutes. The last deploy changed the payment provider configuration."
)
TEXT_OPTIONS = [
    {"id": "sev1", "description": "sev1 - total outage"},
    {"id": "sev2", "description": "sev2 - major degradation"},
    {"id": "sev3", "description": "sev3 - minor issue"},
]
MANY_QUESTIONS = [
    "What is the incident severity?",
    "Should the on-call engineer be paged now?",
    "Is a rollback of the last deploy reasonable?",
    "Does this require a customer communication?",
    "Is the issue likely caused by configuration?",
]
MANY_OPTIONS = [
    {"id": "yes", "description": "yes"},
    {"id": "no", "description": "no"},
]
RANK_REVIEWS = [
    "Third replacement unit and it is broken again. I want a refund now!",
    "The battery stopped charging after two weeks.",
    "It works. Nothing special, does what the listing says.",
    "Pretty good for the price, battery could be better.",
    "Absolutely love it, best purchase this year!",
]


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((q / 100) * (len(ordered) - 1)))))
    return round(ordered[idx], 2)


def _entry(result) -> dict:
    return {
        "wall_ms": round(result.timings.get("wall_ms", 0.0), 2),
        "prompt_ms": round(result.timings.get("prompt_ms", 0.0), 2),
        "prompt_tokens": result.timings.get("prompt_n", 0),
        "reused_tokens": result.reused_tokens,
    }


def _summarize(entries: list[dict]) -> dict:
    walls = [e["wall_ms"] for e in entries]
    return {
        "n": len(entries),
        "wall_ms_p50": _pct(walls, 50),
        "wall_ms_mean": round(sum(walls) / len(walls), 2) if walls else 0.0,
        "wall_ms_min": round(min(walls), 2) if walls else 0.0,
        "wall_ms_max": round(max(walls), 2) if walls else 0.0,
        "prompt_ms_p50": _pct([e["prompt_ms"] for e in entries], 50),
        "prompt_tokens_p50": _pct([e["prompt_tokens"] for e in entries], 50),
        "reused_tokens_p50": _pct([e["reused_tokens"] for e in entries], 50),
    }


def _measure_decide(
    engine, *, state=None, image=None, repeats=3, cold=True, readout=None
) -> list[dict]:
    previous = engine.cfg.engine.readout
    if readout:
        engine.cfg.engine.readout = readout
    entries = []
    try:
        for _ in range(repeats):
            result = engine.decide(
                question="What is the incident severity?",
                options=TEXT_OPTIONS,
                state=state,
                image=image,
                cache_prompt=not cold,
            )
            entries.append(_entry(result))
    finally:
        engine.cfg.engine.readout = previous
    return entries


def _measure_many(
    engine, *, questions, options, state=None, image=None, repeats=3, parallel=False
) -> dict:
    entries: list[dict] = []
    totals: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        results = engine.decide_many(
            questions=questions,
            options=[options] * len(questions),
            state=state,
            image=image,
            parallel=parallel,
        )
        totals.append((time.perf_counter() - started) * 1000)
        entries.extend(_entry(r) for r in results)
    firsts = [entries[i] for i in range(0, len(entries), len(questions))]
    rests = [e for i, e in enumerate(entries) if i % len(questions) != 0]
    return {
        "decisions": len(questions),
        "total_ms_p50": _pct(totals, 50),
        "parallel": parallel,
        "first": _summarize(firsts),
        "rest": _summarize(rests),
        "all": _summarize(entries),
    }


def _measure_index(
    engine, *, inputs, items, prompt, repeats=3, parallel=False
) -> dict:
    entries: list[dict] = []
    totals: list[float] = []
    solve_ms: list[float] = []
    option_ids = [f"input-{i + 1}" for i in range(len(inputs))]
    for _ in range(repeats):
        started = time.perf_counter()
        records = engine.index_inputs(
            inputs=inputs,
            items=items,
            prompt=prompt,
            assignment="hungarian",
            parallel=parallel,
        )
        totals.append((time.perf_counter() - started) * 1000)
        entries.extend(
            {
                "wall_ms": round(r.get("wall_ms", 0.0), 2),
                "prompt_ms": round(r.get("prompt_ms", 0.0), 2),
                "prompt_tokens": int(r["prompt_tokens"]) - 1,
                "reused_tokens": r.get("reused_tokens", 0),
            }
            for r in records
        )
        rows = {r["item"]: r["probabilities"] for r in records}
        t0 = time.perf_counter()
        optimal_assignment([r["item"] for r in records], option_ids, rows)
        solve_ms.append((time.perf_counter() - t0) * 1000)
    firsts = [entries[i] for i in range(0, len(entries), len(items))]
    rests = [e for i, e in enumerate(entries) if i % len(items) != 0]
    return {
        "decisions": len(items),
        "total_ms_p50": _pct(totals, 50),
        "parallel": parallel,
        "first": _summarize(firsts),
        "rest": _summarize(rests),
        "all": _summarize(entries),
        "assignment_solve_ms_p50": _pct(solve_ms, 50),
    }


def _synthetic_assignments() -> dict:
    """Pure-CPU cost of the Hungarian solve, no model involved."""
    import numpy as np

    out = {}
    rng = np.random.default_rng(0)
    for n in (6, 16):
        rows = {
            f"item-{i}": {
                f"input-{j + 1}": float(v) for j, v in enumerate(rng.random(n))
            }
            for i in range(n)
        }
        samples = []
        for _ in range(200):
            t0 = time.perf_counter()
            optimal_assignment(
                [f"item-{i}" for i in range(n)],
                [f"input-{j + 1}" for j in range(n)],
                rows,
            )
            samples.append((time.perf_counter() - t0) * 1000)
        out[f"{n}x{n}"] = {
            "mean_ms": round(sum(samples) / len(samples), 4),
            "max_ms": round(max(samples), 4),
        }
    return out


def _index_fixtures(images: list[Any]) -> dict[str, dict]:
    mixed_inputs: list[dict] = []
    mixed_items: list[str] = []
    for i, path in enumerate(images[:6]):
        label = Path(str(path)).stem
        if i % 2 == 0:
            mixed_inputs.append(
                {"type": "image", "image": path, "label": f"input {i + 1}"}
            )
        else:
            mixed_inputs.append(
                {"type": "text", "text": f"a photo of a {label}", "label": f"input {i + 1}"}
            )
        mixed_items.append(label)
    return {
        "index_photos_6x6": {
            "inputs": [
                {"type": "image", "image": path, "label": f"photo {i + 1}"}
                for i, path in enumerate(images[:6])
            ],
            "items": [Path(str(p)).stem for p in images[:6]],
            "prompt": IMAGE_PROMPT,
        },
        "index_mixed_6": {
            "inputs": mixed_inputs,
            "items": mixed_items,
            "prompt": MIXED_PROMPT,
        },
        "index_text_ordering_5": {
            "inputs": [
                {"type": "text", "text": f"rank {i + 1}", "label": f"rank {i + 1}"}
                for i in range(len(RANK_REVIEWS))
            ],
            "items": RANK_REVIEWS,
            "prompt": ORDERING_PROMPT,
        },
    }


def run_perf(engine, repeats: int = 3, artifacts_key: str = "perf") -> dict:
    artifacts = TaskArtifacts(artifacts_key)
    images = [path for _, path in list_animal_images(DEFAULT_DIR)]
    if len(images) < 6:
        raise RuntimeError(
            "need 6 animal images for the perf benchmark; run scripts/05_fetch_assets.sh"
        )

    shapes: dict[str, Any] = {}
    print("measuring decide (text)...")
    shapes["decide_text_cold"] = _summarize(
        _measure_decide(engine, state=TEXT_STATE, repeats=repeats, cold=True)
    )
    shapes["decide_text_warm"] = _summarize(
        _measure_decide(engine, state=TEXT_STATE, repeats=repeats, cold=False)
    )
    print("measuring decide (image)...")
    shapes["decide_image_cold"] = _summarize(
        _measure_decide(engine, image=images[0], repeats=repeats, cold=True)
    )
    shapes["decide_image_warm"] = _summarize(
        _measure_decide(engine, image=images[0], repeats=repeats, cold=False)
    )
    print("measuring decide (text, forced grammar readout)...")
    shapes["decide_text_grammar_cold"] = _summarize(
        _measure_decide(
            engine, state=TEXT_STATE, repeats=repeats, cold=True, readout="grammar"
        )
    )
    shapes["decide_image_grammar_cold"] = _summarize(
        _measure_decide(
            engine, image=images[0], repeats=repeats, cold=True, readout="grammar"
        )
    )

    fixtures = _index_fixtures(images)
    for parallel in (False, True):
        tag = "par" if parallel else "seq"
        print(f"measuring decide_many ({tag})...")
        shapes[f"decide_many_text_{tag}"] = _measure_many(
            engine,
            questions=MANY_QUESTIONS,
            options=MANY_OPTIONS,
            state=TEXT_STATE,
            repeats=repeats,
            parallel=parallel,
        )
        shapes[f"decide_many_image_{tag}"] = _measure_many(
            engine,
            questions=MANY_QUESTIONS,
            options=TEXT_OPTIONS,
            image=images[0],
            repeats=repeats,
            parallel=parallel,
        )
        for name, fixture in fixtures.items():
            print(f"measuring {name} ({tag})...")
            shapes[f"{name}_{tag}"] = _measure_index(
                engine, **fixture, repeats=repeats, parallel=parallel
            )

    shapes["assignment_solve"] = _synthetic_assignments()

    report = {"repeats": repeats, "shapes": shapes}
    artifacts.write_json("perf.json", report)
    artifacts.write_text("perf.md", render_markdown(report))
    artifacts.summary(repeats=repeats)
    report["artifacts"] = str(artifacts.dir)
    return report


def render_markdown(report: dict) -> str:
    lines = [
        "# Decision-shape performance",
        "",
        f"repeats per shape: {report['repeats']}",
        "",
        "| shape | decisions | first call ms | follow-up ms | total ms | evaluated tok | reused tok |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, stats in report["shapes"].items():
        if name == "assignment_solve":
            continue
        if "first" in stats:
            first = stats["first"]["wall_ms_p50"]
            rest = stats["rest"]["wall_ms_p50"]
            total = stats["total_ms_p50"]
            tok = stats["all"]["prompt_tokens_p50"]
            reused = stats["all"]["reused_tokens_p50"]
        else:
            first = stats["wall_ms_p50"]
            rest = "-"
            total = "-"
            tok = stats["prompt_tokens_p50"]
            reused = stats["reused_tokens_p50"]
        lines.append(
            f"| {name} | {stats.get('decisions', stats.get('n', 1))} | {first} | "
            f"{rest} | {total} | {tok} | {reused} |"
        )
    lines += [
        "",
        "`evaluated tok` is the prompt tokens actually processed by the model;",
        "`reused tok` is what came from the prompt cache (0 for diverging prompts",
        "on this hybrid model).",
        "",
        "## Hungarian solve (CPU only)",
        "",
        "| matrix | mean ms | max ms |",
        "|---|---|---|",
    ]
    for size, stats in report["shapes"]["assignment_solve"].items():
        lines.append(f"| {size} | {stats['mean_ms']} | {stats['max_ms']} |")
    lines.append("")
    return "\n".join(lines)
