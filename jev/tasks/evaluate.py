"""Evaluation suites and reports for JEV.

All suites are decision-shaped, so they all benefit from the same readout:
  * decisions: authored labeled text records (accuracy)
  * animals:   Wikimedia photos classified by the model (accuracy)
  * maze:      step-by-step navigation from rendered images (success + optimality)
  * latency:   prefill time vs prompt length (and peak memory)
  * thinking:  head-to-head comparison of reasoning-skipping strategies
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..artifacts import TaskArtifacts, contact_sheet
from ..config import PROJECT_ROOT
from ..metrics import PeakSampler
from . import animals as animals_task
from . import decisions as decisions_task
from . import indexing as indexing_task
from .maze import solve_maze

log = logging.getLogger("jev.evaluate")

RESULTS_DIR = PROJECT_ROOT / "results"


def _latency_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    arr = np.asarray(values, dtype=float)
    return {
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "mean_ms": round(float(arr.mean()), 2),
        "max_ms": round(float(arr.max()), 2),
    }


# ----------------------------------------------------------------------
def run_decisions(
    engine,
    thinking: str | None = None,
    artifacts_key: str = "decisions",
) -> dict:
    artifacts = TaskArtifacts(artifacts_key)
    rows = []
    for record in decisions_task.DECISIONS:
        result = engine.decide(
            question=record["question"],
            options=record["options"],
            state=record["state"],
            thinking=thinking,
            decision_id=record["id"],
        )
        rows.append(
            {
                "id": record["id"],
                "state": record["state"],
                "question": record["question"],
                "options": record["options"],
                "expected": record["expected"],
                "chosen": result.chosen,
                "correct": result.chosen == record["expected"],
                "confidence": result.confidence,
                "probabilities": {o.id: o.probability for o in result.options},
                "strategy": result.strategy,
                "sampled_letter_match": result.sampled_letter_match,
                "prompt_sha256": result.prompt_sha256,
                "prompt": result.prompt,
                "prompt_ms": result.timings.get("prompt_ms", 0.0),
                "wall_ms": result.timings.get("wall_ms", 0.0),
                "prompt_tokens": result.tokens_evaluated,
            }
        )
    correct = sum(1 for r in rows if r["correct"])
    report = {
        "n": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "latency": _latency_summary([r["wall_ms"] for r in rows]),
        "results": rows,
    }
    artifacts.write_jsonl("decisions.jsonl", rows)
    artifacts.write_text(
        "misses.md",
        "\n".join(
            f"- {r['id']}: expected {r['expected']}, got {r['chosen']} "
            f"({r['confidence']:.3f})"
            for r in rows
            if not r["correct"]
        )
        or "no misses",
    )
    artifacts.summary(accuracy=report["accuracy"], n=report["n"])
    report["artifacts"] = str(artifacts.dir)
    return report


def run_animal_suite(
    engine,
    images_dir: Path | None = None,
    option_count: int = 4,
    thinking: str | None = None,
    limit: int | None = None,
    artifacts_key: str = "animals",
) -> dict:
    report = animals_task.run_animals(
        engine,
        images_dir=images_dir,
        option_count=option_count,
        thinking=thinking,
        limit=limit,
        artifacts_key=artifacts_key,
    )
    report["latency"] = _latency_summary(
        [r["wall_ms"] for r in report.get("results", [])]
    )
    return report


def run_indexing_suite(
    engine,
    thinking: str | None = None,
    trials: int = 3,
    images_per_trial: int = 6,
    artifacts_key: str = "indexing",
) -> dict:
    return indexing_task.run_indexing(
        engine,
        trials=trials,
        images_per_trial=images_per_trial,
        thinking=thinking,
        artifacts_key=artifacts_key,
    )


def run_maze_suite(
    engine,
    thinking: str | None = None,
    seeds: list[int] | None = None,
    width: int = 5,
    height: int = 5,
    max_steps: int = 60,
    legal_moves_only: bool = True,
    artifacts_key: str = "maze",
) -> dict:
    seeds = seeds or [1, 2, 3]
    artifacts = TaskArtifacts(artifacts_key)
    mazes = []
    for seed in seeds:
        outcome = solve_maze(
            engine,
            {
                "width": width,
                "height": height,
                "seed": seed,
                "max_steps": max_steps,
                "thinking": thinking,
                "save_images": True,
                "legal_moves_only": legal_moves_only,
            },
            save_dir=artifacts.dir,
        )
        mazes.append(outcome)

    # One contact sheet per seed: first / middle / final frame plus trajectory.
    for outcome in mazes:
        seed = outcome["seed"]
        seed_dir = Path(outcome["images"])
        step_files = sorted(seed_dir.glob("step-*.png"))
        picks = [step_files[0]] if step_files else []
        if len(step_files) > 2:
            picks.append(step_files[len(step_files) // 2])
        if step_files:
            picks.append(step_files[-1])
        entries = []
        for path in picks:
            step_no = path.stem.split("-")[1]
            entries.append(
                {
                    "image": path,
                    "title": f"seed {seed} step {int(step_no)}",
                    "subtitle": "trajectory frame",
                    "ok": True,
                }
            )
        final = seed_dir / "final.png"
        if final.exists():
            entries.append(
                {
                    "image": final,
                    "title": f"seed {seed} final",
                    "subtitle": (
                        "reached the exit"
                        if outcome["success"]
                        else f"stopped at {tuple(outcome['final_pos'])}"
                    ),
                    "ok": outcome["success"],
                }
            )
        if entries:
            artifacts.save_image(
                f"seed-{seed}-contact-sheet.png", contact_sheet(entries, columns=4)
            )
    artifacts.summary(
        success_rate=sum(1 for m in mazes if m["success"]) / len(mazes)
        if mazes
        else 0.0,
        seeds=[m["seed"] for m in mazes],
    )
    success = sum(1 for m in mazes if m["success"])
    return {
        "n": len(mazes),
        "success_rate": success / len(mazes) if mazes else 0.0,
        "mean_step_optimality": float(
            np.mean([m["step_optimality"] for m in mazes])
        )
        if mazes
        else 0.0,
        "total_invalid_moves": sum(m["invalid_moves"] for m in mazes),
        "mazes": [
            {
                "seed": m["seed"],
                "success": m["success"],
                "steps": m["steps"],
                "optimal_steps": m["optimal_steps"],
                "invalid_moves": m["invalid_moves"],
                "step_optimality": m["step_optimality"],
                "images": m["images"],
            }
            for m in mazes
        ],
        "latency": _latency_summary(
            [
                step["prompt_ms"]
                for m in mazes
                for step in m["decisions"]
            ]
        ),
        "details": mazes,
        "artifacts": str(artifacts.dir),
    }


def latency_sweep(engine, token_targets: list[int], repeats: int = 1) -> dict:
    """Prefill latency and peak memory vs prompt length (the JEV-CPU §7 curve)."""
    pid = engine.server.proc.pid if engine.server and engine.server.proc else None
    artifacts = TaskArtifacts("latency")
    rows = []
    for target in token_targets:
        state = _state_for_tokens(target)
        for _ in range(repeats):
            with PeakSampler(pid) as sampler:
                result = engine.decide(
                    question="Is the text positive, neutral or negative?",
                    options=[
                        {"id": "positive", "description": "positive"},
                        {"id": "negative", "description": "negative"},
                    ],
                    state=state,
                    cache_prompt=False,
                )
                memory = sampler.stop()
            rows.append(
                {
                    "target_tokens": target,
                    "prompt_tokens": result.tokens_evaluated - 1,
                    "prompt_ms": result.timings.get("prompt_ms", 0.0),
                    "wall_ms": result.timings.get("wall_ms", 0.0),
                    "prefill_tokens_per_s": result.timings.get("prompt_per_second", 0.0),
                    "server_pid": pid,
                    **memory,
                }
            )
            print(
                f"  tokens={rows[-1]['prompt_tokens']:>5} "
                f"prefill={rows[-1]['prompt_ms']:>8.1f} ms "
                f"rss={memory['peak_server_rss_mib']:.0f} MiB"
            )
    csv_lines = [
        "target_tokens,prompt_tokens,prompt_ms,prefill_tokens_per_s,"
        "peak_server_rss_mib,peak_gpu_memory_mib"
    ]
    for row in rows:
        gpu = row.get("peak_gpu_memory_mib") or {}
        csv_lines.append(
            f"{row['target_tokens']},{row['prompt_tokens']},{row['prompt_ms']},"
            f"{row.get('prefill_tokens_per_s', 0)},{row['peak_server_rss_mib']},"
            f"\"{gpu}\""
        )
    artifacts.write_text("sweep.csv", "\n".join(csv_lines) + "\n")
    artifacts.write_json("sweep.json", rows)
    artifacts.summary(rows=len(rows))
    return {"rows": rows, "artifacts": str(artifacts.dir)}


def _state_for_tokens(target: int) -> str:
    sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
    text = (sentence * (target // 8 + 2))[: target * 5]
    return text


# ----------------------------------------------------------------------
def thinking_benchmark(
    engine,
    include_animals: bool = True,
    images_dir: Path | None = None,
    out_dir: Path | None = None,
    n_mazes: int = 0,
) -> dict:
    """Run every reasoning-skipping strategy on the same suites and rank them."""
    strategies = ["auto"] + sorted(engine.cfg.thinking.strategies)
    artifacts = TaskArtifacts("thinking_bench")
    rows = []
    for name in strategies:
        print(f"[thinking] strategy={name}")
        started = time.perf_counter()
        override = None if name == "auto" else name
        key_base = f"thinking_bench/{name}"
        try:
            probe = None if name == "auto" else engine.probe_strategy(name).to_dict()
            decision_report = run_decisions(
                engine, thinking=override, artifacts_key=f"{key_base}/decisions"
            )
            animal_report = None
            if include_animals and animals_task.list_animal_images(images_dir):
                animal_report = run_animal_suite(
                    engine,
                    images_dir=images_dir,
                    thinking=override,
                    artifacts_key=f"{key_base}/animals",
                )
            maze_report = None
            if n_mazes:
                maze_report = run_maze_suite(
                    engine,
                    thinking=override,
                    seeds=list(range(1, n_mazes + 1)),
                    artifacts_key=f"{key_base}/maze",
                )
        except Exception as exc:  # noqa: BLE001 - one bad strategy must not abort
            rows.append(
                {
                    "strategy": name,
                    "description": "FAILED",
                    "error": str(exc),
                    "probe": None,
                    "decisions_accuracy": 0.0,
                    "decisions_latency_p50_ms": 0.0,
                    "animals_accuracy": None,
                    "maze_success_rate": None,
                    "maze_step_optimality": None,
                    "elapsed_s": round(time.perf_counter() - started, 2),
                }
            )
            print(f"  ERROR: {exc}")
            continue
        rows.append(
            {
                "strategy": name,
                "description": (
                    engine.cfg.thinking.get(name).description
                    if name != "auto"
                    else "auto-selected strategy"
                ),
                "resolved_same_as": engine.strategy if name == "auto" else None,
                "probe": probe,
                "decisions_accuracy": decision_report["accuracy"],
                "decisions_latency_p50_ms": decision_report["latency"].get("p50_ms", 0.0),
                "animals_accuracy": animal_report["accuracy"] if animal_report else None,
                "maze_success_rate": maze_report["success_rate"] if maze_report else None,
                "maze_step_optimality": (
                    maze_report["mean_step_optimality"] if maze_report else None
                ),
                "elapsed_s": round(time.perf_counter() - started, 2),
            }
        )
        print(
            f"  decisions={rows[-1]['decisions_accuracy']:.3f}"
            + (
                f" animals={rows[-1]['animals_accuracy']:.3f}"
                if rows[-1]["animals_accuracy"] is not None
                else ""
            )
        )

    def rank_key(row):
        return (
            row["decisions_accuracy"],
            row["animals_accuracy"] if row["animals_accuracy"] is not None else 0.0,
            -(row["decisions_latency_p50_ms"] or 0.0),
        )

    best = max(rows, key=rank_key) if rows else None
    report = {
        "strategies": rows,
        "recommended": best["strategy"] if best else None,
        "current": engine.strategy,
        "artifacts": str(artifacts.dir),
    }
    artifacts.write_json("thinking-bench.json", report)
    artifacts.write_text("thinking-bench.md", _thinking_markdown(report))
    artifacts.summary(recommended=report["recommended"], current=report["current"])
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "thinking-bench.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False)
        )
        (out_dir / "thinking-bench.md").write_text(_thinking_markdown(report))
    return report


def _thinking_markdown(report: dict) -> str:
    lines = [
        "# Reasoning-skipping strategy benchmark",
        "",
        f"Current resolved strategy: `{report['current']}`",
        f"Recommended by accuracy: `{report['recommended']}`",
        "",
        "| strategy | boundary ok | think detected | letter top-1 | option mass | decisions acc | animals acc | p50 ms |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in report["strategies"]:
        probe = row.get("probe") or {}
        if row.get("error"):
            lines.append(f"| {row['strategy']} | FAILED: {row['error'][:60]} | | | | | | |")
            continue
        lines.append(
            "| {s} | {b} | {t} | {l} | {m} | {d:.3f} | {a} | {p:.1f} |".format(
                s=row["strategy"],
                b=probe.get("boundary_ok", "-"),
                t=probe.get("thinking_detected", "-"),
                l=probe.get("letter_top1", "-"),
                m=probe.get("letter_mass", "-"),
                d=row["decisions_accuracy"],
                a=(
                    f"{row['animals_accuracy']:.3f}"
                    if row["animals_accuracy"] is not None
                    else "-"
                ),
                p=row["decisions_latency_p50_ms"] or 0.0,
            )
        )
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
def evaluate_all(
    engine,
    suites: list[str] | None = None,
    thinking: str | None = None,
    out_dir: Path | None = None,
    n_mazes: int = 3,
    latency_targets: list[int] | None = None,
    include_thinking_bench: bool = False,
) -> dict:
    suites = suites or [
        "decisions",
        "animals",
        "indexing",
        "maze",
        "latency",
    ]
    out_dir = out_dir or RESULTS_DIR / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": engine.info(),
        "thinking_override": thinking,
        "suites": {},
    }

    if "decisions" in suites:
        print("[eval] authored decisions")
        report["suites"]["decisions"] = run_decisions(engine, thinking=thinking)
    if "animals" in suites:
        print("[eval] animal classification")
        report["suites"]["animals"] = run_animal_suite(engine, thinking=thinking)
    if "indexing" in suites:
        print("[eval] multi-image indexing")
        report["suites"]["indexing"] = run_indexing_suite(engine, thinking=thinking)
    if "maze" in suites:
        print("[eval] maze navigation")
        report["suites"]["maze"] = run_maze_suite(
            engine, thinking=thinking, seeds=list(range(1, n_mazes + 1))
        )
    if "latency" in suites:
        print("[eval] latency + memory sweep")
        report["suites"]["latency"] = latency_sweep(
            engine, latency_targets or [32, 128, 512, 1024], repeats=1
        )
    if include_thinking_bench:
        print("[eval] reasoning-skipping strategy benchmark")
        report["thinking_benchmark"] = thinking_benchmark(
            engine,
            include_animals="animals" in suites,
            out_dir=out_dir,
            n_mazes=1 if "maze" in suites else 0,
        )

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str)
    )
    (out_dir / "report.md").write_text(_report_markdown(report))
    report["output_dir"] = str(out_dir)
    print(f"[eval] report written to {out_dir}/report.md")
    return report


def _report_markdown(report: dict) -> str:
    model = report["model"]
    lines = [
        "# JEV evaluation report",
        "",
        f"- created: `{report['created']}`",
        f"- model: `{model['model']}`",
        f"- device: `{model['device']}`",
        f"- thinking strategy: `{model['thinking_strategy']}` (mode `{model['thinking_mode']}`)",
        f"- ctx: `{model['ctx_size']}` | slots: `{model['parallel_slots']}`",
        "",
    ]
    decisions = report["suites"].get("decisions")
    if decisions:
        lines += [
            "## Authored decisions",
            "",
            f"- accuracy: **{decisions['accuracy']:.3f}** over {decisions['n']} rows",
            f"- latency: {decisions['latency'].get('p50_ms', 0):.1f} ms p50, "
            f"{decisions['latency'].get('p95_ms', 0):.1f} ms p95",
            "",
        ]
    animal = report["suites"].get("animals")
    if animal:
        lines += [
            "## Animal classification",
            "",
            f"- accuracy: **{animal['accuracy']:.3f}** over {animal.get('n', 0)} images",
            "",
        ]
    indexing = report["suites"].get("indexing")
    if indexing:
        lines += [
            "## Mixed-input indexing",
            "",
            f"- item accuracy: **{indexing['item_accuracy']:.3f}** over "
            f"{indexing.get('n', 0)} items",
            f"- argmax accuracy: {indexing.get('argmax_item_accuracy', 0):.3f} "
            f"(assignment changed {indexing.get('items_changed_by_assignment', 0)})",
            f"- perfect trials: {indexing.get('perfect_trials', 0)}/"
            f"{len(indexing.get('trials', []))}",
            "",
        ]
        for trial in indexing.get("trials", []):
            lines.append(
                f"  - trial {trial['trial']} [{trial['kind']}]: "
                f"{trial['correct']}/{trial['n']} perfect={trial['perfect']}"
            )
            lines.append(f"    expected {trial['expected']}")
            lines.append(f"    got      {trial['assignments']}")
        lines.append("")
    maze = report["suites"].get("maze")
    if maze:
        lines += [
            "## Maze navigation",
            "",
            f"- success rate: **{maze['success_rate']:.3f}** over {maze['n']} mazes",
            f"- mean step optimality: {maze['mean_step_optimality']:.3f}",
            "",
        ]
        for item in maze["mazes"]:
            lines.append(
                f"  - seed {item['seed']}: success={item['success']} "
                f"steps={item['steps']}/{item['optimal_steps']} "
                f"optimality={item['step_optimality']:.2f}"
            )
        lines.append("")
    latency = report["suites"].get("latency")
    if latency:
        lines += [
            "## Latency and memory",
            "",
            "| prompt tokens | prefill ms | tok/s | peak server RSS MiB | peak GPU MiB |",
            "|---|---|---|---|---|",
        ]
        for row in latency["rows"]:
            gpu = row.get("peak_gpu_memory_mib") or {}
            gpu_txt = ", ".join(f"{k}:{v}" for k, v in gpu.items()) or "-"
            lines.append(
                f"| {row['prompt_tokens']} | {row['prompt_ms']:.1f} | "
                f"{row.get('prefill_tokens_per_s', 0):.1f} | "
                f"{row['peak_server_rss_mib']:.0f} | {gpu_txt} |"
            )
        lines.append("")
    bench = report.get("thinking_benchmark")
    if bench:
        lines.append(_thinking_markdown(bench))
    artifacts = {
        name: suite.get("artifacts")
        for name, suite in report["suites"].items()
        if isinstance(suite, dict) and suite.get("artifacts")
    }
    if bench and bench.get("artifacts"):
        artifacts["thinking_bench"] = bench["artifacts"]
    if artifacts:
        lines += ["## Artifacts", ""]
        for name, path in artifacts.items():
            lines.append(f"- `{name}`: `{path}`")
        lines.append("")
    return "\n".join(lines)
