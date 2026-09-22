"""Command line interface for JEV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Config
from .engine import JevEngine


def _engine(args, cfg: Config | None = None) -> JevEngine:
    cfg = cfg or Config.load(getattr(args, "config", None))
    return JevEngine(cfg)


def _dump(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def _parse_cli_input(value: str) -> dict:
    """Parse 'image:PATH' / 'text:STRING', optional '|label' suffix."""
    label = None
    if "|" in value:
        value, label = value.rsplit("|", 1)
    kind, sep, payload = value.partition(":")
    if sep and kind in ("image", "text"):
        if kind == "image":
            return {"type": "image", "image_path": payload, "label": label}
        return {"type": "text", "text": payload, "label": label}
    if Path(value).exists():
        return {"type": "image", "image_path": value, "label": label}
    return {"type": "text", "text": value, "label": label}


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None, help="path to config.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="jev",
        description="Semantic-if decisions from one multimodal forward pass.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("serve", help="start the HTTP API (and llama-server)")
    _add_config(p)
    p.add_argument("--api-host", default=None)
    p.add_argument("--api-port", type=int, default=None)

    p = sub.add_parser("decide", help="run one decision")
    _add_config(p)
    p.add_argument("--question", required=True)
    p.add_argument("--option", action="append", required=True, help="repeatable")
    p.add_argument("--state", default=None)
    p.add_argument(
        "--image", action="append", default=None, help="repeatable path or base64"
    )
    p.add_argument("--thinking", default=None)
    p.add_argument("--compact", action="store_true")

    p = sub.add_parser(
        "index", help="assign items to ordered mixed image/text inputs"
    )
    _add_config(p)
    p.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "image:PATH or text:STRING, optional '|label' suffix to name the "
            "input for the model; repeat in order"
        ),
    )
    p.add_argument("--item", action="append", required=True, help="repeatable")
    p.add_argument(
        "--prompt",
        required=True,
        help=(
            "whole indexing instruction (rules + question) containing {item}, "
            "e.g. 'Which input shows a {item}?'"
        ),
    )
    p.add_argument(
        "--assignment", choices=["hungarian", "argmax"], default="hungarian"
    )
    p.add_argument("--objective", choices=["logprob", "prob"], default="logprob")
    p.add_argument("--thinking", default=None)

    p = sub.add_parser("diagnose", help="probe reasoning-skipping strategies")
    _add_config(p)
    p.add_argument(
        "--strategies",
        default=None,
        help="comma-separated; default all configured strategies plus 'on'",
    )

    p = sub.add_parser("maze", help="render a maze and solve it with JEV")
    _add_config(p)
    p.add_argument("--width", type=int, default=5)
    p.add_argument("--height", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--cell-px", type=int, default=96)
    p.add_argument("--thinking", default=None)
    p.add_argument(
        "--all-directions",
        action="store_true",
        help="offer all four moves instead of only the open ones",
    )

    p = sub.add_parser("assets", help="fetch animal images from Wikimedia Commons")
    p.add_argument("--dir", default=None)
    p.add_argument("--label", action="append", default=None)

    p = sub.add_parser("evaluate", help="run evaluation suites and write a report")
    _add_config(p)
    p.add_argument(
        "--suites",
        default="decisions,animals,indexing,maze,latency",
        help="comma-separated subset, or 'all'",
    )
    p.add_argument("--thinking", default=None)
    p.add_argument("--out", default=None)
    p.add_argument("--mazes", type=int, default=3)
    p.add_argument("--latency-targets", default="32,128,512,1024")
    p.add_argument("--thinking-bench", action="store_true")

    p = sub.add_parser("thinking-bench", help="compare reasoning-skipping strategies")
    _add_config(p)
    p.add_argument("--animals", dest="animals", action="store_true", default=True)
    p.add_argument("--no-animals", dest="animals", action="store_false")
    p.add_argument("--mazes", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("bench", help="latency/memory sweep vs prompt length")
    _add_config(p)
    p.add_argument("--targets", default="32,128,512,1024")
    p.add_argument("--repeats", type=int, default=1)

    p = sub.add_parser("config", help="print the resolved configuration")
    _add_config(p)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "config":
        cfg = Config.load(args.config)
        _dump(cfg.__dict__)
        return 0

    if args.command == "serve":
        import signal

        import uvicorn

        from .api import create_app

        # tmux/terminal teardown sends SIGHUP to the whole group; ignoring it
        # here lets uvicorn finish its lifespan shutdown (which stops the
        # llama-server child) instead of dying immediately.
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, signal.SIG_IGN)

        cfg = Config.load(args.config)
        if args.api_host:
            cfg.api.host = args.api_host
        if args.api_port:
            cfg.api.port = args.api_port
        print(f"JEV API on http://{cfg.api.host}:{cfg.api.port}/docs")
        uvicorn.run(create_app(cfg), host=cfg.api.host, port=cfg.api.port)
        return 0

    if args.command == "assets":
        from .tasks.animals import fetch_animal_images

        saved = fetch_animal_images(
            Path(args.dir) if args.dir else None, labels=args.label
        )
        print(f"[ok] {len(saved)} animal images ready")
        return 0

    if args.command == "maze":
        from .tasks.maze import solve_maze

        with _engine(args) as engine:
            outcome = solve_maze(
                engine,
                {
                    "width": args.width,
                    "height": args.height,
                    "seed": args.seed,
                    "max_steps": args.max_steps,
                    "cell_px": args.cell_px,
                    "thinking": args.thinking,
                    "save_images": True,
                    "legal_moves_only": not args.all_directions,
                },
            )
            _dump(
                {
                    "success": outcome["success"],
                    "steps": outcome["steps"],
                    "optimal_steps": outcome["optimal_steps"],
                    "step_optimality": outcome["step_optimality"],
                    "final_pos": outcome["final_pos"],
                    "images": outcome["images"],
                }
            )
        return 0

    with _engine(args) as engine:
        if args.command == "decide":
            result = engine.decide(
                question=args.question,
                options=args.option,
                state=args.state,
                images=args.image,
                thinking=args.thinking,
            )
            _dump(
                {
                    "chosen": result.chosen,
                    "confidence": result.confidence,
                    "strategy": result.strategy,
                    "probabilities": {
                        o.id: round(o.probability, 4) for o in result.options
                    },
                    "prompt_tokens": result.tokens_evaluated,
                    "tokens_cached": result.cached_tokens,
                    "prompt_ms": result.timings.get("prompt_ms"),
                    "wall_ms": result.timings.get("wall_ms"),
                }
            )
            return 0

        if args.command == "index":
            records = engine.index_inputs(
                inputs=[_parse_cli_input(value) for value in args.input],
                items=args.item,
                prompt=args.prompt,
                assignment=args.assignment,
                objective=args.objective,
                thinking=args.thinking,
            )
            print("\n".join(f"{r['item']}:{r['index']}" for r in records))
            _dump(
                {
                    r["item"]: {
                        "index": r["index"],
                        "argmax_index": r["argmax_index"],
                        "confidence": round(r["confidence"], 4),
                        "cached_tokens": r["cached_tokens"],
                    }
                    for r in records
                }
            )
            return 0

        if args.command == "diagnose":
            strategies = args.strategies.split(",") if args.strategies else None
            _dump(engine.diagnose(strategies))
            return 0

        if args.command == "evaluate":
            from .tasks.evaluate import evaluate_all

            suites = (
                ["decisions", "animals", "indexing", "maze", "latency"]
                if args.suites == "all"
                else [s.strip() for s in args.suites.split(",") if s.strip()]
            )
            report = evaluate_all(
                engine,
                suites=suites,
                thinking=args.thinking,
                out_dir=Path(args.out) if args.out else None,
                n_mazes=args.mazes,
                latency_targets=[int(x) for x in args.latency_targets.split(",")],
                include_thinking_bench=args.thinking_bench,
            )
            _dump(
                {
                    key: {
                        k: v
                        for k, v in value.items()
                        if k not in ("results", "mazes", "details", "rows")
                    }
                    for key, value in report["suites"].items()
                }
            )
            return 0

        if args.command == "thinking-bench":
            from .tasks.evaluate import thinking_benchmark

            report = thinking_benchmark(
                engine,
                include_animals=args.animals,
                out_dir=Path(args.out) if args.out else None,
                n_mazes=args.mazes,
            )
            _dump(
                {
                    "recommended": report["recommended"],
                    "current": report["current"],
                    "strategies": [
                        {
                            k: v
                            for k, v in row.items()
                            if k in (
                                "strategy",
                                "decisions_accuracy",
                                "animals_accuracy",
                                "maze_success_rate",
                                "decisions_latency_p50_ms",
                            )
                        }
                        for row in report["strategies"]
                    ],
                }
            )
            return 0

        if args.command == "bench":
            from .tasks.evaluate import latency_sweep

            _dump(
                latency_sweep(
                    engine,
                    [int(x) for x in args.targets.split(",")],
                    repeats=args.repeats,
                )
            )
            return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
