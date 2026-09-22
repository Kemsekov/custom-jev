"""FastAPI surface for JEV.

Design: the API is the requested reference model for testing, not a separate
layer of logic. Every endpoint forwards to the engine and returns typed
option probabilities plus timing and audit fields (prompt hash, tokens).
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from .config import Config
from .engine import JevEngine, JevError, ReadoutError, ValidationError
from .schemas import (
    BatchDecideRequest,
    DecideRequest,
    DiagnoseRequest,
    IndexRequest,
    LatencyBenchRequest,
    MazeRequest,
)

log = logging.getLogger("jev.api")


def create_app(cfg: Config | None = None, engine: JevEngine | None = None) -> FastAPI:
    cfg = cfg or Config.load()
    owns_engine = engine is None

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.engine = engine or JevEngine(cfg)
        app.state.engine.start()
        yield
        app.state.engine.stop()

    app = FastAPI(
        title="JEV API",
        version="0.1.0",
        description=(
            "Semantic-if decisions on a multimodal GGUF model: one forward pass, "
            "no generated text. Feed text and/or an image, get typed option "
            "probabilities and the chosen option."
        ),
        lifespan=lifespan,
    )
    app.state.cfg = cfg
    app.state.owns_engine = owns_engine

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health", tags=["system"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/info", tags=["system"])
    def info() -> dict:
        return app.state.engine.info()

    @app.post("/v1/decide", tags=["decisions"])
    def decide(req: DecideRequest) -> dict:
        try:
            result = app.state.engine.decide(
                question=req.question,
                options=[o.model_dump() for o in req.options],
                state=req.state,
                image=_resolve_image(req),
                images=_resolve_images(req),
                thinking=req.thinking,
                decision_id=req.id,
            )
        except (ValidationError, ReadoutError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JevError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return result.to_dict()

    @app.post("/v1/decide/batch", tags=["decisions"])
    def decide_batch(req: BatchDecideRequest) -> dict:
        try:
            results = app.state.engine.decide_many(
                questions=[i.question for i in req.items],
                options=[[o.model_dump() for o in i.options] for i in req.items],
                state=req.state,
                image=_resolve_image(req),
                images=_resolve_images(req),
                thinking=req.thinking,
                assignment=req.assignment,
                objective=req.objective,
                parallel=req.parallel,
            )
        except (ValidationError, ReadoutError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JevError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        payload = [r.to_dict() for r in results]
        response = {
            "count": len(payload),
            "results": payload,
            "prompt_tokens_per_decision": [r.tokens_evaluated for r in results],
            "cached_tokens_per_decision": [r.cached_tokens for r in results],
        }
        if req.assignment != "none" and results:
            response["assignment"] = {
                "method": results[0].assignment_method,
                "chosen": {r.id: r.chosen for r in results},
                "argmax": {r.id: r.argmax_choice for r in results},
                "changed": [r.id for r in results if r.assignment_changed],
            }
        return response

    @app.post("/v1/index", tags=["tasks"])
    def index_inputs(req: IndexRequest) -> dict:
        inputs = []
        for spec in req.inputs:
            data = spec.model_dump()
            if data["type"] is None:
                data["type"] = (
                    "image"
                    if (data["image_base64"] or data["image_path"])
                    else "text"
                )
            inputs.append(data)
        try:
            records = app.state.engine.index_inputs(
                inputs=inputs,
                items=req.items,
                prompt=req.prompt,
                assignment=req.assignment,
                objective=req.objective,
                thinking=req.thinking,
                parallel=req.parallel,
            )
        except (ValidationError, ReadoutError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JevError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return {
            "assignments": {r["item"]: r["index"] for r in records},
            "argmax_assignments": {r["item"]: r["argmax_index"] for r in records},
            "assignment_method": records[0]["assignment_method"] if records else None,
            "items_changed_by_assignment": sum(
                1 for r in records if r["index_changed"]
            ),
            "text": "\n".join(f"{r['item']}:{r['index']}" for r in records),
            "input_count": len(inputs),
            "probability_matrix": {
                r["item"]: {k: round(v, 5) for k, v in r["probabilities"].items()}
                for r in records
            },
            "decisions": [
                {
                    "item": r["item"],
                    "index": r["index"],
                    "argmax_index": r["argmax_index"],
                    "index_changed": r["index_changed"],
                    "confidence": r["confidence"],
                    "prompt_tokens": r["prompt_tokens"],
                    "cached_tokens": r["cached_tokens"],
                    "reused_tokens": r.get("reused_tokens", 0),
                    "readout": r.get("readout", "auto"),
                    "prompt_ms": round(r.get("prompt_ms", 0.0), 2),
                    "wall_ms": round(r.get("wall_ms", 0.0), 2),
                }
                for r in records
            ],
        }

    @app.post("/v1/diagnose", tags=["diagnostics"])
    def diagnose(req: DiagnoseRequest) -> dict:
        try:
            return app.state.engine.diagnose(req.strategies)
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except JevError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/maze/solve", tags=["tasks"])
    def maze_solve(req: MazeRequest) -> dict:
        from .tasks.maze import solve_maze

        try:
            return solve_maze(
                app.state.engine, req.model_dump(), cfg=app.state.cfg
            )
        except JevError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/v1/bench/latency", tags=["diagnostics"])
    def bench_latency(req: LatencyBenchRequest) -> dict:
        from .tasks.evaluate import latency_sweep

        try:
            return latency_sweep(app.state.engine, req.token_targets, req.repeats)
        except JevError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


def _resolve_image(req: Any) -> Any:
    if getattr(req, "image_base64", None):
        return req.image_base64
    if getattr(req, "image_path", None):
        return req.image_path
    return None


def _resolve_images(req: Any, required: bool = False) -> list[Any] | None:
    images: list[Any] = []
    if getattr(req, "images_base64", None):
        images.extend(req.images_base64)
    if getattr(req, "images_paths", None):
        images.extend(req.images_paths)
    if required and not images:
        raise HTTPException(
            status_code=422,
            detail="provide images_base64 or images_paths (at least one image)",
        )
    return images or None


app = None


def main() -> None:  # pragma: no cover - convenience entry point
    import uvicorn

    cfg = Config.load()
    uvicorn.run(create_app(cfg), host=cfg.api.host, port=cfg.api.port)
