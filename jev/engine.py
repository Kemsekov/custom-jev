"""JEV engine: semantic-if decisions read from one forward pass.

The readout path:
  1. render the chat template (with a reasoning-skipping strategy),
  2. pin every option letter to exactly one vocabulary token and verify the
     answer boundary,
  3. run ONE completion with `n_predict=1`, a GBNF grammar over the option
     letters and `post_sampling_probs=true`,
  4. softmax-normalize the returned option probabilities and take the argmax.

No text is generated, no sampling loop runs, no KV cache grows.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .assignment import argmax_assignment, optimal_assignment
from .client import LlamaClient, LlamaClientError
from .config import PROJECT_ROOT, Config, StrategyConfig
from .images import encode_image, encode_images
from .llama_server import LlamaServer
from .prompting import (
    LETTERS,
    ProbeResult,
    build_grammar,
    build_messages,
    looks_like_thinking,
    render_prompt,
    template_supports_thinking,
)

log = logging.getLogger("jev.engine")


class JevError(RuntimeError):
    pass


class ValidationError(JevError):
    pass


class ReadoutError(JevError):
    pass


@dataclass
class OptionScore:
    id: str
    letter: str
    description: str
    token_id: int | None
    probability: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DecisionResult:
    id: str | None
    question: str
    options: list[OptionScore]
    chosen: str | None
    chosen_letter: str | None
    confidence: float
    strategy: str
    strategy_description: str
    injected: bool
    grammar: str
    prompt: str
    prompt_sha256: str
    sampled_token_id: int
    sampled_letter_match: bool
    has_image: bool
    image_count: int = 0
    argmax_choice: str | None = None
    assignment_method: str | None = None
    assignment_changed: bool = False
    reused_tokens: int = 0
    readout: str = "auto"
    timings: dict[str, float] = field(default_factory=dict)
    tokens_evaluated: int = 0
    cached_tokens: int = 0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["probabilities"] = {
            o["id"]: o["probability"] for o in data["options"]
        }
        return data


@dataclass
class PinningReport:
    option_count: int
    slot_token_ids: list[int]
    slot_texts: list[str]
    boundary_ok: bool
    exact_roundtrip: bool
    detail: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class JevEngine:
    def __init__(
        self,
        cfg: Config | None = None,
        client: LlamaClient | None = None,
        server: LlamaServer | None = None,
    ):
        self.cfg = cfg or Config.load()
        self._own_server = server is None and client is None
        self.server = server if server is not None else (
            LlamaServer(self.cfg) if self._own_server else None
        )
        self.client = client
        self._started = False
        self._supports_thinking = False
        self._chat_template = ""
        self._model_props: dict = {}
        self._strategy: str | None = None
        self._strategy_source: str = "none"
        self._probe_results: list[ProbeResult] = []
        self._letter_tokens: dict[str, int] = {}
        self._pin_cache: dict[str, PinningReport] = {}

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def start(self) -> "JevEngine":
        if self._started:
            return self
        if self._own_server and self.server is not None:
            self.server.start()
        if self.client is None:
            self.client = LlamaClient(
                self.cfg.server.base_url, timeout=self.cfg.engine.request_timeout
            )
        self._load_model_info()
        self._resolve_strategy()
        self._started = True
        return self

    def stop(self) -> None:
        if self.client is not None:
            self.client.close()
        if self._own_server and self.server is not None:
            self.server.stop()
        self._started = False

    def __enter__(self) -> "JevEngine":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def _load_model_info(self) -> None:
        assert self.client is not None
        try:
            props = self.client.props()
        except LlamaClientError as exc:
            raise JevError(f"llama-server not reachable: {exc}") from exc
        self._model_props = props
        self._chat_template = props.get("chat_template") or ""
        self._supports_thinking = template_supports_thinking(self._chat_template)

    # ------------------------------------------------------------------
    # reasoning-skipping strategy selection
    # ------------------------------------------------------------------
    def _spec(self, name: str) -> StrategyConfig:
        if name == "on":
            return StrategyConfig(
                name="on",
                description="reasoning enabled (diagnostic only)",
                template_kwargs={"enable_thinking": True},
            )
        try:
            return self.cfg.thinking.get(name)
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

    def _resolve_strategy(self) -> None:
        mode = (self.cfg.thinking.mode or "auto").lower()
        if mode != "auto":
            spec = self._spec(mode)
            self._strategy = spec.name
            self._strategy_source = "configured"
            log.info("thinking strategy: %s (configured)", spec.name)
            return

        if self.cfg.thinking.cache and not self.cfg.thinking.refresh:
            cached = self._load_cached_strategy()
            if cached:
                self._strategy = cached
                self._strategy_source = "cached"
                log.info(
                    "thinking strategy: %s (cached for %s; no probe)",
                    cached,
                    self.cfg.model.gguf_path.name,
                )
                return

        order = [n for n in self.cfg.thinking.auto_order if n != "on"]
        results: list[ProbeResult] = []
        chosen: str | None = None
        for name in order:
            spec = self._spec(name)
            probe = self._probe(spec)
            # A strategy whose cue merges with the next letter is invalid:
            # the answer boundary must stay a single option-letter token.
            try:
                rendered = render_prompt(
                    self.client,
                    _probe_messages(4),
                    spec,
                    supports_thinking=self._supports_thinking,
                )
                self._pin(rendered.prompt, 4)
                probe.boundary_ok = True
            except ValidationError as exc:
                probe.boundary_ok = False
                probe.note = str(exc)
            results.append(probe)
            if probe.boundary_ok and probe.letter_top1 and not probe.thinking_detected:
                chosen = name
                break
        usable = [p for p in results if p.boundary_ok]
        if chosen is None and usable:
            chosen = max(
                usable, key=lambda r: (r.letter_mass, -r.thinking_detected)
            ).strategy
        self._probe_results = results
        self._strategy = chosen or "off"
        self._strategy_source = "probed"
        if not usable:
            log.error(
                "no strategy preserved the answer boundary; JEV readout will "
                "fail until the thinking strategies are fixed"
            )
        if self.cfg.thinking.cache:
            self._save_cached_strategy()
        log.info(
            "thinking strategy auto-selected: %s (probes: %s)",
            self._strategy,
            [
                f"{p.strategy}=top1:{p.letter_top1},think:{p.thinking_detected},"
                f"mass:{p.letter_mass:.3f}"
                for p in results
            ],
        )

    # ------------------------------------------------------------------
    # thinking-strategy cache (per model fingerprint)
    # ------------------------------------------------------------------
    def _thinking_cache_file(self) -> Path:
        path = Path(self.cfg.thinking.cache_path)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def _model_fingerprint(self) -> dict:
        try:
            size = self.cfg.model.gguf_path.stat().st_size
        except OSError:
            size = 0
        return {
            "model": self.cfg.model.gguf_path.name,
            "model_bytes": size,
            "chat_template_sha256": hashlib.sha256(
                self._chat_template.encode()
            ).hexdigest()[:16],
        }

    def _load_cached_strategy(self) -> str | None:
        path = self._thinking_cache_file()
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
        entry = data.get(self.cfg.model.gguf_path.name)
        if not isinstance(entry, dict):
            return None
        fingerprint = self._model_fingerprint()
        if (
            entry.get("model_bytes") != fingerprint["model_bytes"]
            or entry.get("chat_template_sha256") != fingerprint["chat_template_sha256"]
        ):
            log.warning(
                "cached thinking strategy for %s is stale (model or template "
                "changed); re-probing",
                fingerprint["model"],
            )
            return None
        strategy = entry.get("strategy")
        if strategy != "off" and strategy not in self.cfg.thinking.strategies:
            return None
        return strategy

    def _save_cached_strategy(self) -> None:
        path = self._thinking_cache_file()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            data = json.loads(path.read_text()) if path.exists() else {}
        except (OSError, json.JSONDecodeError):
            data = {}
        data[self.cfg.model.gguf_path.name] = {
            **self._model_fingerprint(),
            "strategy": self._strategy,
            "source": self._strategy_source,
            "probes": [p.to_dict() for p in self._probe_results],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        try:
            path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        except OSError as exc:  # pragma: no cover - cache is best effort
            log.warning("could not write thinking-strategy cache: %s", exc)
            return
        log.info(
            "cached thinking strategy %s for %s at %s",
            self._strategy,
            self.cfg.model.gguf_path.name,
            path,
        )

    @property
    def strategy(self) -> str:
        return self._strategy or "off"

    def _probe(self, spec: StrategyConfig, option_count: int = 4) -> ProbeResult:
        assert self.client is not None
        messages = _probe_messages(option_count)
        rendered = render_prompt(
            self.client, messages, spec, supports_thinking=self._supports_thinking
        )
        resp = self.client.completion(
            rendered.prompt, grammar=None, n_predict=1, n_probs=16, cache_prompt=False
        )
        top = _top_tokens(resp)
        slot_ids = {self._slot_token(LETTERS[i]) for i in range(option_count)}
        letter_mass = sum(float(t.get("prob", 0.0)) for t in top if int(t["id"]) in slot_ids)
        letter_top1 = bool(top) and int(top[0]["id"]) in slot_ids
        thinking = bool(top) and looks_like_thinking(
            top[0]["token"], self.cfg.thinking.think_markers
        )
        return ProbeResult(
            strategy=spec.name or "custom",
            thinking_detected=thinking,
            letter_top1=letter_top1,
            letter_mass=letter_mass,
            top_tokens=[
                {
                    "id": int(t["id"]),
                    "token": t.get("token", ""),
                    "prob": round(float(t.get("prob", 0.0)), 5),
                }
                for t in top[:8]
            ],
            inject=spec.inject,
        )

    def probe_strategy(self, name: str) -> ProbeResult:
        self.start()
        return self._probe(self._spec(name))

    def diagnose(self, strategies: Iterable[str] | None = None) -> dict:
        self.start()
        if strategies is None:
            names = list(self.cfg.thinking.auto_order) + ["on"]
        else:
            names = list(strategies)
        reports = []
        seen = set()
        for name in names:
            if name in seen:
                continue
            seen.add(name)
            reports.append(self._probe(self._spec(name)).to_dict())
        messages = _probe_messages(4)
        rendered = render_prompt(
            self.client,
            messages,
            self._spec(self.strategy),
            supports_thinking=self._supports_thinking,
        )
        pinning = self._pin(rendered.prompt, 4)
        return {
            "model": self.info(),
            "resolved_strategy": self.strategy,
            "template_supports_thinking": self._supports_thinking,
            "probes": reports,
            "pinning": pinning.to_dict(),
        }

    # ------------------------------------------------------------------
    # token pinning
    # ------------------------------------------------------------------
    def _slot_token(self, letter: str) -> int:
        assert self.client is not None
        if letter in self._letter_tokens:
            return self._letter_tokens[letter]
        tokens = self.client.tokenize(letter, parse_special=False)
        if len(tokens) != 1:
            raise ValidationError(
                f"answer slot {letter!r} is not a single token (got {tokens})"
            )
        token_id = int(tokens[0])
        decoded = self.client.detokenize([token_id])
        if decoded != letter:
            raise ValidationError(
                f"answer slot {letter!r} does not round-trip (decoded {decoded!r})"
            )
        self._letter_tokens[letter] = token_id
        return token_id

    def _pin(self, prompt: str, n_options: int) -> PinningReport:
        assert self.client is not None
        if not 2 <= n_options <= self.cfg.engine.max_options:
            raise ValidationError(
                f"option count must be 2..{self.cfg.engine.max_options}, got {n_options}"
            )
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._pin_cache.get(prompt_hash)
        if cached is not None:
            return cached

        slots = [self._slot_token(LETTERS[i]) for i in range(n_options)]
        if len(set(slots)) != len(slots):
            raise ValidationError("answer-slot tokens collide")
        base = self.client.tokenize(prompt, parse_special=True)
        for i, slot in enumerate(slots):
            extended = self.client.tokenize(prompt + LETTERS[i], parse_special=True)
            if extended != list(base) + [slot]:
                raise ValidationError(
                    f"answer boundary changes tokenization for slot {LETTERS[i]}; "
                    "refusing to run a corrupted JEV readout"
                )
        report = PinningReport(
            option_count=n_options,
            slot_token_ids=slots,
            slot_texts=[LETTERS[i] for i in range(n_options)],
            boundary_ok=True,
            exact_roundtrip=True,
        )
        self._pin_cache[prompt_hash] = report
        if len(self._pin_cache) > 512:
            self._pin_cache.pop(next(iter(self._pin_cache)))
        return report

    # ------------------------------------------------------------------
    # decisions
    # ------------------------------------------------------------------
    def _encode_sources(self, image: Any, images: list[Any] | None) -> list[str]:
        sources: list[Any] = []
        if image is not None:
            sources.append(image)
        if images:
            sources.extend(images)
        return [encode_image(src, self.cfg.image)[0] for src in sources]

    def decide(
        self,
        question: str,
        options: list[dict],
        state: Any = None,
        image: Any = None,
        images: list[Any] | None = None,
        thinking: str | None = None,
        decision_id: str | None = None,
        cache_prompt: bool = True,
        _data_urls: list[str] | None = None,
    ) -> DecisionResult:
        self.start()
        options = _normalize_options(options, self.cfg.engine.max_options)
        strategy_name = (thinking or "auto").lower()
        if strategy_name == "auto":
            strategy_name = self.strategy
        if strategy_name == "on":
            raise ValidationError(
                "thinking='on' cannot produce a JEV readout: the first token "
                "would be a reasoning marker. Use 'auto' or a skipping strategy."
            )
        spec = self._spec(strategy_name)

        data_urls = (
            list(_data_urls)
            if _data_urls is not None
            else self._encode_sources(image, images)
        )
        messages = build_messages(state, question, options, image_data_urls=data_urls)
        rendered = render_prompt(
            self.client, messages, spec, supports_thinking=self._supports_thinking
        )
        pinning = self._pin(rendered.prompt, len(options))
        grammar = build_grammar(len(options))
        readout_mode = (self.cfg.engine.readout or "auto").lower()
        if readout_mode not in ("auto", "grammar"):
            raise ValidationError(
                f"engine.readout must be 'auto' or 'grammar', got {readout_mode!r}"
            )
        # Without a grammar, the top-n probabilities already contain the option
        # slots whenever the model answers with a letter; renormalising them
        # yields the exact same conditional distribution the grammar would have
        # enforced. Grammar is then only a safety net for the rare case where a
        # slot is missing from the returned candidates.
        n_probs = max(self.cfg.engine.top_probs, len(options) + 8)

        t0 = time.perf_counter()
        try:
            resp = self.client.completion(
                rendered.prompt,
                grammar=grammar if readout_mode == "grammar" else None,
                multimodal_data=data_urls or None,
                n_probs=n_probs,
                cache_prompt=cache_prompt,
            )
        except LlamaClientError as exc:
            raise JevError(str(exc)) from exc
        readout_used = "grammar" if readout_mode == "grammar" else "probs"
        if readout_mode == "auto" and not _slots_present(resp, pinning.slot_token_ids):
            # fall back to a grammar-constrained pass for an exact readout
            log.info("option slots missing from top-%d; retrying with grammar", n_probs)
            try:
                resp = self.client.completion(
                    rendered.prompt,
                    grammar=grammar,
                    multimodal_data=data_urls or None,
                    n_probs=n_probs,
                    cache_prompt=cache_prompt,
                )
            except LlamaClientError as exc:
                raise JevError(str(exc)) from exc
            readout_used = "grammar-fallback"
        wall_ms = (time.perf_counter() - t0) * 1000.0

        probs = _resolve_option_probs(resp, pinning.slot_token_ids)
        chosen_idx = max(range(len(probs)), key=probs.__getitem__)
        sampled_id = _sampled_token_id(resp)
        scores = [
            OptionScore(
                id=options[i]["id"],
                letter=LETTERS[i],
                description=options[i]["description"],
                token_id=pinning.slot_token_ids[i],
                probability=probs[i],
            )
            for i in range(len(options))
        ]
        timings = _timings(resp)
        timings["wall_ms"] = wall_ms
        prompt_n = int(timings.get("prompt_n", 0))
        tokens_cached = int(resp.get("tokens_cached", 0))
        return DecisionResult(
            id=decision_id,
            question=question,
            options=scores,
            chosen=options[chosen_idx]["id"],
            chosen_letter=LETTERS[chosen_idx],
            confidence=probs[chosen_idx],
            strategy=rendered.strategy,
            strategy_description=rendered.description,
            injected=rendered.injected,
            grammar=grammar,
            prompt=rendered.prompt,
            prompt_sha256=hashlib.sha256(rendered.prompt.encode()).hexdigest()[:16],
            sampled_token_id=sampled_id,
            sampled_letter_match=(sampled_id == pinning.slot_token_ids[chosen_idx]),
            has_image=bool(data_urls),
            image_count=len(data_urls),
            argmax_choice=options[chosen_idx]["id"],
            reused_tokens=max(0, tokens_cached - prompt_n),
            readout=readout_used,
            timings=timings,
            tokens_evaluated=int(timings.get("prompt_n", 0))
            + int(timings.get("predicted_n", 0)),
            cached_tokens=int(resp.get("tokens_cached", 0)),
        )

    def _dispatch(self, jobs: list, parallel: bool | None = None) -> list:
        """Run decision jobs sequentially or across the server's slots.

        Requests are independent, and llama-server with -np N batch-processes
        concurrent prompts (continuous batching), which is the only practical
        way to amortize prefill on this hybrid model: partial prompt-cache
        reuse is unavailable, so concurrency is what buys throughput.
        """
        parallel = self.cfg.engine.parallel_decisions if parallel is None else parallel
        if not parallel or len(jobs) <= 1:
            return [job() for job in jobs]
        workers = min(len(jobs), max(1, self.cfg.model.parallel))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(job) for job in jobs]
            return [future.result() for future in futures]

    def decide_many(
        self,
        questions: list[str],
        options: list[list[dict]],
        state: Any = None,
        image: Any = None,
        images: list[Any] | None = None,
        thinking: str | None = None,
        assignment: str = "none",
        objective: str = "logprob",
        parallel: bool | None = None,
    ) -> list[DecisionResult]:
        """One state/image set, many criteria: prefix cache is reused.

        assignment="none" (default): each question is decided independently by
        its own argmax. Correct for arbitrary criteria, where two items may
        legitimately share an answer.

        assignment="hungarian": declare that the answers form a one-to-one
        matching between items and options (every option used at most once).
        All items must use the same option ids, and the final choices are the
        global maximum of sum(log P) over the item x option probability matrix,
        solved with scipy's linear_sum_assignment. The independent argmax stays
        available as `argmax_choice`.
        """
        if len(questions) != len(options):
            raise ValidationError("questions and options must have equal length")
        if assignment not in ("none", "hungarian"):
            raise ValidationError(
                f"assignment must be 'none' or 'hungarian', got {assignment!r}"
            )

        normalized: list[list[dict]] | None = None
        if assignment != "none":
            if not questions:
                raise ValidationError("assignment needs at least one item")
            normalized = [
                _normalize_options(opts, self.cfg.engine.max_options)
                for opts in options
            ]
            option_ids = [o["id"] for o in normalized[0]]
            for q, opts in zip(questions, normalized):
                if [o["id"] for o in opts] != option_ids:
                    raise ValidationError(
                        "assignment requires every item to use the same option "
                        f"ids; {q!r} differs"
                    )
            if len(questions) > len(option_ids):
                raise ValidationError(
                    "assignment requires at least as many options as items"
                )

        data_urls = self._encode_sources(image, images)

        def job(i: int, question: str, opts: list[dict]):
            return lambda: self.decide(
                question=question,
                options=opts,
                state=state,
                thinking=thinking,
                decision_id=f"shared-{i}",
                cache_prompt=True,
                _data_urls=data_urls,
            )

        results = self._dispatch(
            [job(i, q, opts) for i, (q, opts) in enumerate(zip(questions, options))],
            parallel,
        )

        if assignment == "hungarian" and results and normalized is not None:
            item_ids = [r.id for r in results]
            option_ids = [o["id"] for o in normalized[0]]
            rows = {
                r.id: {o.id: o.probability for o in r.options} for r in results
            }
            argmax_map = argmax_assignment(item_ids, rows)
            mapping, method = optimal_assignment(
                item_ids, option_ids, rows, objective=objective
            )
            slot_tokens = {
                o.id: o.token_id for r in results for o in r.options
            }
            for result in results:
                assigned = mapping.get(result.id)
                result.argmax_choice = argmax_map.get(result.id)
                result.assignment_method = method
                result.assignment_changed = assigned != result.argmax_choice
                if assigned:
                    option = next(o for o in result.options if o.id == assigned)
                    result.chosen = assigned
                    result.chosen_letter = option.letter
                    result.confidence = option.probability
                    result.sampled_letter_match = (
                        result.sampled_token_id == slot_tokens.get(assigned)
                    )
        return results

    def index_inputs(
        self,
        inputs: list[dict],
        items: list[str],
        prompt: str,
        assignment: str = "hungarian",
        objective: str = "logprob",
        thinking: str | None = None,
        parallel: bool | None = None,
    ) -> list[dict]:
        """Indexing readout over mixed inputs, with a caller-supplied prompt.

        inputs: ordered list, each one of
          {"type": "image", "image": <path|bytes|PIL|base64>, "label": str?}
          {"type": "text",  "text": str, "label": str?}
        items: strings to assign to the inputs (short labels or long texts).
        prompt: the whole indexing instruction - rules and question in one
          string. It must contain "{item}", which is substituted per item,
          e.g. "Each input is a photo. Which input shows a {item}? Use each
          input at most once."

        The engine adds no indexing phrasing of its own: `prompt` is the whole
        instruction. Returns one record per item; `index` comes from the global
        one-to-one assignment of items to inputs, while `argmax_index` is the
        independent per-item argmax.
        """
        if len(inputs) < 2:
            raise ValidationError("index_inputs needs at least 2 inputs")
        if len(inputs) > self.cfg.engine.max_options:
            raise ValidationError(
                f"index_inputs supports up to {self.cfg.engine.max_options} inputs"
            )
        if not items:
            raise ValidationError("index_inputs needs at least one item")
        if "{item}" not in prompt:
            raise ValidationError(
                "prompt must contain the '{item}' placeholder, e.g. "
                "'Which input shows a {item}?'"
            )

        normalized = [
            self._normalize_input(spec, i) for i, spec in enumerate(inputs)
        ]
        data_urls = [
            encode_image(spec["source"], self.cfg.image)[0]
            for spec in normalized
            if spec["kind"] == "image"
        ]
        option_ids = [f"input-{i + 1}" for i in range(len(normalized))]
        options = [
            {"id": option_ids[i], "description": spec["label"]}
            for i, spec in enumerate(normalized)
        ]
        input_evidence = []
        for i, spec in enumerate(normalized):
            entry = {"input": i + 1, "kind": spec["kind"], "label": spec["label"]}
            if spec["kind"] == "text":
                entry["text"] = spec["text"]
            input_evidence.append(entry)
        base_state = {"inputs": input_evidence}

        item_texts = [str(item) for item in items]

        def job(index: int, item_text: str):
            return lambda: (
                item_text,
                self.decide(
                    question=prompt.replace("{item}", item_text),
                    options=options,
                    state={**base_state, "item_to_assign": item_text},
                    thinking=thinking,
                    decision_id=f"index-{index}",
                    cache_prompt=True,
                    _data_urls=data_urls,
                ),
            )

        records = []
        for item_text, result in self._dispatch(
            [job(i, text) for i, text in enumerate(item_texts)], parallel
        ):
            records.append(
                {
                    "item": item_text,
                    "confidence": result.confidence,
                    "probabilities": {o.id: o.probability for o in result.options},
                    "cached_tokens": result.cached_tokens,
                    "reused_tokens": result.reused_tokens,
                    "readout": result.readout,
                    "prompt_tokens": result.tokens_evaluated,
                    "wall_ms": result.timings.get("wall_ms", 0.0),
                    "prompt_ms": result.timings.get("prompt_ms", 0.0),
                    "result": result,
                }
            )

        rows = {r["item"]: r["probabilities"] for r in records}
        item_ids = [r["item"] for r in records]
        argmax_map = argmax_assignment(item_ids, rows)
        if assignment == "argmax":
            mapping, method = argmax_map, "argmax"
        else:
            mapping, method = optimal_assignment(
                item_ids, option_ids, rows, objective=objective
            )
        for record in records:
            assigned = mapping.get(record["item"], argmax_map.get(record["item"]))
            argmax = argmax_map.get(record["item"])
            record["index"] = int(assigned.split("-", 1)[1])
            record["argmax_index"] = int(argmax.split("-", 1)[1]) if argmax else None
            record["index_changed"] = record["index"] != record["argmax_index"]
            record["assignment_method"] = method
        return records

    @staticmethod
    def _normalize_input(spec: dict, position: int) -> dict:
        if not isinstance(spec, dict):
            raise ValidationError(f"input {position + 1} must be an object")
        label = spec.get("label") or f"input {position + 1}"
        source = spec.get("image")
        if source is None:
            source = spec.get("image_path") or spec.get("image_base64")
        text = spec.get("text")
        kind = spec.get("type")
        if kind is None:
            kind = "image" if source is not None else "text"
        if kind == "image":
            if source is None:
                raise ValidationError(
                    f"input {position + 1}: image needs image/image_path/image_base64"
                )
            return {"kind": "image", "source": source, "label": str(label)}
        if kind == "text":
            if text is None or not str(text).strip():
                raise ValidationError(
                    f"input {position + 1}: text inputs need non-empty text"
                )
            return {
                "kind": "text",
                "source": None,
                "label": str(label),
                "text": str(text),
            }
        raise ValidationError(f"input {position + 1}: unknown type {kind!r}")


    # ------------------------------------------------------------------
    def info(self) -> dict:
        model_path = str(self.cfg.model.gguf_path)
        try:
            size = self.cfg.model.gguf_path.stat().st_size
        except OSError:
            size = 0
        return {
            "model": model_path,
            "model_bytes": size,
            "mmproj": str(self.cfg.model.mmproj_path or ""),
            "device": self.cfg.describe_device(),
            "ctx_size": self.cfg.model.ctx_size,
            "parallel_slots": self.cfg.model.parallel,
            "thinking_mode": self.cfg.thinking.mode,
            "thinking_strategy": self.strategy,
            "thinking_strategy_source": self._strategy_source,
            "thinking_cache": str(self._thinking_cache_file()),
            "available_strategies": sorted(self.cfg.thinking.strategies),
            "auto_probes": [p.to_dict() for p in self._probe_results],
            "template_supports_thinking": self._supports_thinking,
            "server_url": self.cfg.server.base_url,
            "server_pid": self.server.proc.pid if self.server and self.server.proc else None,
            "slots": self._model_props.get("total_slots"),
            "chat_template_sha256": hashlib.sha256(
                self._chat_template.encode()
            ).hexdigest()[:16],
        }


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------
def _normalize_options(options: list[dict], max_options: int) -> list[dict]:
    if not isinstance(options, list) or not 2 <= len(options) <= max_options:
        raise ValidationError(f"need 2..{max_options} options, got {len(options or [])}")
    out = []
    seen = set()
    for i, opt in enumerate(options):
        if isinstance(opt, str):
            opt = {"id": opt, "description": opt}
        oid = str(opt.get("id") or f"opt-{i}")
        desc = str(opt.get("description") or oid)
        if oid in seen:
            raise ValidationError(f"duplicate option id {oid!r}")
        seen.add(oid)
        out.append({"id": oid, "description": desc})
    return out


def _resolve_option_probs(resp: dict, slot_ids: list[int]) -> list[float]:
    top = _top_tokens(resp)
    by_id = {int(t["id"]): float(t.get("prob", t.get("p", 0.0))) for t in top}
    raw = [max(0.0, by_id.get(sid, 0.0)) for sid in slot_ids]
    total = sum(raw)
    if total <= 0:
        raise ReadoutError(
            "readout failed: no option-slot probability mass returned by the "
            "server. Check that the grammar is accepted and n_probs >= option "
            f"count. Top tokens: {[t['token'] for t in top][:8]}"
        )
    return [r / total for r in raw]


def _slots_present(resp: dict, slot_ids: list[int]) -> bool:
    """True when every option token appears in the returned candidate list."""
    present = {
        int(t["id"]) for t in _top_tokens(resp) if float(t.get("prob", 0.0)) > 0.0
    }
    return all(slot_id in present for slot_id in slot_ids)


def _top_tokens(resp: dict) -> list[dict]:
    cp = resp.get("completion_probabilities") or []
    if not cp:
        return []
    entry = cp[0]
    return entry.get("top_probs") or entry.get("top_logprobs") or []


def _sampled_token_id(resp: dict) -> int:
    cp = resp.get("completion_probabilities") or []
    return int(cp[0]["id"]) if cp else -1


def _timings(resp: dict) -> dict[str, float]:
    timings = dict(resp.get("timings") or {})
    keep = (
        "prompt_n",
        "prompt_ms",
        "prompt_per_token_ms",
        "prompt_per_second",
        "predicted_n",
        "predicted_ms",
        "predicted_per_token_ms",
        "predicted_per_second",
    )
    return {k: timings[k] for k in keep if k in timings}


def _probe_messages(option_count: int = 4) -> list[dict]:
    animals = ["cat", "dog", "horse", "bird", "fish", "frog"]
    options = [
        {"id": animals[i], "description": animals[i]} for i in range(option_count)
    ]
    return build_messages(
        state="A large orange cat is sleeping on the sofa.",
        question="Which animal is described?",
        options=options,
    )
