"""Unit tests that do not need llama-server (run: .venv/bin/python -m pytest tests)."""

from __future__ import annotations

import math

import pytest

from jev.config import Config, _parse_thinking, default_strategies
from jev.engine import ValidationError, _normalize_options, _resolve_option_probs
from jev.images import encode_image, load_image
from jev.prompting import build_grammar, build_messages, render_prompt
from jev.tasks.maze import (
    bfs_distances,
    generate_maze,
    optimal_direction,
    render_maze,
    valid_moves,
)


def test_maze_is_perfect_and_renderable():
    walls = generate_maze(6, 6, seed=3)
    dist = bfs_distances(walls, (5, 5))
    assert len(dist) == 36
    for y in range(6):
        for x in range(6):
            assert (x, y) in dist
    assert optimal_direction(walls, (0, 0), (5, 5)) in valid_moves(walls, (0, 0))
    img = render_maze(walls, (0, 0), (5, 5), 32)
    assert img.size == (6 * 32 + 6, 6 * 32 + 6)


def test_image_downscale():
    class ImgCfg:
        max_side = 100
        max_pixels = 262144
        format = "png"

    img = render_maze(generate_maze(6, 6, 1), (0, 0), (5, 5), 64)
    url, info = encode_image(img, ImgCfg)
    assert url.startswith("data:image/png;base64,")
    assert max(info["width"], info["height"]) <= 100
    assert load_image(url).size == (info["width"], info["height"])


def test_grammar_and_message_shape():
    assert build_grammar(4) == 'root ::= "A" | "B" | "C" | "D"'
    msgs = build_messages(
        {"k": "v"},
        "pick one",
        [{"id": "a", "description": "first"}, {"id": "b", "description": "second"}],
    )
    assert msgs[0]["role"] == "system"
    payload = msgs[1]["content"]
    assert '"letter": "A"' in payload and '"letter": "B"' in payload


def test_option_normalization():
    opts = _normalize_options(["cat", "dog"], 16)
    assert opts == [
        {"id": "cat", "description": "cat"},
        {"id": "dog", "description": "dog"},
    ]
    with pytest.raises(ValidationError):
        _normalize_options([{"id": "a", "description": "a"}], 16)
    with pytest.raises(ValidationError):
        _normalize_options(
            [{"id": "a", "description": "a"}, {"id": "a", "description": "b"}], 16
        )


def test_option_probabilities_are_conditional_and_normalized():
    resp = {
        "completion_probabilities": [
            {
                "id": 100,
                "top_probs": [
                    {"id": 100, "token": "A", "prob": 0.6},
                    {"id": 101, "token": "B", "prob": 0.3},
                    {"id": 999, "token": "Z", "prob": 0.1},
                ],
            }
        ]
    }
    probs = _resolve_option_probs(resp, [100, 101])
    assert probs == pytest.approx([0.6 / 0.9, 0.3 / 0.9])
    assert math.isclose(sum(probs), 1.0)


def test_option_probabilities_fail_loudly_when_no_mass():
    resp = {
        "completion_probabilities": [
            {"id": 5, "top_probs": [{"id": 5, "token": "<", "prob": 0.9}]}
        ]
    }
    from jev.engine import ReadoutError

    with pytest.raises(ReadoutError):
        _resolve_option_probs(resp, [100, 101])


def test_config_handles_yaml_off_boolean():
    thinking = _parse_thinking(
        {"mode": "auto", "auto_order": ["off", False], "strategies": {False: {"inject": "x"}}}
    )
    assert thinking.strategies["off"].inject == "x"
    assert "off" in thinking.auto_order


def test_default_strategies_have_names():
    strategies = default_strategies()
    assert strategies["think_skip_cue"].inject.startswith("<think>")
    assert all(spec.name == name for name, spec in strategies.items())


def test_build_messages_with_multiple_images():
    urls = [
        "data:image/png;base64,AAAA",
        "data:image/png;base64,BBBB",
        "data:image/png;base64,CCCC",
    ]
    msgs = build_messages(
        {"item": "horse"},
        "Which image index contains the horse?",
        [{"id": "image-1", "description": "image 1"},
         {"id": "image-2", "description": "image 2"}],
        image_data_urls=urls,
    )
    parts = msgs[1]["content"]
    assert [p["type"] for p in parts] == ["image_url"] * 3 + ["text"]
    assert parts[0]["image_url"]["url"] == urls[0]
    assert parts[2]["image_url"]["url"] == urls[2]


def test_indexing_harness_passes_explicit_prompts(tmp_path, monkeypatch):
    from pathlib import Path as _Path

    import jev.artifacts as artifacts_mod
    from PIL import Image
    from jev.tasks.indexing import MIXED_PROMPT, run_indexing

    monkeypatch.setattr(artifacts_mod, "ARTIFACTS_ROOT", tmp_path / "artifacts")
    labels = ["cat", "dog", "frog", "owl", "cow", "horse"]
    images = []
    for label in labels:
        p = tmp_path / f"{label}.png"
        Image.new("RGB", (8, 8), (200, 100, 50)).save(p)
        images.append(p)

    seen_calls: list[dict] = []

    class FakeEngine:
        def index_inputs(self, inputs, items, prompt, **kwargs):
            seen_calls.append(
                {"inputs": inputs, "items": items, "prompt": prompt}
            )
            def label_of(spec: dict) -> str:
                if spec.get("type") == "image":
                    return _Path(str(spec["image"])).stem
                return str(spec.get("text", "")).rsplit(" ", 1)[-1]

            names = [label_of(spec) for spec in inputs]
            idx = {label: names.index(label) + 1 for label in labels}
            return [
                {
                    "item": item,
                    "index": idx[item],
                    "argmax_index": idx[item],
                    "index_changed": False,
                    "assignment_method": "fake",
                    "confidence": 1.0,
                    "cached_tokens": 0,
                    "prompt_tokens": 0,
                }
                for item in items
            ]

    report = run_indexing(FakeEngine(), images=images, trials=4, seed=11)
    assert report["item_accuracy"] == 1.0
    assert report["perfect_trials"] == 5  # 4 photo trials + 1 mixed trial
    for trial in report["trials"]:
        assert trial["assignments"] == trial["expected"]
    # the harness supplies its own explicit prompts, never engine defaults
    assert all(call["prompt"] for call in seen_calls)
    assert any(call["prompt"] == MIXED_PROMPT for call in seen_calls)
    assert all("{item}" in call["prompt"] for call in seen_calls)


def test_task_artifacts_reset_replaces_stale_files(tmp_path, monkeypatch):
    import jev.artifacts as artifacts_mod
    from jev.artifacts import TaskArtifacts

    monkeypatch.setattr(artifacts_mod, "ARTIFACTS_ROOT", tmp_path)
    art = TaskArtifacts("unit-task")
    art.write_text("old.txt", "stale")
    art.write_json("data.json", {"v": 1})
    assert (tmp_path / "unit-task" / "old.txt").exists()

    fresh = TaskArtifacts("unit-task")
    assert not (tmp_path / "unit-task" / "old.txt").exists()
    fresh.write_text("new.txt", "fresh")
    assert (tmp_path / "unit-task" / "new.txt").read_text() == "fresh"

    # a second instance must not delete unless reset=False
    kept = TaskArtifacts("unit-task", reset=False)
    assert (tmp_path / "unit-task" / "new.txt").exists()
    assert kept.dir == fresh.dir


def test_assignment_resolves_argmax_collisions():
    from jev.assignment import argmax_assignment, optimal_assignment

    rows = {
        "a": {"image-1": 0.55, "image-2": 0.45, "image-3": 0.0},
        "b": {"image-1": 0.54, "image-2": 0.46, "image-3": 0.0},
        "c": {"image-1": 0.10, "image-2": 0.20, "image-3": 0.70},
    }
    items = ["a", "b", "c"]
    options = ["image-1", "image-2", "image-3"]
    argmax = argmax_assignment(items, rows)
    assert argmax["a"] == argmax["b"] == "image-1"  # collision

    mapping, method = optimal_assignment(items, options, rows, objective="logprob")
    assert method == "hungarian-logprob"
    assert sorted(mapping.values()) == options  # bijection restored
    assert mapping["a"] == "image-1"
    assert mapping["b"] == "image-2"
    assert mapping["c"] == "image-3"

    linear, method2 = optimal_assignment(items, options, rows, objective="prob")
    assert method2 == "hungarian-prob"
    assert sorted(linear.values()) == options


def test_assignment_falls_back_to_argmax_when_more_items_than_options():
    from jev.assignment import optimal_assignment

    rows = {"a": {"image-1": 0.9}, "b": {"image-1": 0.1}}
    mapping, method = optimal_assignment(["a", "b"], ["image-1"], rows)
    assert method == "argmax"
    assert mapping["a"] == "image-1"


def test_index_inputs_requires_explicit_prompt(tmp_path):
    from jev.engine import JevEngine

    engine = JevEngine(Config.load())
    base = {
        "inputs": [
            {"type": "image", "image_path": "a.png"},
            {"type": "image", "image_path": "b.png"},
        ],
        "items": ["cat"],
        "prompt": "Which input shows a {item}?",
    }
    with pytest.raises(ValidationError):
        engine.index_inputs(**{**base, "prompt": "Which input shows the item?"})
    with pytest.raises(ValidationError):
        engine.index_inputs(**{**base, "inputs": base["inputs"][:1]})
    with pytest.raises(ValidationError):
        engine.index_inputs(**{**base, "items": []})


def test_thinking_strategy_cache_roundtrip(tmp_path):
    import json as _json

    from jev.engine import JevEngine

    cfg = Config.load()
    cfg.thinking.cache_path = str(tmp_path / "thinking.json")
    engine = JevEngine(cfg)
    engine._chat_template = "template with enable_thinking"
    engine._strategy = "answer_cue"
    engine._strategy_source = "probed"
    engine._probe_results = []

    assert engine._load_cached_strategy() is None
    engine._save_cached_strategy()
    assert engine._load_cached_strategy() == "answer_cue"

    # a different chat template invalidates the entry
    engine._chat_template = "a different template"
    assert engine._load_cached_strategy() is None

    # an unknown strategy name is ignored
    engine._chat_template = "template with enable_thinking"
    path = tmp_path / "thinking.json"
    data = _json.loads(path.read_text())
    key = next(iter(data))
    data[key]["strategy"] = "not-a-strategy"
    path.write_text(_json.dumps(data))
    assert engine._load_cached_strategy() is None


def test_decide_many_assignment_validation():
    from jev.engine import JevEngine

    engine = JevEngine(Config.load())
    opts_a = [{"id": "x", "description": "x"}, {"id": "y", "description": "y"}]
    opts_b = [{"id": "p", "description": "p"}, {"id": "q", "description": "q"}]
    # heterogeneous option sets cannot be a matching
    with pytest.raises(ValidationError):
        engine.decide_many(["a", "b"], [opts_a, opts_b], assignment="hungarian")
    # more items than options cannot be a one-to-one matching
    with pytest.raises(ValidationError):
        engine.decide_many(
            ["a", "b", "c"], [opts_a, opts_a, opts_a], assignment="hungarian"
        )
    # unknown assignment mode
    with pytest.raises(ValidationError):
        engine.decide_many(["a", "b"], [opts_a, opts_a], assignment="bogus")


def test_index_input_normalization():
    from jev.engine import JevEngine

    image = JevEngine._normalize_input({"image_path": "cat.png"}, 0)
    assert image["kind"] == "image"
    assert image["label"] == "input 1"
    text = JevEngine._normalize_input({"text": "a cat", "label": "note"}, 1)
    assert text["kind"] == "text"
    assert text["label"] == "note"
    with pytest.raises(ValidationError):
        JevEngine._normalize_input({"type": "text"}, 2)
    with pytest.raises(ValidationError):
        JevEngine._normalize_input({"type": "image"}, 2)


def test_render_prompt_applies_system_suffix_and_inject():
    class FakeClient:
        def apply_template(self, messages, chat_template_kwargs=None, add_generation_prompt=True):
            system = next(m for m in messages if m["role"] == "system")
            return f"<s>{system['content']}</s><assistant>{chat_template_kwargs}"

    spec = default_strategies()["system_no_reasoning"]
    rendered = render_prompt(FakeClient(), [{"role": "system", "content": "base"}], spec, True)
    assert "Do not reason" in rendered.prompt
    assert rendered.strategy == "system_no_reasoning"
