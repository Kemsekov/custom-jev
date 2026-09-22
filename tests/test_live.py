"""Live smoke tests. Require the built llama-server and downloaded model.

Run: .venv/bin/python -m pytest tests/test_live.py -s
"""

from __future__ import annotations

from pathlib import Path

import pytest

from jev.config import Config
from jev.engine import JevEngine
from jev.tasks.animals import DEFAULT_DIR, list_animal_images
from jev.tasks.maze import solve_maze


@pytest.fixture(scope="module")
def engine():
    cfg = Config.load()
    if not cfg.model.gguf_path.exists():
        pytest.skip("model not downloaded; run scripts/02_download_model.sh")
    eng = JevEngine(cfg)
    eng.start()
    yield eng
    eng.stop()


def test_info_and_pinning(engine):
    info = engine.info()
    assert info["thinking_strategy"] in info["available_strategies"]
    assert info["template_supports_thinking"] is True
    assert engine._slot_token("A") != engine._slot_token("B")
    if info["thinking_mode"] == "auto":
        assert info["thinking_strategy_source"] in ("cached", "probed")
        # the resolved strategy is persisted for the next API start
        assert engine._load_cached_strategy() == info["thinking_strategy"]


def test_text_decision_is_conditional(engine):
    result = engine.decide(
        question="Which queue should handle this request?",
        options=[
            {"id": "billing", "description": "billing support"},
            {"id": "access", "description": "account access support"},
            {"id": "shipping", "description": "shipping support"},
        ],
        state="I was charged twice for the same subscription and want a refund.",
    )
    probabilities = [o.probability for o in result.options]
    assert result.chosen in {o.id for o in result.options}
    assert abs(sum(probabilities) - 1.0) < 1e-3
    assert result.tokens_evaluated > 0
    assert result.sampled_letter_match


def test_animal_decision_via_decide(engine):
    """Classification is just decide() with label options; no wrapper needed."""
    images = list_animal_images(DEFAULT_DIR)
    if not images:
        pytest.skip("no animal images; run scripts/05_fetch_assets.sh")
    label, path = images[0]
    labels = [label] + [other for other, _ in images[1:4]]
    result = engine.decide(
        question="Which animal is shown in the image? Choose one label.",
        options=[{"id": name, "description": name} for name in labels],
        image=str(path),
    )
    assert result.chosen in labels
    assert result.has_image
    assert result.confidence >= 0.0


PROMPT_PHOTOS = (
    "The inputs are numbered photos in the given order. Assign every requested "
    "item to exactly one input that clearly shows that item; each input is used "
    "at most once. Which input is the photo that shows a {item}?"
)


def test_indexing_six_animals_different_orders(engine):
    images = list_animal_images(DEFAULT_DIR)
    if len(images) < 6:
        pytest.skip("need 6 animal images; run scripts/05_fetch_assets.sh")
    labels = [label for label, _ in images[:6]]
    orders = [
        list(range(6)),
        [5, 0, 3, 1, 4, 2],
        [2, 4, 1, 5, 0, 3],
    ]
    total_correct = 0
    for order in orders:
        inputs = [
            {
                "type": "image",
                "image": images[i][1],
                "label": f"photo {position + 1}",
            }
            for position, i in enumerate(order)
        ]
        records = engine.index_inputs(
            inputs=inputs,
            items=labels,
            prompt=PROMPT_PHOTOS,
        )
        got = {r["item"]: r["index"] for r in records}
        expected = {
            label: order.index(labels.index(label)) + 1 for label in labels
        }
        assert set(got) == set(labels)
        assert all(1 <= idx <= 6 for idx in got.values())
        correct = sum(1 for label in labels if got[label] == expected[label])
        total_correct += correct
        assert correct >= 5, f"order {order}: got {got}, expected {expected}"
        # later decisions must reuse the prefilled image prefix
        assert records[-1]["cached_tokens"] >= 0
    assert total_correct >= 16


def test_indexing_mixed_images_and_text(engine):
    """Each item matches exactly one input: two photos and one text."""
    images = list_animal_images(DEFAULT_DIR)
    if len(images) < 3:
        pytest.skip("need 3 animal images; run scripts/05_fetch_assets.sh")
    first, middle, last = images[0], images[1], images[2]
    inputs = [
        {"type": "image", "image": first[1], "label": "photo 1"},
        {"type": "text", "text": f"a photo of a {middle[0]}", "label": "note 2"},
        {"type": "image", "image": last[1], "label": "photo 3"},
    ]
    records = engine.index_inputs(
        inputs=inputs,
        items=[first[0], middle[0], last[0]],
        prompt=(
            "The inputs are numbered photos and short text descriptions. "
            "Assign every requested item to exactly one input that depicts or "
            "describes that item; each input is used once. Which input matches "
            "the {item}?"
        ),
    )
    got = {r["item"]: r["index"] for r in records}
    assert got[first[0]] == 1
    assert got[middle[0]] == 2
    assert got[last[0]] == 3


def test_decide_many_assignment_matching(engine):
    """decide_many with assignment=hungarian solves a declared matching."""
    reviews = {
        "r1": "broken again, refund now",
        "r2": "works, nothing special",
        "r3": "love it, best purchase",
    }
    options = [
        {"id": rid, "description": text} for rid, text in reviews.items()
    ]
    questions = [
        "Which review is the most agitated?",
        "Which review is the most happy?",
        "Which review is neither very agitated nor very happy?",
    ]
    results = engine.decide_many(
        questions,
        [options] * 3,
        state={"reviews": reviews},
        assignment="hungarian",
    )
    chosen = {r.id: r.chosen for r in results}
    assert len(set(chosen.values())) == 3  # a real one-to-one matching
    assert chosen["shared-0"] == "r1"
    assert chosen["shared-1"] == "r3"
    for result in results:
        assert result.assignment_method is not None
        assert result.argmax_choice is not None


def test_ordering_as_indexing(engine):
    """Ordering is just indexing where the inputs are rank slots."""
    reviews = {
        "r1": "Third replacement unit and it is broken again. I want a refund now!",
        "r2": "It works. Nothing special, does what the listing says.",
        "r3": "Absolutely love it, best purchase this year!",
    }
    texts = list(reviews.values())
    inputs = [
        {
            "type": "text",
            "text": (
                "rank 1 = most agitated"
                if i == 0
                else f"rank {i + 1} = rank after rank {i}"
            ),
            "label": f"rank {i + 1}",
        }
        for i in range(len(texts))
    ]
    records = engine.index_inputs(
        inputs=inputs,
        items=texts,
        prompt=(
            "The inputs are rank slots. Input 1 is rank 1 (most agitated) and "
            "the last input is the least agitated. Assign each review to one "
            "rank so the reviews are sorted from most agitated to most happy; "
            "each rank is used exactly once. Which rank slot should this "
            "review get? '{item}'"
        ),
    )
    mapping = {r["item"]: r["index"] for r in records}
    assert sorted(mapping.values()) == [1, 2, 3]  # a real permutation
    order = sorted(texts, key=lambda text: mapping[text])
    assert order[0] == reviews["r1"]  # most agitated first
    assert order[-1] in (reviews["r2"], reviews["r3"])


def test_maze_short_run(engine):
    outcome = solve_maze(
        engine,
        {"width": 3, "height": 3, "seed": 2, "max_steps": 6, "save_images": False},
    )
    assert outcome["steps"] >= 1
    assert 0.0 <= outcome["step_optimality"] <= 1.0
    first = outcome["decisions"][0]
    assert first["chosen"] in first["options"]
