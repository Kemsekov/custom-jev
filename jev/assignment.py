"""Globally consistent assignment of items to options.

Indexing decisions produce a probability for every (item, option) pair. Taking
the per-item argmax independently can collide (two items mapped to the same
image) and leaves the mapping non-bijective. Instead we maximise the total
probability subject to a one-to-one constraint, i.e. a linear sum assignment
problem solved with SciPy's Hungarian algorithm.
"""

from __future__ import annotations

import logging
import math

log = logging.getLogger("jev.assignment")


def optimal_assignment(
    item_ids: list[str],
    option_ids: list[str],
    probability_rows: dict[str, dict[str, float]],
    objective: str = "logprob",
) -> tuple[dict[str, str], str]:
    """Map each item to a distinct option id, maximising the joint score.

    objective="logprob" maximises sum(log P(item|option)), i.e. the MAP
    permutation under independent categorical rows (the principled choice).
    objective="prob" maximises sum(P), i.e. the expected number of correct
    matches. Falls back to greedy matching if SciPy is missing and to per-item
    argmax when there are more items than options.
    """
    n_items, n_options = len(item_ids), len(option_ids)
    if n_items == 0:
        return {}, "empty"
    if n_items > n_options:
        return argmax_assignment(item_ids, probability_rows), "argmax"

    try:
        import numpy as np
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # pragma: no cover - scipy is a declared dependency
        log.warning("scipy unavailable; using greedy assignment")
        return _greedy(item_ids, option_ids, probability_rows)

    # Small floor keeps log finite; probabilities below it are indistinguishable
    # from "not this option" for assignment purposes.
    floor = 1e-6
    cost = np.zeros((n_items, n_options), dtype=float)
    for i, item in enumerate(item_ids):
        row = probability_rows.get(item, {})
        for j, option in enumerate(option_ids):
            p = float(row.get(option, 0.0))
            if objective == "logprob":
                cost[i, j] = -math.log(max(p, floor))
            else:
                cost[i, j] = -max(p, 0.0)
    rows, cols = linear_sum_assignment(cost)
    mapping = {item_ids[i]: option_ids[cols[i]] for i in rows}
    return mapping, f"hungarian-{objective}"


def _greedy(
    item_ids: list[str],
    option_ids: list[str],
    probability_rows: dict[str, dict[str, float]],
) -> tuple[dict[str, str], str]:
    pairs = [
        (float(probability_rows.get(item, {}).get(option, 0.0)), item, option)
        for item in item_ids
        for option in option_ids
    ]
    pairs.sort(reverse=True)
    mapping: dict[str, str] = {}
    used_options: set[str] = set()
    for _, item, option in pairs:
        if item in mapping or option in used_options:
            continue
        mapping[item] = option
        used_options.add(option)
        if len(mapping) == len(item_ids):
            break
    return mapping, "greedy"


def argmax_assignment(
    item_ids: list[str],
    probability_rows: dict[str, dict[str, float]],
) -> dict[str, str]:
    mapping = {}
    for item in item_ids:
        row = probability_rows.get(item, {})
        if row:
            mapping[item] = max(row, key=row.get)
    return mapping
