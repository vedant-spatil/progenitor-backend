"""Ranking and verdict metrics (PLAN.md Phase 6). Pure functions.

Relevance follows TREC: 2 = eligible, 1 = excluded, 0 = not relevant. "Eligible" for P@k and MRR
means rel == 2. A trial missing from the qrels counts as rel 0.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

ELIGIBLE = 2


def dcg(rels: Sequence[int], k: int) -> float:
    return float(sum((2**r - 1) / math.log2(i + 2) for i, r in enumerate(rels[:k])))


def ndcg_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    """NDCG@k of a ranking against graded relevance. The ideal order uses every judged trial for the patient."""
    ideal = dcg(sorted(qrels.values(), reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg([qrels.get(n, 0) for n in ranking], k) / ideal


def precision_at_k(ranking: Sequence[str], qrels: Mapping[str, int], k: int = 10) -> float:
    return sum(qrels.get(n, 0) == ELIGIBLE for n in ranking[:k]) / k


def reciprocal_rank(ranking: Sequence[str], qrels: Mapping[str, int]) -> float:
    for i, n in enumerate(ranking, start=1):
        if qrels.get(n, 0) == ELIGIBLE:
            return 1.0 / i
    return 0.0


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def criterion_accuracy(pairs: Iterable[tuple[str, str]]) -> float:
    """(expected, predicted) verdict statuses on a labelled set."""
    items = list(pairs)
    return sum(e == p for e, p in items) / len(items) if items else 0.0


def unknown_rate(verdicts: Iterable[Mapping[str, Any]]) -> float:
    items = list(verdicts)
    return sum(v["status"] == "unknown" for v in items) / len(items) if items else 0.0


def citation_validity(verdicts: Iterable[Mapping[str, Any]], valid_refs: set[str]) -> float:
    """Share of met/not_met verdicts that cite at least one id present in the patient's profile."""
    evidenced = [v for v in verdicts if v["status"] in ("met", "not_met")]
    if not evidenced:
        return 1.0
    return sum(any(r in valid_refs for r in v.get("evidence_refs") or []) for v in evidenced) / len(evidenced)


def cost_usd(tokens_in: int, tokens_out: int, price: Mapping[str, float | None] | None) -> float | None:
    """Σ(tokens_in·price_in + tokens_out·price_out) / 1e6; None when the model has no pinned price."""
    if not price:
        return None
    p_in, p_out = price.get("input"), price.get("output")
    if p_in is None or p_out is None:
        return None
    return (tokens_in * float(p_in) + tokens_out * float(p_out)) / 1e6
