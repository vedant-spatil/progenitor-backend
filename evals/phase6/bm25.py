"""Okapi BM25 baseline: rank a pool of trials by their text against the patient text. No LLM."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Mapping

from agents.workers.common import STOPWORDS

_TOKEN = re.compile(r"[a-z0-9]+")


def tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def rank(query: str, docs: Mapping[str, str], k1: float = 1.2, b: float = 0.75) -> list[str]:
    """Doc ids by descending BM25 score; ties broken by id so the order is deterministic."""
    toks = {d: tokens(t) for d, t in docs.items()}
    n = len(toks)
    if n == 0:
        return []
    avgdl = sum(len(t) for t in toks.values()) / n or 1.0
    df: Counter[str] = Counter()
    for t in toks.values():
        df.update(set(t))
    q = tokens(query)

    def score(doc: str) -> float:
        tf = Counter(toks[doc])
        dl = len(toks[doc])
        total = 0.0
        for term in q:
            if term not in tf:
                continue
            idf = math.log(1 + (n - df[term] + 0.5) / (df[term] + 0.5))
            total += idf * tf[term] * (k1 + 1) / (tf[term] + k1 * (1 - b + b * dl / avgdl))
        return total

    return sorted(toks, key=lambda d: (-score(d), d))
