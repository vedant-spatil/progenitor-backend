"""TrialGPT cohorts: patients, qrels, seeded patient sample, and a fixed candidate pool per patient."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from data.loaders.eval_data import EVAL_DIR

COHORTS = ("sigir", "trec_2021", "trec_2022")


@dataclass
class EvalPatient:
    cohort: str
    topic_id: str
    text: str
    qrels: dict[str, int]  # nct_id -> 0/1/2
    pool: list[str] = field(default_factory=list)


def cohort_dir(cohort: str, root: Path | None = None) -> Path:
    return (root or EVAL_DIR / "trialgpt") / cohort


def load_cohort(cohort: str, root: Path | None = None) -> list[EvalPatient]:
    base = cohort_dir(cohort, root)
    qrels: dict[str, dict[str, int]] = defaultdict(dict)
    with (base / "qrels" / "test.tsv").open() as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])
    patients = []
    with (base / "queries.jsonl").open() as fh:
        for line in fh:
            q = json.loads(line)
            if q["_id"] in qrels:
                patients.append(EvalPatient(cohort, q["_id"], q["text"], dict(qrels[q["_id"]])))
    return sorted(patients, key=lambda p: p.topic_id)


def sample(patients: list[EvalPatient], n: int, seed: int) -> list[EvalPatient]:
    """Seeded sample, keeping only patients with at least one eligible trial (NDCG/MRR need one)."""
    usable = [p for p in patients if 2 in p.qrels.values()]
    if n >= len(usable):
        return usable
    return sorted(random.Random(seed).sample(usable, n), key=lambda p: p.topic_id)  # noqa: S311  # nosec B311 - seeded sampling, not security


def build_pool(p: EvalPatient, size: int, seed: int) -> list[str]:
    """Fixed candidate pool: stratified over relevance (eligible / excluded / not relevant), seeded.

    All three systems rank the same pool, so NDCG@10 compares rankers on equal input. Strata are
    filled round-robin starting with eligible trials, so every pool has positives when they exist.
    """
    rng = random.Random(f"{seed}:{p.topic_id}")  # noqa: S311  # nosec B311 - seeded sampling, not security
    strata = {rel: sorted(n for n, r in p.qrels.items() if r == rel) for rel in (2, 1, 0)}
    for ids in strata.values():
        rng.shuffle(ids)
    pool: list[str] = []
    while len(pool) < size and any(strata.values()):
        for rel in (2, 1, 0):
            if strata[rel] and len(pool) < size:
                pool.append(strata[rel].pop())
    return sorted(pool)


def select(per_cohort: int, pool_size: int, seed: int, cohorts: tuple[str, ...] = COHORTS) -> list[EvalPatient]:
    chosen: list[EvalPatient] = []
    for cohort in cohorts:
        for p in sample(load_cohort(cohort), per_cohort, seed):
            p.pool = build_pool(p, pool_size, seed)
            chosen.append(p)
    return chosen
