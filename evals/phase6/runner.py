"""Run the three systems over the sampled patients and score them.

Systems (all rank the same fixed pool per patient):
- bm25:        Okapi BM25 over trial text vs patient text. No LLM.
- fixed:       parse every trial -> judge every trial -> rank. Same workers as the supervisor, fixed order.
- supervisor:  the Phase 4 LLM supervisor graph, with the offline snapshot tools.

A system's ranking is its ranked list followed by any pool trials it did not rank (dropped or
unjudged), in pool order, so every system is scored on a full permutation of the pool.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

from agents.graph import Deps, build_graph, initial_state
from agents.guards import Limits
from agents.ledger import MemoryLedger
from agents.runner import paused_question, run_config
from agents.schemas import Criterion, PatientProfile, TrialAssessment, TrialSummary, Verdict
from agents.workers.llm_workers import judge_trial, parse_criteria, rank_trials
from data.loaders.common import DATA_DIR
from evals.phase6 import bm25, metrics
from evals.phase6.cache import EvalLLM
from evals.phase6.cohorts import EvalPatient
from evals.phase6.snapshot import Snapshot, available, trial_text
from evals.phase6.text_patient import patient_query, profile_from_text
from evals.phase6.tools import SnapshotTools, TextPatientReader

SYSTEMS = ("bm25", "fixed", "supervisor")
AS_OF = date(2026, 11, 16)  # text topics carry no dates; a fixed date keeps prompts (and the cache) stable
CLINICIAN_REPLY = "No further information is available beyond the case description."
MAX_CLINICIAN_REPLIES = 3
PRICES_PATH = DATA_DIR / "llm_prices.json"


@dataclass
class SystemResult:
    system: str
    ranking: list[str]
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    steps: int = 0
    clinician_questions: int = 0
    error: str = ""


@dataclass
class PatientResult:
    cohort: str
    topic_id: str
    pool: list[str]
    qrels: dict[str, int]
    valid_refs: list[str]
    systems: dict[str, SystemResult] = field(default_factory=dict)


def complete_ranking(ranked: list[str], pool: list[str]) -> list[str]:
    seen = [n for n in dict.fromkeys(ranked) if n in pool]
    return seen + [n for n in pool if n not in seen]


def _summary(trials: Snapshot, nct: str) -> TrialSummary:
    s = trials[nct]["summary"]
    return TrialSummary.model_validate({k: s.get(k) for k in TrialSummary.model_fields if k in s} | {"nct_id": nct})


async def run_fixed(llm: EvalLLM, profile: PatientProfile, trials: Snapshot, pool: list[str]) -> SystemResult:
    assessments: list[TrialAssessment] = []
    verdicts: list[dict[str, Any]] = []
    for nct in pool:  # sequential: EvalLLM serializes live calls anyway
        criteria = await parse_criteria(llm, trials[nct].get("criteria_text", ""))
        judged = await asyncio.to_thread(judge_trial, llm, profile, criteria, as_of=AS_OF)
        verdicts += [v.model_dump(mode="json") | {"nct_id": nct} for v in judged]
        assessments.append(TrialAssessment(trial=_summary(trials, nct), verdicts=judged, criteria=criteria))
    ranking = await asyncio.to_thread(rank_trials, llm, assessments) if assessments else None
    ranked = [r.nct_id for r in sorted(ranking.ranked, key=lambda r: r.rank)] if ranking else []
    return SystemResult("fixed", complete_ranking(ranked, pool), verdicts)


async def run_supervisor(
    llm: EvalLLM, topic_id: str, profile: PatientProfile, trials: Snapshot, pool: list[str], limits: Limits
) -> SystemResult:
    deps = Deps(
        llm=llm,
        call_tool=SnapshotTools(trials, pool),
        reader=TextPatientReader(topic_id),
        ledger=MemoryLedger(),
        limits=limits,
        parallel=1,
    )
    graph = build_graph(deps, checkpointer=InMemorySaver())
    config = run_config(f"phase6-{topic_id}")
    state = {**initial_state(f"phase6-{topic_id}", topic_id, AS_OF), "profile": profile.model_dump(mode="json")}
    out: dict[str, Any] = await graph.ainvoke(state, config)
    asked = 0
    from langgraph.types import Command

    while paused_question(out) and asked < MAX_CLINICIAN_REPLIES:
        asked += 1
        out = await graph.ainvoke(Command(resume={"answer": CLINICIAN_REPLY, "actor_user_id": None}), config)
    ranked = [r["nct_id"] for r in sorted(out.get("shortlist") or [], key=lambda r: r["rank"])]
    verdicts = [v | {"nct_id": nct} for nct, t in (out.get("trials") or {}).items() for v in (t.get("verdicts") or [])]
    return SystemResult(
        "supervisor",
        complete_ranking(ranked, pool),
        verdicts,
        steps=int(out.get("steps", 0)),
        clinician_questions=asked,
    )


async def run_patient(
    llm: EvalLLM, p: EvalPatient, trials: Snapshot, systems: tuple[str, ...], limits: Limits
) -> PatientResult:
    pool = available(trials, p.pool)
    profile = profile_from_text(p.topic_id, p.text)
    result = PatientResult(p.cohort, p.topic_id, pool, p.qrels, sorted(profile.resource_ids()))
    for system in systems:  # fixed before supervisor: the supervisor then reuses its parse and judge replies
        before = (llm.usage.tokens_in, llm.usage.tokens_out)
        started = time.perf_counter()
        try:
            if system == "bm25":
                res = SystemResult("bm25", bm25.rank(patient_query(p.text), {n: trial_text(trials[n]) for n in pool}))
            elif system == "fixed":
                res = await run_fixed(llm, profile, trials, pool)
            else:
                res = await run_supervisor(llm, p.topic_id, profile, trials, pool, limits)
        except Exception as exc:  # one failed patient/system is reported; the eval goes on
            from evals.phase6.cache import ReplayMiss

            if isinstance(exc, ReplayMiss):
                raise
            res = SystemResult(system, list(pool), error=type(exc).__name__)
        res.seconds = round(time.perf_counter() - started, 2)
        res.tokens_in = llm.usage.tokens_in - before[0]
        res.tokens_out = llm.usage.tokens_out - before[1]
        result.systems[system] = res
    return result


# ---------------- scoring ----------------


def price_for(model: str, path: Path = PRICES_PATH) -> dict[str, Any] | None:
    table: dict[str, Any] = json.loads(path.read_text())["models"]
    if model in table:
        return dict(table[model])
    bare = model.split(".", 1)[1] if model.split(".", 1)[0] in ("global", "us", "eu", "apac") else model
    return dict(table[bare]) if bare in table else None


def _pooled(r: PatientResult) -> PatientResult:
    return PatientResult(r.cohort, r.topic_id, r.pool, {n: r.qrels.get(n, 0) for n in r.pool}, r.valid_refs)


def score(results: list[PatientResult], price: dict[str, Any] | None) -> dict[str, Any]:
    """Per cohort and system: NDCG@5/10, P@5, MRR, unknown rate, citation validity, tokens, cost, latency."""
    table: dict[str, dict[str, dict[str, Any]]] = {}
    cohorts = sorted({r.cohort for r in results})
    for cohort in [*cohorts, "all"]:
        rows = [r for r in results if cohort in ("all", r.cohort)]
        table[cohort] = {}
        for system in SYSTEMS:
            # Re-ranking eval: relevance (and the ideal order for NDCG) comes from the patient's pool only.
            runs = [(_pooled(r), r.systems[system]) for r in rows if system in r.systems]
            if not runs:
                continue
            verdicts = [v for _, s in runs for v in s.verdicts]
            n = len(runs)
            t_in = sum(s.tokens_in for _, s in runs)
            t_out = sum(s.tokens_out for _, s in runs)
            cost = metrics.cost_usd(t_in, t_out, price)
            table[cohort][system] = {
                "patients": n,
                "ndcg@10": round(metrics.mean(metrics.ndcg_at_k(s.ranking, r.qrels, 10) for r, s in runs), 4),
                "ndcg@5": round(metrics.mean(metrics.ndcg_at_k(s.ranking, r.qrels, 5) for r, s in runs), 4),
                "p@5": round(metrics.mean(metrics.precision_at_k(s.ranking, r.qrels, 5) for r, s in runs), 4),
                "mrr": round(metrics.mean(metrics.reciprocal_rank(s.ranking, r.qrels) for r, s in runs), 4),
                "unknown_rate": round(metrics.unknown_rate(verdicts), 4) if verdicts else None,
                "citation_validity": (
                    round(
                        metrics.mean(
                            metrics.citation_validity(s.verdicts, set(r.valid_refs)) for r, s in runs if s.verdicts
                        ),
                        4,
                    )
                    if verdicts
                    else None
                ),
                "tokens_per_patient": round((t_in + t_out) / n),
                "cost_per_patient_usd": round(cost / n, 5) if cost is not None else None,
                "seconds_per_patient": round(metrics.mean(s.seconds for _, s in runs), 2),
                "errors": sum(bool(s.error) for _, s in runs),
            }
    return table


def check_thresholds(table: dict[str, Any], thresholds: dict[str, Any] | None) -> list[str]:
    """Failures against data/eval_thresholds.json "phase6" (set after baseline). Empty list = pass."""
    if thresholds is None:
        return ["phase6 thresholds not set (baseline run): record them with an ADR and owner approval"]
    margin_bm25 = float(thresholds.get("supervisor_minus_bm25_ndcg10", 0.0))
    margin_fixed = float(thresholds.get("supervisor_minus_fixed_ndcg10", 0.0))
    fails = []
    for cohort, systems in table.items():
        if cohort == "all" or not {"supervisor", "bm25", "fixed"} <= set(systems):
            continue
        sup, base, fixed = systems["supervisor"], systems["bm25"], systems["fixed"]
        if sup["ndcg@10"] - base["ndcg@10"] <= margin_bm25:
            fails.append(f"{cohort}: supervisor NDCG@10 {sup['ndcg@10']} does not beat BM25 {base['ndcg@10']}")
        cheaper = (sup["cost_per_patient_usd"] or 0) < (fixed["cost_per_patient_usd"] or 0)
        if sup["ndcg@10"] - fixed["ndcg@10"] < margin_fixed and not cheaper:
            fails.append(
                f"{cohort}: supervisor NDCG@10 {sup['ndcg@10']} below fixed pipeline {fixed['ndcg@10']} "
                "without lower cost"
            )
    if "citation_validity" in thresholds:
        for cohort, systems in table.items():
            for name, row in systems.items():
                cv = row.get("citation_validity")
                if cv is not None and cv < float(thresholds["citation_validity"]):
                    fails.append(f"{cohort}/{name}: citation validity {cv} < {thresholds['citation_validity']}")
    return fails


def results_json(results: list[PatientResult]) -> list[dict[str, Any]]:
    return [asdict(r) for r in results]


_ = (Criterion, Verdict)  # schema types used in the serialized verdicts
