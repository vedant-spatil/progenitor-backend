"""LLM workers: Criteria Parser, Eligibility Judge, Evidence Agent, Ranker, Referral Drafter.

Each takes typed input, makes one structured LLM call (plus deterministic tool
calls), and returns the Pydantic models in agents.schemas.
"""

from __future__ import annotations

import asyncio
import json
from datetime import date

from agents import prompts
from agents.guards import enforce_citations, enforce_relevance
from agents.llm import StructuredLLM
from agents.schemas import (
    Criterion,
    EntityType,
    EvidenceOutput,
    JudgeOutput,
    Paper,
    ParsedCriteria,
    PatientProfile,
    RankedTrial,
    Ranking,
    ReferralNote,
    TrialAssessment,
    TrialSummary,
    Verdict,
    VerdictStatus,
)
from agents.telemetry import timed
from agents.workers.common import ToolCaller, ToolCallError, criteria_lines, focus_terms, profile_context

MAX_CODED_ENTITIES = 25


# ---------------- Criteria Parser ----------------


async def parse_criteria(
    llm: StructuredLLM, criteria_text: str, *, call_tool: ToolCaller | None = None
) -> list[Criterion]:
    """Eligibility text -> criteria. With `call_tool`, attaches LOINC / RxNorm / ICD-10 codes to entities."""
    with timed("worker", worker="parser"):
        # LLM calls block (HTTP + DB cache); run them off the event loop.
        out = await asyncio.to_thread(
            llm.complete, task="parse_criteria", system=prompts.PARSER, user=criteria_text, schema=ParsedCriteria
        )
        criteria = [c.model_copy(update={"index": i}) for i, c in enumerate(out.criteria)]
        if call_tool:
            await _attach_codes(criteria, call_tool)
    return criteria


async def _attach_codes(criteria: list[Criterion], call_tool: ToolCaller) -> None:
    lookups = {
        EntityType.LAB: ("terminology_lookup_loinc", "name", "loinc"),
        EntityType.DRUG: ("terminology_normalize_drug", "name", "rxnorm"),
        EntityType.CONDITION: ("terminology_lookup_icd10", "text", "icd10cm"),
    }
    cache: dict[tuple[EntityType, str], list[dict[str, str]]] = {}
    for crit in criteria:
        for ent in crit.entities:
            if ent.type not in lookups:
                continue
            key = (ent.type, ent.text.lower())
            if key not in cache:
                if len(cache) >= MAX_CODED_ENTITIES:
                    continue
                tool, arg, system = lookups[ent.type]
                try:
                    res = await call_tool(tool, {arg: ent.text})
                except ToolCallError:
                    cache[key] = []
                    continue
                if ent.type == EntityType.LAB:
                    best = res.get("preferred") or next(iter(res.get("matches", [])), None)
                    cache[key] = [{"system": system, "code": best["code"]}] if best else []
                elif ent.type == EntityType.DRUG:
                    best = next(iter(res.get("candidates", [])), None)
                    cache[key] = [{"system": system, "code": best["rxcui"]}] if best else []
                else:
                    best = next(iter(res.get("matches", [])), None)
                    cache[key] = [{"system": system, "code": best["code"]}] if best else []
            ent.codes = cache[key]


# ---------------- Eligibility Judge ----------------


def judge_trial(
    llm: StructuredLLM, profile: PatientProfile, criteria: list[Criterion], *, as_of: date | None = None
) -> list[Verdict]:
    """One verdict per criterion, citations enforced. Missing verdicts become 'unknown'."""
    with timed("worker", worker="judge", n_criteria=len(criteria)):
        focus = focus_terms([c.source_text for c in criteria] + [c.entity for c in criteria])
        text, aliases = profile_context(profile, as_of, focus=focus)
        user = f"PATIENT PROFILE\n{text}\n\nCRITERIA\n{criteria_lines(criteria)}"
        out = llm.complete(task="judge_trial", system=prompts.JUDGE, user=user, schema=JudgeOutput)
        # Map short ids back to FHIR refs; anything else is kept as-is for the citation guard to check.
        by_index = {
            v.criterion_index: v.model_copy(
                update={"evidence_refs": [aliases.get(r.strip(), r.strip()) for r in v.evidence_refs]}
            )
            for v in out.verdicts
        }
        verdicts = [
            by_index.get(c.index)
            or Verdict(criterion_index=c.index, status=VerdictStatus.UNKNOWN, reason="no verdict returned")
            for c in criteria
        ]
        return enforce_relevance(enforce_citations(verdicts, profile.resource_ids()), criteria, profile)


# ---------------- Evidence Agent ----------------


async def gather_evidence(
    llm: StructuredLLM,
    trial: TrialSummary,
    interventions: list[str],
    call_tool: ToolCaller,
    max_results: int = 8,
) -> list[Paper]:
    with timed("worker", worker="evidence"):
        query = " ".join(filter(None, [" OR ".join(trial.conditions[:2]), " ".join(interventions[:2])])) or trial.title
        found = await call_tool("pubmed_search_pubmed", {"query": query, "max_results": max_results, "min_year": 2015})
        articles = found.get("articles", [])
        if not articles:
            return []
        user = json.dumps(
            {
                "trial": {
                    "title": trial.title,
                    "conditions": trial.conditions,
                    "interventions": interventions,
                },
                "results": articles,
            }
        )
        out = await asyncio.to_thread(
            llm.complete, task="gather_evidence", system=prompts.EVIDENCE, user=user, schema=EvidenceOutput
        )
        known = {a["pmid"] for a in articles}
        return [p for p in out.papers if p.pmid in known][:5]


# ---------------- Ranker ----------------


def verdict_counts(assessment: TrialAssessment) -> dict[str, int]:
    """Display-only counts (code); ranking itself is the LLM's job."""
    kinds = {c.index: c.kind for c in assessment.criteria}
    counts = {"inclusion_met": 0, "inclusion_not_met": 0, "exclusion_met": 0, "unknown": 0}
    for v in assessment.verdicts:
        if v.status == VerdictStatus.UNKNOWN:
            counts["unknown"] += 1
        elif kinds.get(v.criterion_index) == "exclusion":
            counts["exclusion_met"] += v.status == VerdictStatus.MET
        else:
            counts["inclusion_met" if v.status == VerdictStatus.MET else "inclusion_not_met"] += 1
    return counts


def rank_trials(llm: StructuredLLM, assessments: list[TrialAssessment]) -> Ranking:
    with timed("worker", worker="ranker", n_trials=len(assessments)):
        payload = []
        for a in assessments:
            kinds = {c.index: (c.kind, c.source_text[:160]) for c in a.criteria}
            payload.append(
                {
                    "nct_id": a.trial.nct_id,
                    "title": a.trial.title,
                    "counts": verdict_counts(a),
                    "verdicts": [
                        {
                            "kind": kinds.get(v.criterion_index, ("", ""))[0],
                            "criterion": kinds.get(v.criterion_index, ("", ""))[1],
                            "status": v.status,
                            "reason": v.reason[:200],
                        }
                        for v in a.verdicts
                    ],
                    "papers": len(a.papers),
                }
            )
        out = llm.complete(task="rank_trials", system=prompts.RANKER, user=json.dumps(payload), schema=Ranking)
        expected = {a.trial.nct_id for a in assessments}
        ranked: list[RankedTrial] = []
        seen: set[str] = set()
        for r in sorted(out.ranked, key=lambda r: r.rank):
            if r.nct_id in expected and r.nct_id not in seen:
                seen.add(r.nct_id)
                ranked.append(r)
        # Trials the model skipped go to the bottom, in input order.
        ranked += [
            RankedTrial(nct_id=a.trial.nct_id, rank=0, explanation="not ranked by model")
            for a in assessments
            if a.trial.nct_id not in seen
        ]
        return Ranking(ranked=[r.model_copy(update={"rank": i}) for i, r in enumerate(ranked, start=1)])


# ---------------- Referral Drafter ----------------


class NotApprovedError(RuntimeError):
    pass


def draft_referral(
    llm: StructuredLLM, assessment: TrialAssessment, profile: PatientProfile, *, approved: bool
) -> ReferralNote:
    """Draft only after clinician approval (checked here and by the Phase 4 approval guard)."""
    if not approved:
        raise NotApprovedError("draft_referrals runs only after clinician approval")
    with timed("worker", worker="drafter"):
        kinds = {c.index: c for c in assessment.criteria}
        lines = [
            f"- ({kinds[v.criterion_index].kind}) {kinds[v.criterion_index].source_text[:200]}: {v.status} — {v.reason}"
            for v in assessment.verdicts
            if v.criterion_index in kinds
        ]
        user = (
            f"TRIAL {assessment.trial.nct_id}: {assessment.trial.title}\n"
            f"Patient: age {profile.age_years}, sex {profile.sex}\n"
            "Criteria assessment:\n" + "\n".join(lines)
        )
        note = llm.complete(task="draft_referral", system=prompts.DRAFTER, user=user, schema=ReferralNote)
        return note.model_copy(update={"nct_id": assessment.trial.nct_id})
