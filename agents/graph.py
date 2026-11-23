"""Supervisor graph (LangGraph).

START -> supervisor; supervisor --Command(goto=worker)--> worker --> supervisor;
judge_trials fans out with Send (one judge per trial); ask_clinician pauses with interrupt();
finish -> END. There are no edges between workers: the LLM supervisor picks every step.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, Send, interrupt
from pydantic import ValidationError

from agents import guards
from agents.fhir_reader import FhirReader
from agents.guards import GuardRejection, Limits, Outcome, forced_finish
from agents.ledger import LedgerEntry, LedgerSink
from agents.llm import LlmError, StructuredLLM
from agents.schemas import (
    Criterion,
    Decision,
    LabFact,
    PatientProfile,
    SearchSpec,
    TrialAssessment,
    TrialSummary,
    Verdict,
)
from agents.state import RunState
from agents.supervisor import decide, summarize
from agents.tracing import Tracer
from agents.workers.common import ToolCaller, ToolCallError
from agents.workers.llm_workers import (
    draft_referral,
    gather_evidence,
    judge_trial,
    parse_criteria,
    rank_trials,
    verdict_counts,
)
from agents.workers.profiler import build_profile
from agents.workers.scout import search_trials

ACTION_TO_NODE = {
    "fetch_patient_data": "profiler",
    "search_trials": "scout",
    "refine_search": "scout",
    "parse_criteria": "parser",
    "resolve_unknowns": "resolver",
    "drop_trials": "dropper",
    "gather_evidence": "evidence",
    "rank": "ranker",
    "ask_clinician": "ask_clinician",
    "draft_referrals": "drafter",
}
WORKERS = (*sorted(set(ACTION_TO_NODE.values())), "judge")


def error_label(exc: Exception) -> str:
    """Stable error name for the supervisor's state.

    The supervisor sees this text in its prompt, so it must not change when an error gets a more specific
    subclass: every LLM failure is "LlmError" (prompts stay byte-identical, cached decisions stay valid).
    """
    return "LlmError" if isinstance(exc, LlmError) else type(exc).__name__


@dataclass
class Deps:
    llm: StructuredLLM
    call_tool: ToolCaller
    reader: FhirReader
    ledger: LedgerSink
    limits: Limits = field(default_factory=Limits)
    tracer: Tracer | None = None
    parallel: int = 4


def _profile(state: dict[str, Any]) -> PatientProfile:
    if not state.get("profile"):
        raise RuntimeError("profile not loaded")
    return PatientProfile.model_validate(state["profile"])


def _as_of(state: dict[str, Any]) -> date:
    return date.fromisoformat(state["as_of"]) if state.get("as_of") else date.today()


def build_graph(deps: Deps, checkpointer: Any = None) -> Any:
    async def call_llm_decide(summary: str) -> tuple[Decision, int, int]:
        before = deps.llm.usage.tokens_in + deps.llm.usage.tokens_out
        start = time.perf_counter()
        d = await asyncio.to_thread(decide, deps.llm, summary)
        used = deps.llm.usage.tokens_in + deps.llm.usage.tokens_out - before
        return d, used, round((time.perf_counter() - start) * 1000)

    async def record(
        state: dict[str, Any], d: Decision, outcome: str, detail: str, tokens: int, ms: int, attempt: int = 0
    ) -> None:
        entry = LedgerEntry(
            attempt=attempt,
            run_id=state["run_id"],
            step=state.get("steps", 0),
            action=d.action,
            args=d.args,
            rationale=d.rationale,
            state_refs=d.state_refs,
            confidence=d.confidence,
            guard_outcome=outcome,
            guard_detail=detail,
            model=deps.llm.model,
            tokens_in=tokens,
            latency_ms=ms,
        )
        await asyncio.to_thread(deps.ledger.record, entry)  # ORM ledgers must not run on the event loop
        if deps.tracer:
            await asyncio.to_thread(deps.tracer.decision, entry)

    # ---------------- supervisor ----------------

    async def supervisor(state: RunState) -> Command[str]:
        s: dict[str, Any] = dict(state)
        tokens = 0
        ms = 0
        attempt = 0
        try:
            d, tokens, ms = await call_llm_decide(summarize(s))
            try:
                d, outcome, detail = guards.check(d, s, deps.limits)
            except GuardRejection as rej:
                await record(s, d, Outcome.REJECTED, str(rej), tokens, ms)
                attempt = 1
                d, t2, ms2 = await call_llm_decide(summarize({**s, "rejection": str(rej)}))
                tokens, ms = tokens + t2, ms + ms2
                try:
                    d, outcome, detail = guards.check(d, s, deps.limits)
                except GuardRejection as rej2:
                    d, outcome, detail = forced_finish(f"repeated rejection: {rej2}"), Outcome.FORCED_FINISH, str(rej2)
        except LlmError as exc:
            d, outcome, detail = forced_finish("orchestration_error"), "orchestration_error", str(exc)[:200]
        await record(s, d, outcome, detail, tokens, ms, attempt)

        update: dict[str, Any] = {
            "pending": d.model_dump(),
            "steps": s.get("steps", 0) + 1,
            "tokens_used": s.get("tokens_used", 0) + tokens,
            "decisions": [{**d.model_dump(), "guard_outcome": outcome}],
            "rejection": None,
        }
        if d.action == "finish":
            update.update(
                {"status": "finished", "finish_reason": outcome if outcome != Outcome.ACCEPTED else "supervisor"}
            )
            update["summary"] = str(d.args.get("summary", ""))
            return Command(goto=END, update=update)
        if d.action == "judge_trials":
            sends = [
                Send(
                    "judge",
                    {
                        "trial_id": nct,
                        "profile": s["profile"],
                        "criteria": s["trials"][nct]["criteria"],
                        "as_of": s.get("as_of"),
                    },
                )
                for nct in d.args["trial_ids"]
            ]
            return Command(goto=sends, update=update)
        return Command(goto=ACTION_TO_NODE[d.action], update=update)

    # ---------------- workers ----------------

    async def profiler(state: RunState) -> dict[str, Any]:
        s: dict[str, Any] = dict(state)
        args = s.get("pending", {}).get("args", {})
        codes = list(args.get("loinc_codes") or [])
        types = args.get("resource_types") or ["Condition", "MedicationRequest", "Observation"]
        fetched = await build_profile(
            deps.reader, s["patient_id"], resource_types=types, loinc_codes=codes, as_of=_as_of(s)
        )
        if not s.get("profile"):
            return {"profile": fetched.model_dump(mode="json")}
        current = _profile(s)  # targeted fetch: merge newer facts into the existing profile
        labs = {(lab.system, lab.code): lab for lab in current.labs}
        for lab in fetched.labs:
            old = labs.get((lab.system, lab.code))
            if old is None or (lab.effective or date.min) >= (old.effective or date.min):
                labs[(lab.system, lab.code)] = lab
        merged = current.model_copy(update={"labs": list(labs.values())})
        if "Condition" in types:
            merged.conditions = fetched.conditions
        if "MedicationRequest" in types:
            merged.medications = fetched.medications
        return {"profile": merged.model_dump(mode="json")}

    async def scout(state: RunState) -> dict[str, Any]:
        s: dict[str, Any] = dict(state)
        pending = s.get("pending", {})
        args = dict(pending.get("args", {}))
        last = (s.get("searches") or [{}])[-1].get("spec", {}) if s.get("searches") else {}
        if pending.get("action") == "refine_search":
            spec_args = {**last, **{k: v for k, v in args.items() if k in SearchSpec.model_fields and v}}
            if args.get("mode") == "broaden":
                spec_args.pop("phases", None) if "phases" not in args else None
                spec_args.pop("location", None) if "location" not in args else None
                spec_args["max_results"] = max(int(spec_args.get("max_results", 20)), 30)
        else:
            spec_args = {k: v for k, v in args.items() if k in SearchSpec.model_fields}
        try:
            spec = SearchSpec.model_validate({"max_results": 20, **spec_args})
        except ValidationError as exc:
            # Bad search arguments must not crash the run: record the failure; the supervisor sees it next step.
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            return {"searches": [{"spec": spec_args, "total": 0, "added": 0, "error": f"invalid args: {fields}"}]}
        found, total = await search_trials(spec, deps.call_tool)
        existing = s.get("trials") or {}
        new = {t.nct_id: {"summary": t.model_dump()} for t in found if t.nct_id not in existing}
        return {
            "trials": new,
            "searches": [{"spec": spec.model_dump(exclude_defaults=True), "total": total, "added": len(new)}],
        }

    async def parser(state: RunState) -> dict[str, Any]:
        s: dict[str, Any] = dict(state)
        ids = s["pending"]["args"]["trial_ids"]
        sem = asyncio.Semaphore(deps.parallel)

        async def one(nct: str) -> tuple[str, dict[str, Any]]:
            async with sem:
                try:
                    elig = await deps.call_tool("clinicaltrials_get_eligibility", {"nct_id": nct})
                    crits = await parse_criteria(deps.llm, elig.get("criteria_text", ""))
                except (ToolCallError, LlmError) as exc:
                    return nct, {"parse_error": error_label(exc)}
                return nct, {"criteria": [c.model_dump(mode="json") for c in crits], "eligibility": elig}

        results = await asyncio.gather(*(one(n) for n in ids))
        return {"trials": dict(results)}

    async def judge(payload: dict[str, Any]) -> dict[str, Any]:
        profile = PatientProfile.model_validate(payload["profile"])
        criteria = [Criterion.model_validate(c) for c in payload["criteria"]]
        try:
            verdicts = await asyncio.to_thread(judge_trial, deps.llm, profile, criteria, as_of=_as_of(payload))
        except LlmError as exc:
            return {"trials": {payload["trial_id"]: {"judge_error": error_label(exc)}}}
        return {"trials": {payload["trial_id"]: {"verdicts": [v.model_dump(mode="json") for v in verdicts]}}}

    async def resolver(state: RunState) -> dict[str, Any]:
        """Look again in the record for specific data, then re-judge only the named criteria."""
        s: dict[str, Any] = dict(state)
        args = s["pending"]["args"]
        nct = args["trial_id"]
        profile = _profile(s)
        if args.get("loinc_codes"):
            extra = await build_profile(
                deps.reader,
                s["patient_id"],
                resource_types=["Observation"],
                loinc_codes=args["loinc_codes"],
                as_of=_as_of(s),
            )
            have = {(lab.system, lab.code) for lab in profile.labs}
            profile.labs += [lab for lab in extra.labs if (lab.system, lab.code) not in have]
        wanted = {int(i) for i in args["criterion_indexes"]}
        criteria = [Criterion.model_validate(c) for c in s["trials"][nct].get("criteria") or []]
        subset = [c for c in criteria if c.index in wanted]
        fresh = await asyncio.to_thread(judge_trial, deps.llm, profile, subset, as_of=_as_of(s)) if subset else []
        by_index = {v["criterion_index"]: v for v in s["trials"][nct].get("verdicts") or []}
        by_index.update({v.criterion_index: v.model_dump(mode="json") for v in fresh})
        return {
            "profile": profile.model_dump(mode="json"),
            "trials": {nct: {"verdicts": [by_index[k] for k in sorted(by_index)]}},
        }

    async def dropper(state: RunState) -> dict[str, Any]:
        args = state["pending"]["args"]
        return {"trials": {nct: {"dropped": str(args["reason"])[:300]} for nct in args["trial_ids"]}}

    async def evidence(state: RunState) -> dict[str, Any]:
        s: dict[str, Any] = dict(state)
        out: dict[str, dict[str, Any]] = {}
        for nct in s["pending"]["args"]["trial_ids"]:
            trial = TrialSummary.model_validate(s["trials"][nct]["summary"])
            try:
                papers = await gather_evidence(deps.llm, trial, [], deps.call_tool)
            except (ToolCallError, LlmError):
                papers = []
            out[nct] = {"papers": [p.model_dump() for p in papers]}
        return {"trials": out}

    def _assessments(s: dict[str, Any], ids: list[str] | None = None) -> list[TrialAssessment]:
        trials = s.get("trials") or {}
        pool = ids or [n for n, t in trials.items() if t.get("verdicts") is not None and not t.get("dropped")]
        return [
            TrialAssessment(
                trial=TrialSummary.model_validate(trials[n]["summary"]),
                verdicts=[Verdict.model_validate(v) for v in trials[n].get("verdicts") or []],
                criteria=[Criterion.model_validate(c) for c in trials[n].get("criteria") or []],
            )
            for n in pool
            if n in trials and trials[n].get("verdicts") is not None
        ]

    async def ranker(state: RunState) -> dict[str, Any]:
        s: dict[str, Any] = dict(state)
        assessments = _assessments(s, s["pending"]["args"].get("trial_ids"))
        if not assessments:
            return {"shortlist": []}
        ranking = await asyncio.to_thread(rank_trials, deps.llm, assessments)
        counts = {a.trial.nct_id: verdict_counts(a) for a in assessments}
        return {
            "shortlist": [
                {"nct_id": r.nct_id, "rank": r.rank, "explanation": r.explanation, "counts": counts.get(r.nct_id, {})}
                for r in ranking.ranked
            ]
        }

    async def ask_clinician(state: RunState) -> dict[str, Any]:
        question = str(state["pending"]["args"]["question"])
        answer = interrupt({"run_id": state["run_id"], "question": question})
        return {"clinician_qa": [{"question": question, "answer": answer}]}

    async def drafter(state: RunState) -> dict[str, Any]:
        s: dict[str, Any] = dict(state)
        approved = set(s.get("approved_trial_ids") or [])
        wanted = s["pending"]["args"].get("trial_ids") or sorted(approved)
        ids = [n for n in wanted if n in approved]  # never draft for a trial the clinician did not approve
        profile = _profile(s)
        notes = []
        for a in _assessments(s, ids):
            note = await asyncio.to_thread(draft_referral, deps.llm, a, profile, approved=True)
            notes.append(note.model_dump())
        return {"referrals": notes}

    g: StateGraph[RunState] = StateGraph(RunState)
    g.add_node("supervisor", supervisor)
    nodes = {
        "profiler": profiler,
        "scout": scout,
        "parser": parser,
        "judge": judge,
        "resolver": resolver,
        "dropper": dropper,
        "evidence": evidence,
        "ranker": ranker,
        "ask_clinician": ask_clinician,
        "drafter": drafter,
    }
    for name, fn in nodes.items():
        g.add_node(name, fn)  # type: ignore[arg-type]  # nodes take RunState or a Send payload
        g.add_edge(name, "supervisor")  # every worker reports back; no worker-to-worker edges
    g.add_edge(START, "supervisor")
    return g.compile(checkpointer=checkpointer)


def initial_state(run_id: str, patient_id: str, as_of: date | None = None) -> RunState:
    return RunState(
        run_id=run_id,
        patient_id=patient_id,
        as_of=(as_of or date.today()).isoformat(),
        profile=None,
        trials={},
        steps=0,
        tokens_used=0,
        shortlist=[],
        approved_trial_ids=[],
        referrals=[],
        status="running",
    )


_ = LabFact  # re-exported type used by callers merging profiles
