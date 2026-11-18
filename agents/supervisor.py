"""LLM supervisor: reads a compact run summary and chooses the next action.

The supervisor decides the workflow from the data. Code here only builds the summary
(no raw record text) and calls the LLM; routing happens in agents.graph via
Command(goto=...), and guards may only reject a decision or force a finish.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from agents.llm import StructuredLLM
from agents.prompts import PROMPT_VERSION
from agents.schemas import Decision

SYSTEM = f"""[{PROMPT_VERSION}] You are the supervisor of a clinical trial matching run for one patient.
You see a summary of the run and choose exactly ONE next action. Decide from the data, not from a fixed order.

Actions and args:
- fetch_patient_data {{resource_types?: ["Condition","MedicationRequest","Observation"], loinc_codes?: [..]}}
  Load the patient profile, or fetch specific labs (LOINC) that trials need but the profile lacks.
- search_trials {{conditions, terms?, interventions?, phases?: ["PHASE2","PHASE3"], location?, max_results?}}
  Search ClinicalTrials.gov for RECRUITING trials. Use condition and drug terms only, never patient identifiers.
- refine_search {{mode: "broaden"|"narrow", reason, conditions?, terms?, phases?, location?}}
  Broaden when a search found very few trials (synonyms, drop phase filter); narrow when it found hundreds.
- parse_criteria {{trial_ids}}  Parse eligibility text into criteria (needed before judging).
- judge_trials {{trial_ids}}  Judge parsed trials against the profile (runs in parallel, max 30 trials per run).
- resolve_unknowns {{trial_id, criterion_indexes, what_to_look_for, loinc_codes?}}
  Many unknown verdicts on a promising trial: look again in the record for specific data, then re-judge.
- drop_trials {{trial_ids, reason}}  Remove trials that cannot fit (an exclusion criterion is met, or required
  inclusion criteria are clearly not met).
- gather_evidence {{trial_ids}}  Find PubMed papers for promising trials.
- rank {{trial_ids?}}  Produce the ranked shortlist from judged, non-dropped trials.
- ask_clinician {{question}}  Ask one specific question when the record cannot answer something important.
- draft_referrals {{trial_ids?}}  Only after a clinician approved the shortlist (state shows approved trials).
- finish {{summary}}  Stop. Use when the shortlist is ranked and stable, or nothing more can be done.

Rules:
- Base every choice on facts in the summary; put the ids you used (NCT ids, FHIR refs) in state_refs.
- Check the last search's total_matches before parsing its trials: fewer than 5 -> refine_search "broaden";
  more than 100 -> refine_search "narrow". Parse only once the search size is reasonable or refining is done.
- resolve_unknowns already ran for a trial and its unknowns remain: the record does not hold the answer. Do not
  re-judge without new data; ask_clinician about that criterion if the trial is otherwise promising.
- Do not repeat an action that just ran with the same args unless the state changed.
- Prefer acting on the data you have; ask the clinician only for information the record cannot provide.
- rationale: 1-3 sentences, citing the facts. Never include patient names or identifiers."""


def _trial_status(t: dict[str, Any]) -> str:
    if t.get("dropped"):
        return "dropped"
    if t.get("verdicts") is not None:
        return "judged"
    if t.get("criteria"):
        return "parsed"
    return "found"


def summarize(state: dict[str, Any], *, max_trials: int = 40) -> str:
    """Compact JSON view of the run for the supervisor. Facts carry ids; no raw record text."""
    profile = state.get("profile")
    patient: dict[str, Any] = {"loaded": profile is not None}
    if profile:
        active = [c for c in profile.get("conditions", []) if c.get("clinical_status") == "active"]
        meds = [m for m in profile.get("medications", []) if m.get("status") == "active"]
        patient.update(
            {
                "age": profile.get("age_years"),
                "sex": profile.get("sex"),
                "active_conditions": [f"{c['display']} [{c['resource_id']}]" for c in active[:20]],
                "active_medications": sorted({m["display"] for m in meds})[:15],
                "labs": [
                    f"{lab['display'][:50]} (loinc {lab['code']}) = {lab.get('value')} {lab.get('unit', '')} "
                    f"on {lab.get('effective')} [{lab['resource_id']}]"
                    for lab in profile.get("labs", [])
                    if lab.get("category") == "laboratory"
                ][:25],
                "truncated": profile.get("truncated", False),
            }
        )
    trials_out = []
    for nct, t in list((state.get("trials") or {}).items())[:max_trials]:
        verdicts = t.get("verdicts") or []
        counts = Counter(v["status"] for v in verdicts)
        kinds = {c["index"]: c["kind"] for c in t.get("criteria") or []}
        exclusion_met = sum(
            1 for v in verdicts if v["status"] == "met" and kinds.get(v["criterion_index"]) == "exclusion"
        )
        trials_out.append(
            {
                "nct_id": nct,
                "title": (t.get("summary") or {}).get("title", "")[:90],
                "status": _trial_status(t),
                "criteria": len(t.get("criteria") or []),
                "verdicts": dict(counts),
                "exclusions_met": exclusion_met,
                "unknown_ratio": round(counts.get("unknown", 0) / len(verdicts), 2) if verdicts else None,
                "papers": len(t.get("papers") or []),
                "dropped_reason": t.get("dropped"),
            }
        )
    summary = {
        "patient": patient,
        "searches": [
            {"spec": s.get("spec"), "total_matches": s.get("total"), "trials_added": s.get("added")}
            for s in (state.get("searches") or [])[-5:]
        ],
        "trials": trials_out,
        "trial_count": len(state.get("trials") or {}),
        "shortlist": [{"nct_id": r["nct_id"], "rank": r["rank"]} for r in state.get("shortlist") or []],
        "approved_trial_ids": state.get("approved_trial_ids") or [],
        "referrals_drafted": len(state.get("referrals") or []),
        "clinician_qa": state.get("clinician_qa") or [],
        "budget": {"steps_used": state.get("steps", 0), "tokens_used": state.get("tokens_used", 0)},
        "last_decisions": [
            {"step": i, "action": d["action"], "args": d.get("args", {}), "outcome": d.get("guard_outcome")}
            for i, d in list(enumerate(state.get("decisions") or []))[-5:]
        ],
    }
    if state.get("rejection"):
        summary["last_decision_rejected_because"] = state["rejection"]
    return json.dumps(summary, default=str)


def decide(llm: StructuredLLM, summary: str) -> Decision:
    return llm.complete(task="supervisor", system=SYSTEM, user=summary, schema=Decision)
