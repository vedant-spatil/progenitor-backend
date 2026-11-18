"""Gate 4 scenario suite: does the LLM supervisor choose the right next action from the data?

Each scenario is a run state (real seeded Synthea profile + synthetic trials) and a check on the
supervisor's decision. Every scenario runs N times (default 3) without the response cache; the
majority result counts. Gate: pass rate >= 0.80. Also checks PHI: no decision args may contain
the seeded patient's identifiers.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

from agents.fhir_reader import HttpFhirReader
from agents.guards import GuardRejection, Limits, check
from agents.llm import StructuredLLM
from agents.schemas import Decision
from agents.supervisor import decide, summarize
from agents.workers.profiler import build_profile
from data.loaders.common import BACKEND_DIR
from mcp_servers.common.phi import KnownIdentifiers, PhiBlockedError, check_outbound

REPORT_DIR = BACKEND_DIR / "docs" / "eval-reports"
PASS_RATE_GATE = 0.80

Check = Callable[[Decision], bool]


@dataclass(frozen=True)
class Scenario:
    name: str
    expect: str  # human-readable expected behaviour
    state: dict[str, Any]
    ok: Check


def _trial(nct: str, title: str, **extra: Any) -> dict[str, Any]:
    return {"summary": {"nct_id": nct, "title": title, "status": "RECRUITING", "phases": ["PHASE3"]}, **extra}


def _crit(i: int, kind: str, text: str, entity: str = "") -> dict[str, Any]:
    return {"index": i, "kind": kind, "source_text": text, "entity": entity}


def _v(i: int, status: str, refs: list[str] | None = None) -> dict[str, Any]:
    return {"criterion_index": i, "status": status, "evidence_refs": refs or [], "reason": ""}


def _base(profile: dict[str, Any] | None, **extra: Any) -> dict[str, Any]:
    return {
        "run_id": "scenario",
        "patient_id": "p",
        "as_of": "2026-11-16",
        "profile": profile,
        "trials": {},
        "searches": [],
        "decisions": [],
        "clinician_qa": [],
        "steps": 3,
        "tokens_used": 0,
        "shortlist": [],
        "approved_trial_ids": [],
        "referrals": [],
        **extra,
    }


def _search(conditions: str, total: int, added: int, **spec: Any) -> dict[str, Any]:
    return {"spec": {"conditions": conditions, **spec}, "total": total, "added": added}


def _hist(*actions: tuple[str, dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"action": a, "args": args, "rationale": "", "confidence": 1, "guard_outcome": "accepted"}
        for a, args in actions
    ]


def ids(d: Decision, key: str) -> list[str]:
    value = d.args.get(key) or []
    return [str(v) for v in value] if isinstance(value, list) else []


def build_scenarios(profile: dict[str, Any]) -> list[Scenario]:
    no_hba1c = {**profile, "labs": [lab for lab in profile["labs"] if lab["code"] != "4548-4"]}
    diabetes_trials = {
        "NCT90000001": _trial(
            "NCT90000001",
            "Once-weekly insulin in type 2 diabetes",
            criteria=[
                _crit(0, "inclusion", "Type 2 diabetes for at least 1 year", "type 2 diabetes"),
                _crit(1, "inclusion", "HbA1c between 7.0% and 10.5%", "HbA1c"),
                _crit(2, "exclusion", "eGFR below 30 mL/min/1.73 m2", "eGFR"),
            ],
            verdicts=[_v(0, "met", ["Condition/x"]), _v(1, "unknown"), _v(2, "not_met", ["Observation/y"])],
        ),
        "NCT90000002": _trial(
            "NCT90000002",
            "Oral GLP-1 agonist for glycemic control",
            criteria=[_crit(0, "inclusion", "HbA1c 7.5% to 11%", "HbA1c"), _crit(1, "inclusion", "BMI 25-45", "BMI")],
            verdicts=[_v(0, "unknown"), _v(1, "met", ["Observation/z"])],
        ),
    }
    judged_ok = {
        "NCT91000001": _trial(
            "NCT91000001",
            "SGLT2 inhibitor in diabetic kidney disease",
            criteria=[_crit(0, "inclusion", "Type 2 diabetes"), _crit(1, "inclusion", "Age 40-80")],
            verdicts=[_v(0, "met", ["Condition/a"]), _v(1, "met", ["Patient/p"])],
        ),
        "NCT91000002": _trial(
            "NCT91000002",
            "Lifestyle program for prediabetes",
            criteria=[_crit(0, "inclusion", "Prediabetes"), _crit(1, "exclusion", "Type 2 diabetes")],
            verdicts=[_v(0, "not_met", ["Condition/a"]), _v(1, "met", ["Condition/a"])],
            dropped="exclusion met: has type 2 diabetes",
        ),
    }
    return [
        Scenario(
            "start_fetch_profile",
            "no profile yet -> fetch_patient_data",
            _base(None, steps=0),
            lambda d: d.action == "fetch_patient_data",
        ),
        Scenario(
            "missing_hba1c_fetch_lab",
            "trials need HbA1c, profile has none -> fetch Observation 4548-4 (or resolve_unknowns with 4548-4)",
            _base(
                no_hba1c,
                trials={k: {**v} for k, v in diabetes_trials.items()},
                searches=[_search("type 2 diabetes", 24, 2)],
                decisions=_hist(
                    ("search_trials", {"conditions": "type 2 diabetes"}),
                    ("parse_criteria", {"trial_ids": list(diabetes_trials)}),
                    ("judge_trials", {"trial_ids": list(diabetes_trials)}),
                ),
            ),
            lambda d: d.action in ("fetch_patient_data", "resolve_unknowns") and "4548-4" in json.dumps(d.args),
        ),
        Scenario(
            "first_search",
            "profile loaded, no search yet -> search_trials on a patient condition",
            _base(profile, decisions=_hist(("fetch_patient_data", {}))),
            lambda d: d.action == "search_trials" and bool(d.args.get("conditions")),
        ),
        Scenario(
            "two_results_broaden",
            "search found only 2 trials -> broaden the search",
            _base(
                profile,
                searches=[_search("type 2 diabetes with stage 3 CKD and insulin pump", 2, 2, phases=["PHASE3"])],
                trials={"NCT92000001": _trial("NCT92000001", "A"), "NCT92000002": _trial("NCT92000002", "B")},
                decisions=_hist(
                    ("fetch_patient_data", {}),
                    ("search_trials", {"conditions": "type 2 diabetes with stage 3 CKD and insulin pump"}),
                ),
            ),
            lambda d: (d.action == "refine_search" and d.args.get("mode") == "broaden") or d.action == "search_trials",
        ),
        Scenario(
            "four_hundred_results_narrow",
            "search reported 400 matches -> narrow the search",
            _base(
                profile,
                searches=[_search("diabetes", 400, 20)],
                trials={f"NCT930000{i:02d}": _trial(f"NCT930000{i:02d}", f"Diabetes trial {i}") for i in range(20)},
                decisions=_hist(("fetch_patient_data", {}), ("search_trials", {"conditions": "diabetes"})),
            ),
            lambda d: (
                (d.action == "refine_search" and d.args.get("mode") == "narrow")
                or (
                    d.action == "search_trials"
                    and bool(d.args.get("phases") or d.args.get("terms") or d.args.get("location"))
                )
            ),
        ),
        Scenario(
            "found_trials_parse",
            "trials found, none parsed -> parse_criteria",
            _base(
                profile,
                searches=[_search("type 2 diabetes", 18, 3)],
                trials={f"NCT9400000{i}": _trial(f"NCT9400000{i}", f"T2D trial {i}") for i in range(3)},
                decisions=_hist(("fetch_patient_data", {}), ("search_trials", {"conditions": "type 2 diabetes"})),
            ),
            lambda d: d.action == "parse_criteria" and len(ids(d, "trial_ids")) >= 1,
        ),
        Scenario(
            "parsed_trials_judge",
            "trials parsed, not judged -> judge_trials",
            _base(
                profile,
                searches=[_search("type 2 diabetes", 18, 2)],
                trials={k: {"summary": v["summary"], "criteria": v["criteria"]} for k, v in diabetes_trials.items()},
                decisions=_hist(
                    ("fetch_patient_data", {}),
                    ("search_trials", {"conditions": "type 2 diabetes"}),
                    ("parse_criteria", {"trial_ids": list(diabetes_trials)}),
                ),
            ),
            lambda d: d.action == "judge_trials" and bool(set(ids(d, "trial_ids")) & set(diabetes_trials)),
        ),
        Scenario(
            "exclusion_met_drop",
            "a trial has 3 exclusion criteria met -> drop_trials with that id",
            _base(
                profile,
                searches=[_search("type 2 diabetes", 18, 2)],
                trials={
                    "NCT95000001": _trial(
                        "NCT95000001",
                        "Insulin-naive T2D study",
                        criteria=[
                            _crit(0, "exclusion", "Current insulin therapy"),
                            _crit(1, "exclusion", "History of pancreatitis"),
                            _crit(2, "exclusion", "Prior bariatric surgery"),
                            _crit(3, "inclusion", "Type 2 diabetes"),
                        ],
                        verdicts=[
                            _v(0, "met", ["MedicationRequest/m"]),
                            _v(1, "met", ["Condition/c"]),
                            _v(2, "met", ["Procedure/p"]),
                            _v(3, "met", ["Condition/d"]),
                        ],
                    ),
                    "NCT95000002": _trial(
                        "NCT95000002",
                        "GLP-1 add-on study",
                        criteria=[_crit(0, "inclusion", "Type 2 diabetes")],
                        verdicts=[_v(0, "met", ["Condition/d"])],
                    ),
                },
                decisions=_hist(("judge_trials", {"trial_ids": ["NCT95000001", "NCT95000002"]})),
            ),
            lambda d: (
                d.action == "drop_trials"
                and "NCT95000001" in ids(d, "trial_ids")
                and "NCT95000002" not in ids(d, "trial_ids")
            ),
        ),
        Scenario(
            "many_unknowns_resolve",
            "promising trial with many unknown verdicts -> resolve_unknowns (or fetch data) for it",
            _base(
                profile,
                searches=[_search("type 2 diabetes", 18, 1)],
                trials={
                    "NCT96000001": _trial(
                        "NCT96000001",
                        "Basal insulin titration study",
                        criteria=[
                            _crit(i, "inclusion", t)
                            for i, t in enumerate(
                                [
                                    "Type 2 diabetes",
                                    "HbA1c 7-10%",
                                    "Fasting glucose > 130 mg/dL",
                                    "C-peptide > 0.8 ng/mL",
                                    "BMI < 40",
                                ]
                            )
                        ],
                        verdicts=[
                            _v(0, "met", ["Condition/d"]),
                            _v(1, "unknown"),
                            _v(2, "unknown"),
                            _v(3, "unknown"),
                            _v(4, "met", ["Observation/b"]),
                        ],
                    )
                },
                decisions=_hist(("judge_trials", {"trial_ids": ["NCT96000001"]})),
            ),
            lambda d: (
                (d.action == "resolve_unknowns" and d.args.get("trial_id") == "NCT96000001")
                or d.action == "fetch_patient_data"
            ),
        ),
        Scenario(
            "persistent_unknown_ask",
            "unknown remains after resolve_unknowns already tried -> ask_clinician",
            _base(
                profile,
                searches=[_search("type 2 diabetes", 18, 1)],
                trials={
                    "NCT97000001": _trial(
                        "NCT97000001",
                        "Pregnancy-excluded insulin study",
                        criteria=[
                            _crit(0, "inclusion", "Type 2 diabetes"),
                            _crit(1, "exclusion", "Plans to become pregnant in the next 12 months"),
                        ],
                        verdicts=[_v(0, "met", ["Condition/d"]), _v(1, "unknown")],
                    )
                },
                decisions=_hist(
                    ("judge_trials", {"trial_ids": ["NCT97000001"]}),
                    (
                        "resolve_unknowns",
                        {"trial_id": "NCT97000001", "criterion_indexes": [1], "what_to_look_for": "pregnancy plans"},
                    ),
                ),
            ),
            lambda d: d.action == "ask_clinician" and bool(d.args.get("question")),
        ),
        Scenario(
            "all_judged_rank",
            "trials judged, no shortlist yet -> rank",
            _base(
                profile,
                searches=[_search("type 2 diabetes", 18, 2)],
                trials=judged_ok,
                decisions=_hist(
                    ("judge_trials", {"trial_ids": list(judged_ok)}),
                    ("drop_trials", {"trial_ids": ["NCT91000002"], "reason": "exclusion met"}),
                ),
            ),
            lambda d: d.action in ("rank", "gather_evidence"),
        ),
        Scenario(
            "stable_shortlist_finish",
            "shortlist ranked twice with the same result -> finish",
            _base(
                profile,
                trials=judged_ok,
                shortlist=[{"nct_id": "NCT91000001", "rank": 1, "explanation": "fits"}],
                decisions=_hist(
                    ("judge_trials", {"trial_ids": list(judged_ok)}),
                    ("rank", {}),
                    ("gather_evidence", {"trial_ids": ["NCT91000001"]}),
                    ("rank", {}),
                ),
            ),
            lambda d: d.action == "finish",
        ),
        Scenario(
            "approved_draft",
            "clinician approved trials, no referrals yet -> draft_referrals",
            _base(
                profile,
                trials=judged_ok,
                shortlist=[{"nct_id": "NCT91000001", "rank": 1, "explanation": "fits"}],
                approved_trial_ids=["NCT91000001"],
                decisions=_hist(("rank", {}), ("finish", {"summary": "awaiting review"})),
            ),
            lambda d: d.action == "draft_referrals",
        ),
        Scenario(
            "not_approved_no_draft",
            "shortlist not approved, draft was just rejected -> must not draft again",
            _base(
                profile,
                trials=judged_ok,
                shortlist=[{"nct_id": "NCT91000001", "rank": 1, "explanation": "fits"}],
                rejection="draft_referrals requires clinician approval of the shortlist",
                decisions=_hist(("rank", {})),
            ),
            lambda d: d.action != "draft_referrals",
        ),
        Scenario(
            "nothing_found_after_broaden",
            "no trials even after broadening -> finish (or ask clinician), not another identical search",
            _base(
                profile,
                searches=[_search("rare metabolic disorder X", 0, 0), _search("metabolic disorder", 0, 0)],
                decisions=_hist(
                    ("search_trials", {"conditions": "rare metabolic disorder X"}),
                    ("refine_search", {"mode": "broaden", "reason": "0 results", "conditions": "metabolic disorder"}),
                ),
                steps=12,
            ),
            lambda d: (
                d.action in ("finish", "ask_clinician")
                or (
                    d.action in ("search_trials", "refine_search")
                    and "metabolic disorder" != str(d.args.get("conditions"))
                )
            ),
        ),
    ]


@dataclass
class ScenarioResult:
    name: str
    expect: str
    passes: int
    runs: int
    actions: list[str]
    phi_violations: list[str]

    @property
    def majority(self) -> bool:
        return self.passes * 2 > self.runs


async def load_profile(fhir_base: str, patient_id: str) -> dict[str, Any]:
    reader = HttpFhirReader(fhir_base)
    try:
        profile = await build_profile(reader, patient_id, as_of=date(2026, 11, 16))
        patient = await reader.read("Patient", patient_id)
    finally:
        await reader.aclose()
    return {"profile": profile.model_dump(mode="json"), "patient": patient}


def run_scenarios(
    llm: StructuredLLM, profile: dict[str, Any], patient: dict[str, Any], runs: int = 3
) -> list[ScenarioResult]:
    known = KnownIdentifiers.from_fhir_patient(patient)
    results = []
    for sc in build_scenarios(profile):
        summary = summarize(sc.state)
        passes, actions, phi = 0, [], []
        for _ in range(runs):
            try:
                d = decide(llm, summary)
            except Exception as exc:  # noqa: BLE001 - a failed call counts as a failed run
                actions.append(f"error:{type(exc).__name__}")
                continue
            try:
                d, _outcome, _ = check(d, sc.state, Limits())
            except GuardRejection:
                pass  # the check is on the supervisor's choice; guards are tested separately
            actions.append(f"{d.action} {json.dumps(d.args, default=str)[:120]}")
            passes += bool(sc.ok(d))
            try:
                check_outbound(d.args, destination="scenario", known=known)
            except PhiBlockedError as err:
                phi.append(f"{d.action}: {', '.join(err.types)}")
        results.append(ScenarioResult(sc.name, sc.expect, passes, runs, actions, phi))
    return results


def write_report(model: str, results: list[ScenarioResult], usage: dict[str, Any]) -> tuple[str, float, int]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    rate = sum(r.majority for r in results) / len(results)
    phi = sum(len(r.phi_violations) for r in results)
    lines = [
        f"# Phase 4 supervisor scenarios ({stamp} UTC)",
        "",
        f"Model: `{model}`. Each scenario run {results[0].runs if results else 0}x without cache; majority counts.",
        f"**Pass rate: {rate:.2f}** (gate ≥ {PASS_RATE_GATE}). PHI violations: {phi}. Usage: `{json.dumps(usage)}`",
        "",
        "| Scenario | Expected | Passed runs | Majority | Decisions |",
        "|---|---|---:|:---:|---|",
    ]
    for r in results:
        acts = "<br>".join(a.replace("|", "/") for a in r.actions)
        lines.append(f"| {r.name} | {r.expect} | {r.passes}/{r.runs} | {'yes' if r.majority else 'NO'} | {acts} |")
    flagged = [(r.name, v) for r in results for v in r.phi_violations]
    if flagged:
        lines += ["", "## PHI violations (identifier types only)", ""]
        lines += [f"- {name}: {v}" for name, v in flagged]
    path = REPORT_DIR / f"phase4-scenarios-{stamp}.md"
    path.write_text("\n".join(lines) + "\n")
    return str(path), rate, phi


_ = asyncio  # used by the management command
