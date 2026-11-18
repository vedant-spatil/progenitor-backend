"""Supervisor summary and scenario fixtures (deterministic; the live suite is evals/phase4.py)."""

import json
from typing import Any

from agents.schemas import Decision
from agents.supervisor import summarize
from evals.phase4 import build_scenarios
from mcp_servers.common.phi import KnownIdentifiers, find_phi

PATIENT = {
    "resourceType": "Patient",
    "id": "pat-77",
    "name": [{"family": "Okafor", "given": ["Adaeze"]}],
    "birthDate": "1961-07-19",
    "address": [{"city": "Somerville", "postalCode": "02143"}],
}
PROFILE: dict[str, Any] = {
    "patient_ref": "Patient/pat-77",
    "age_years": 65,
    "sex": "female",
    "conditions": [
        {
            "resource_id": "Condition/c1",
            "display": "Diabetes mellitus type 2",
            "code": "44054006",
            "system": "http://snomed.info/sct",
            "clinical_status": "active",
        }
    ],
    "medications": [],
    "labs": [
        {
            "resource_id": "Observation/o1",
            "display": "Hemoglobin A1c",
            "code": "4548-4",
            "system": "http://loinc.org",
            "value": 8.2,
            "unit": "%",
            "effective": "2026-06-01",
            "category": "laboratory",
        }
    ],
    "truncated": False,
}

CORRECT: dict[str, dict[str, Any]] = {
    "start_fetch_profile": {"action": "fetch_patient_data"},
    "missing_hba1c_fetch_lab": {"action": "fetch_patient_data", "args": {"loinc_codes": ["4548-4"]}},
    "first_search": {"action": "search_trials", "args": {"conditions": "type 2 diabetes"}},
    "two_results_broaden": {"action": "refine_search", "args": {"mode": "broaden", "reason": "few"}},
    "four_hundred_results_narrow": {"action": "refine_search", "args": {"mode": "narrow", "reason": "many"}},
    "found_trials_parse": {"action": "parse_criteria", "args": {"trial_ids": ["NCT94000000"]}},
    "parsed_trials_judge": {"action": "judge_trials", "args": {"trial_ids": ["NCT90000001"]}},
    "exclusion_met_drop": {"action": "drop_trials", "args": {"trial_ids": ["NCT95000001"], "reason": "x"}},
    "many_unknowns_resolve": {
        "action": "resolve_unknowns",
        "args": {"trial_id": "NCT96000001", "criterion_indexes": [1]},
    },
    "persistent_unknown_ask": {"action": "ask_clinician", "args": {"question": "Pregnancy plans?"}},
    "all_judged_rank": {"action": "rank"},
    "stable_shortlist_finish": {"action": "finish", "args": {"summary": "done"}},
    "approved_draft": {"action": "draft_referrals"},
    "not_approved_no_draft": {"action": "finish", "args": {"summary": "awaiting approval"}},
    "nothing_found_after_broaden": {"action": "finish", "args": {"summary": "no trials"}},
}


def test_summary_has_no_patient_identifiers() -> None:
    state = {"profile": PROFILE, "trials": {}, "decisions": [], "steps": 1}
    text = summarize(state)
    known = KnownIdentifiers.from_fhir_patient(PATIENT)
    # FHIR refs ("other_unique") and service dates are needed for citations and time windows; the summary
    # goes to the LLM provider (synthetic data in dev; BAA-covered Bedrock in Phase 8), never to public APIs.
    # Direct identifiers (name, address, birth date, record numbers) must not appear.
    found = find_phi(text, known) - {"other_unique", "date"}
    assert found == set()
    assert "Okafor" not in text
    assert "1961-07-19" not in text


def test_summary_carries_ids_for_citation() -> None:
    data = json.loads(summarize({"profile": PROFILE, "trials": {}, "decisions": []}))
    assert "Observation/o1" in json.dumps(data["patient"]["labs"])
    assert data["patient"]["active_conditions"][0].endswith("[Condition/c1]")


def test_every_scenario_accepts_a_correct_decision() -> None:
    scenarios = build_scenarios(PROFILE)
    assert len(scenarios) == 15
    assert {s.name for s in scenarios} == set(CORRECT)
    for sc in scenarios:
        d = Decision.model_validate({"rationale": "t", "confidence": 1.0, "args": {}, **CORRECT[sc.name]})
        assert sc.ok(d), sc.name
        assert (
            not sc.ok(Decision.model_validate({"action": "draft_referrals", "rationale": "t", "confidence": 1.0}))
            or sc.name == "approved_draft"
        ), sc.name
