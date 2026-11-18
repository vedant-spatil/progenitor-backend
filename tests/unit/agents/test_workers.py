"""LLM workers with the scripted fake LLM: shapes, guards, and post-processing."""

from datetime import date
from typing import Any

import pytest

from agents.guards import enforce_citations
from agents.schemas import (
    Criterion,
    CriterionKind,
    LabFact,
    PatientProfile,
    SearchSpec,
    TrialAssessment,
    TrialSummary,
    Verdict,
    VerdictStatus,
)
from agents.workers.llm_workers import (
    NotApprovedError,
    draft_referral,
    gather_evidence,
    judge_trial,
    parse_criteria,
    rank_trials,
    verdict_counts,
)
from agents.workers.scout import search_trials
from tests.fakes.llm import ScriptedLLM

PROFILE = PatientProfile(
    patient_ref="Patient/p1",
    age_years=58,
    sex="female",
    labs=[LabFact(resource_id="Observation/o1", code="4548-4", display="HbA1c", value=8.2, unit="%")],
)
CRITERIA = [
    Criterion(index=0, kind=CriterionKind.INCLUSION, source_text="HbA1c 7-10%", entity="HbA1c"),
    Criterion(index=1, kind=CriterionKind.INCLUSION, source_text="Age 18-75"),
    Criterion(index=2, kind=CriterionKind.EXCLUSION, source_text="Pregnancy"),
]


class TestCitationGuard:
    def test_met_without_valid_ref_becomes_unknown(self) -> None:
        out = enforce_citations(
            [Verdict(criterion_index=0, status=VerdictStatus.MET, evidence_refs=["Observation/made-up"], reason="r")],
            {"Observation/o1"},
        )
        assert out[0].status == "unknown"
        assert out[0].downgraded
        assert out[0].evidence_refs == []

    def test_valid_refs_kept_invalid_dropped(self) -> None:
        out = enforce_citations(
            [
                Verdict(
                    criterion_index=0, status=VerdictStatus.NOT_MET, evidence_refs=["Observation/o1", "Observation/x"]
                )
            ],
            {"Observation/o1"},
        )
        assert (out[0].status, out[0].evidence_refs, out[0].downgraded) == ("not_met", ["Observation/o1"], False)

    def test_unknown_untouched(self) -> None:
        out = enforce_citations([Verdict(criterion_index=0, status=VerdictStatus.UNKNOWN)], set())
        assert out[0].status == "unknown"
        assert not out[0].downgraded


async def test_parser_reindexes_and_attaches_codes() -> None:
    llm = ScriptedLLM(
        {
            "parse_criteria": [
                {
                    "criteria": [
                        {
                            "index": 7,
                            "kind": "inclusion",
                            "source_text": "HbA1c 7-10%",
                            "entities": [{"text": "HbA1c", "type": "lab"}],
                        },
                        {
                            "index": 7,
                            "kind": "exclusion",
                            "source_text": "insulin use",
                            "entities": [{"text": "insulin", "type": "drug"}, {"text": "HbA1c", "type": "lab"}],
                        },
                    ]
                }
            ]
        }
    )
    calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append((name, args))
        if name == "terminology_lookup_loinc":
            return {"preferred": {"code": "4548-4"}, "matches": []}
        return {"candidates": [{"rxcui": "5856"}]}

    crits = await parse_criteria(llm, "text", call_tool=call_tool)
    assert [c.index for c in crits] == [0, 1]
    assert crits[0].entities[0].codes == [{"system": "loinc", "code": "4548-4"}]
    assert crits[1].entities[0].codes == [{"system": "rxnorm", "code": "5856"}]
    assert len(calls) == 2  # HbA1c looked up once


def test_judge_fills_missing_and_enforces_citations() -> None:
    llm = ScriptedLLM(
        {
            "judge_trial": [
                {
                    "verdicts": [
                        {"criterion_index": 0, "status": "met", "evidence_refs": ["L1"], "reason": "8.2%"},
                        {"criterion_index": 1, "status": "met", "evidence_refs": ["Patient/other"], "reason": "58"},
                    ]
                }
            ]
        }
    )
    verdicts = judge_trial(llm, PROFILE, CRITERIA, as_of=date(2026, 11, 1))
    assert [(v.criterion_index, v.status) for v in verdicts] == [(0, "met"), (1, "unknown"), (2, "unknown")]
    assert verdicts[1].downgraded
    prompt = llm.calls[0].user
    assert "L1 | HbA1c [loinc 4548-4] = 8.2 %" in prompt
    assert "Observation/o1" not in prompt  # short ids only; mapped back after the call
    assert "[2] (exclusion) Pregnancy" in prompt


async def test_evidence_keeps_only_returned_pmids() -> None:
    llm = ScriptedLLM(
        {"gather_evidence": [{"papers": [{"pmid": "1", "relevance": "r"}, {"pmid": "999", "relevance": "invented"}]}]}
    )

    async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        assert name == "pubmed_search_pubmed"
        return {"articles": [{"pmid": "1", "title": "t", "year": "2020"}]}

    papers = await gather_evidence(
        llm, TrialSummary(nct_id="NCT00000001", conditions=["Diabetes"]), ["metformin"], call_tool
    )
    assert [p.pmid for p in papers] == ["1"]


def _assessment(nct: str, statuses: list[str]) -> TrialAssessment:
    return TrialAssessment(
        trial=TrialSummary(nct_id=nct),
        criteria=CRITERIA,
        verdicts=[Verdict(criterion_index=i, status=VerdictStatus(s)) for i, s in enumerate(statuses)],
    )


def test_verdict_counts() -> None:
    assert verdict_counts(_assessment("NCT1", ["met", "not_met", "met"])) == {
        "inclusion_met": 1,
        "inclusion_not_met": 1,
        "exclusion_met": 1,
        "unknown": 0,
    }


def test_ranker_dedupes_and_appends_missing() -> None:
    a, b, c = (_assessment(n, ["met", "met", "not_met"]) for n in ("NCT1", "NCT2", "NCT3"))
    llm = ScriptedLLM(
        {
            "rank_trials": [
                {
                    "ranked": [
                        {"nct_id": "NCT2", "rank": 1, "explanation": "x"},
                        {"nct_id": "NCT2", "rank": 2, "explanation": "dup"},
                        {"nct_id": "NCT9", "rank": 3, "explanation": "unknown trial"},
                        {"nct_id": "NCT1", "rank": 4, "explanation": "y"},
                    ]
                }
            ]
        }
    )
    ranking = rank_trials(llm, [a, b, c])
    assert [(r.nct_id, r.rank) for r in ranking.ranked] == [("NCT2", 1), ("NCT1", 2), ("NCT3", 3)]


def test_drafter_requires_approval() -> None:
    with pytest.raises(NotApprovedError):
        draft_referral(ScriptedLLM(), _assessment("NCT1", ["met"] * 3), PROFILE, approved=False)


def test_drafter_forces_nct_id() -> None:
    llm = ScriptedLLM({"draft_referral": [{"nct_id": "NCT-wrong", "body": "Dear team"}]})
    note = draft_referral(llm, _assessment("NCT1", ["met"] * 3), PROFILE, approved=True)
    assert note.nct_id == "NCT1"
    assert "age 58" in llm.calls[0].user


async def test_scout_pages_until_max_results() -> None:
    pages = iter(
        [
            {"studies": [{"nct_id": f"NCT0000000{i}"} for i in range(3)], "total": 50, "next_page_token": "t2"},
            {"studies": [{"nct_id": f"NCT0000001{i}"} for i in range(3)], "total": 50, "next_page_token": "t3"},
        ]
    )
    seen: list[dict[str, Any]] = []

    async def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        seen.append(args)
        return next(pages)

    trials, total = await search_trials(SearchSpec(conditions="asthma", phases=["PHASE3"], max_results=5), call_tool)
    assert (len(trials), total) == (5, 50)
    assert seen[1]["page_token"] == "t2"
    assert seen[0]["phases"] == ["PHASE3"]


def test_profile_context_fits_budget_dropping_old_facts_first() -> None:
    from agents.schemas import MedicationFact
    from agents.workers.common import profile_context

    labs = [
        LabFact(
            resource_id=f"Observation/{i}", code=f"c{i}", display=f"lab {i}", value=1.0, effective=date(2020, 1, 1 + i)
        )
        for i in range(25)
    ]
    meds = [
        MedicationFact(
            resource_id="MedicationRequest/old",
            code="1",
            display="old drug",
            status="stopped",
            authored_on=date(2010, 1, 1),
        )
    ]
    profile = PatientProfile(patient_ref="Patient/p1", age_years=50, sex="male", labs=labs, medications=meds)
    full, _ = profile_context(profile, date(2026, 1, 1), max_chars=100_000)
    small, aliases = profile_context(profile, date(2026, 1, 1), max_chars=len(full) - 200)
    assert len(small) <= len(full) - 200
    assert "old drug" not in small  # stopped meds go first
    assert "lab 24" in small  # newest labs kept
    assert "older facts omitted" in small
    assert "P | age 50" in small
    assert aliases["P"] == "Patient/p1"


class TestRelevanceGuard:
    def _profile(self) -> PatientProfile:
        from agents.schemas import ConditionFact

        return PatientProfile(
            patient_ref="Patient/p1",
            conditions=[
                ConditionFact(resource_id="Condition/sinus", code="1", display="Acute sinusitis"),
                ConditionFact(
                    resource_id="Condition/asthma-old", code="2", display="Asthma", clinical_status="resolved"
                ),
            ],
        )

    def test_not_met_citing_unrelated_condition_becomes_unknown(self) -> None:
        from agents.guards import enforce_relevance

        crit = [Criterion(index=0, kind=CriterionKind.INCLUSION, source_text="Active diagnosis of asthma")]
        v = [Verdict(criterion_index=0, status=VerdictStatus.NOT_MET, evidence_refs=["Condition/sinus"])]
        out = enforce_relevance(v, crit, self._profile())
        assert out[0].status == VerdictStatus.UNKNOWN
        assert out[0].downgraded

    def test_not_met_citing_the_condition_itself_is_kept(self) -> None:
        from agents.guards import enforce_relevance

        crit = [Criterion(index=0, kind=CriterionKind.INCLUSION, source_text="Active diagnosis of asthma")]
        v = [Verdict(criterion_index=0, status=VerdictStatus.NOT_MET, evidence_refs=["Condition/asthma-old"])]
        assert enforce_relevance(v, crit, self._profile())[0].status == VerdictStatus.NOT_MET

    def test_not_applicable_needs_evidence(self) -> None:
        out = enforce_citations([Verdict(criterion_index=0, status=VerdictStatus.NOT_APPLICABLE)], {"Patient/p1"})
        assert out[0].status == VerdictStatus.UNKNOWN


def test_focus_terms_keep_relevant_lab_when_trimming() -> None:
    from agents.workers.common import focus_terms, profile_context

    labs = [
        LabFact(
            resource_id=f"Observation/{i}",
            code=f"c{i}",
            display=f"filler lab {i}",
            value=1.0,
            effective=date(2026, 1, 1),
        )
        for i in range(30)
    ]
    labs.append(
        LabFact(
            resource_id="Observation/plt",
            code="777-3",
            display="Platelets [#/volume] in Blood",
            value=331.0,
            effective=date(2015, 1, 1),
        )
    )
    profile = PatientProfile(patient_ref="Patient/p1", labs=labs)
    focus = focus_terms(["Platelet count of at least 100 x10^3/uL"])
    text, _ = profile_context(profile, date(2026, 11, 1), max_chars=600, focus=focus)
    assert "Platelets" in text  # oldest line, but relevant: kept
    assert "older facts omitted" in text
