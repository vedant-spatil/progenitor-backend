"""Gate 4: graph mechanics and guards with a scripted supervisor (no live LLM)."""

from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents.graph import Deps, build_graph, initial_state
from agents.guards import GuardRejection, Limits, check
from agents.ledger import MemoryLedger
from agents.schemas import Decision
from tests.fakes.llm import ScriptedLLM

PATIENT = {"resourceType": "Patient", "id": "p1", "birthDate": "1970-01-01", "gender": "female"}
HBA1C = {
    "resourceType": "Observation",
    "id": "o1",
    "category": [{"coding": [{"code": "laboratory"}]}],
    "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}]},
    "valueQuantity": {"value": 8.2, "unit": "%"},
    "effectiveDateTime": "2026-06-01",
}


class FakeReader:
    async def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        return PATIENT

    async def search(
        self, resource_type: str, params: dict[str, str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if resource_type == "Observation" and params.get("category") in (None, "laboratory"):
            return [HBA1C], []
        return [], []


class FakeTools:
    def __init__(self, n_trials: int = 2) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.n = n_trials

    async def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append((name, args))
        if name == "clinicaltrials_search_trials":
            studies = [
                {"nct_id": f"NCT0000000{i}", "title": f"Trial {i}", "status": "RECRUITING"} for i in range(self.n)
            ]
            return {"studies": studies, "total": self.n, "next_page_token": None}
        if name == "clinicaltrials_get_eligibility":
            return {"nct_id": args["nct_id"], "criteria_text": "Inclusion Criteria:\n- HbA1c 7-10%"}
        if name == "pubmed_search_pubmed":
            return {"articles": [{"pmid": "1", "title": "t", "year": "2020"}]}
        raise AssertionError(name)


def dec(action: str, **args: Any) -> dict[str, Any]:
    return {"action": action, "args": args, "rationale": "test", "confidence": 1.0}


CRITERIA = {"criteria": [{"index": 0, "kind": "inclusion", "source_text": "HbA1c 7-10%", "entity": "HbA1c"}]}
VERDICT = {"verdicts": [{"criterion_index": 0, "status": "met", "evidence_refs": ["L1"], "reason": "8.2%"}]}


def make(
    replies: dict[str, list[dict[str, Any]]], limits: Limits | None = None, n_trials: int = 2
) -> tuple[Any, MemoryLedger, FakeTools]:
    ledger = MemoryLedger()
    tools = FakeTools(n_trials)
    deps = Deps(
        llm=ScriptedLLM(replies), call_tool=tools, reader=FakeReader(), ledger=ledger, limits=limits or Limits()
    )
    return build_graph(deps, checkpointer=InMemorySaver()), ledger, tools


def cfg(thread: str = "t1") -> dict[str, Any]:
    return {"configurable": {"thread_id": thread}, "recursion_limit": 100}


async def test_happy_path_routes_and_fans_out() -> None:
    a, b = "NCT00000000", "NCT00000001"
    graph, ledger, tools = make(
        {
            "supervisor": [
                dec("fetch_patient_data"),
                dec("search_trials", conditions="type 2 diabetes"),
                dec("parse_criteria", trial_ids=[a, b]),
                dec("judge_trials", trial_ids=[a, b]),
                dec("rank"),
                dec("finish", summary="done"),
            ],
            "parse_criteria": [CRITERIA, CRITERIA],
            "judge_trial": [VERDICT, VERDICT],
            "rank_trials": [
                {"ranked": [{"nct_id": b, "rank": 1, "explanation": "x"}, {"nct_id": a, "rank": 2, "explanation": "y"}]}
            ],
        }
    )
    out = await graph.ainvoke(initial_state("r1", "p1"), cfg())
    assert out["status"] == "finished"
    assert out["profile"]["age_years"] is not None
    assert set(out["trials"]) == {a, b}
    # fan-out: one supervisor decision produced verdicts for both trials
    assert all(t["verdicts"][0]["status"] == "met" for t in out["trials"].values())
    assert out["trials"][a]["verdicts"][0]["evidence_refs"] == ["Observation/o1"]
    assert [r["nct_id"] for r in out["shortlist"]] == [b, a]
    assert [e.action for e in ledger.entries] == [
        "fetch_patient_data",
        "search_trials",
        "parse_criteria",
        "judge_trials",
        "rank",
        "finish",
    ]
    assert all(e.guard_outcome == "accepted" for e in ledger.entries)
    assert ("clinicaltrials_search_trials", {"conditions": "type 2 diabetes", "page_size": 20}) in tools.calls


async def test_drop_evidence_refine_and_resolve_route() -> None:
    a = "NCT00000000"
    graph, _ledger, _tools = make(
        {
            "supervisor": [
                dec("fetch_patient_data"),
                dec("search_trials", conditions="asthma", phases=["PHASE3"]),
                dec("refine_search", mode="broaden", reason="few results"),
                dec("parse_criteria", trial_ids=[a]),
                dec("judge_trials", trial_ids=[a]),
                dec(
                    "resolve_unknowns",
                    trial_id=a,
                    criterion_indexes=[0],
                    what_to_look_for="HbA1c",
                    loinc_codes=["4548-4"],
                ),
                dec("gather_evidence", trial_ids=[a]),
                dec("drop_trials", trial_ids=["NCT00000001"], reason="exclusion met"),
                dec("finish", summary="done"),
            ],
            "parse_criteria": [CRITERIA],
            "judge_trial": [VERDICT, VERDICT],
            "gather_evidence": [{"papers": [{"pmid": "1", "relevance": "r"}]}],
        }
    )
    out = await graph.ainvoke(initial_state("r2", "p1"), cfg("t2"))
    assert out["trials"]["NCT00000001"]["dropped"] == "exclusion met"
    assert out["trials"][a]["papers"][0]["pmid"] == "1"
    refine_spec = out["searches"][1]["spec"]
    assert "phases" not in refine_spec  # broaden drops the phase filter
    assert out["status"] == "finished"


async def test_ask_clinician_interrupts_and_resumes() -> None:
    graph, _ledger, _ = make(
        {
            "supervisor": [
                dec("fetch_patient_data"),
                dec("ask_clinician", question="Is the patient pregnant?"),
                dec("finish", summary="done"),
            ]
        }
    )
    first = await graph.ainvoke(initial_state("r3", "p1"), cfg("t3"))
    assert "__interrupt__" in first
    assert first["__interrupt__"][0].value["question"] == "Is the patient pregnant?"
    out = await graph.ainvoke(Command(resume={"answer": "No", "actor_user_id": 7}), cfg("t3"))
    assert out["clinician_qa"] == [
        {"question": "Is the patient pregnant?", "answer": {"answer": "No", "actor_user_id": 7}}
    ]
    assert out["status"] == "finished"


async def test_draft_referrals_blocked_without_approval_then_reasked() -> None:
    graph, ledger, _ = make({"supervisor": [dec("draft_referrals"), dec("finish", summary="nothing approved")]})
    out = await graph.ainvoke(initial_state("r4", "p1"), cfg("t4"))
    assert [(e.action, e.guard_outcome) for e in ledger.entries] == [
        ("draft_referrals", "rejected"),
        ("finish", "accepted"),
    ]
    assert "approval" in ledger.entries[0].guard_detail
    assert out.get("referrals") == []


async def test_repeated_rejection_forces_finish() -> None:
    graph, ledger, _ = make({"supervisor": [dec("draft_referrals"), dec("draft_referrals")]})
    out = await graph.ainvoke(initial_state("r5", "p1"), cfg("t5"))
    assert out["finish_reason"] == "forced_finish"
    assert ledger.entries[-1].action == "finish"


async def test_step_budget_forces_finish() -> None:
    graph, ledger, _ = make({"supervisor": [dec("fetch_patient_data")]}, limits=Limits(max_steps=40))
    state = {**initial_state("r6", "p1"), "steps": 40}
    out = await graph.ainvoke(state, cfg("t6"))
    assert out["finish_reason"] == "forced_finish"
    assert ledger.entries[0].guard_detail == "steps"


async def test_loop_guard_stops_third_identical_decision() -> None:
    graph, ledger, _ = make({"supervisor": [dec("fetch_patient_data")] * 3})
    out = await graph.ainvoke(initial_state("r7", "p1"), cfg("t7"))
    assert out["finish_reason"] == "loop_stopped"
    assert [e.guard_outcome for e in ledger.entries] == ["accepted", "accepted", "loop_stopped"]


async def test_llm_failure_is_orchestration_error() -> None:
    _graph, ledger, _ = make({"supervisor": []})  # ScriptedLLM raises: no reply scripted

    class Boom:
        model = "x"

        def __init__(self) -> None:
            from agents.llm import Usage

            self.usage = Usage()

        def complete(self, **kw: Any) -> Any:
            from agents.llm import LlmError

            raise LlmError("supervisor: reply failed schema validation twice")

    deps = Deps(llm=Boom(), call_tool=FakeTools(), reader=FakeReader(), ledger=ledger)
    out = await build_graph(deps, InMemorySaver()).ainvoke(initial_state("r8", "p1"), cfg("t8"))
    assert out["finish_reason"] == "orchestration_error"


class TestGuardsDirect:
    state: dict[str, Any] = {
        "steps": 3,
        "trials": {"NCT1": {"summary": {}}, "NCT2": {"summary": {}, "criteria": [{"index": 0}]}},
        "decisions": [],
    }

    def test_invalid_action_rejected_by_schema(self) -> None:
        with pytest.raises(ValueError, match="action"):
            Decision.model_validate(dec("delete_patient"))

    def test_unknown_trial_ids_rejected(self) -> None:
        with pytest.raises(GuardRejection, match="unknown trial ids"):
            check(Decision.model_validate(dec("parse_criteria", trial_ids=["NCT9"])), self.state)

    def test_judge_before_parse_rejected(self) -> None:
        with pytest.raises(GuardRejection, match="not parsed"):
            check(Decision.model_validate(dec("judge_trials", trial_ids=["NCT1"])), self.state)

    def test_missing_args_rejected(self) -> None:
        with pytest.raises(GuardRejection, match="needs args"):
            check(Decision.model_validate(dec("search_trials")), self.state)

    def test_judge_budget_truncates(self) -> None:
        trials = {f"N{i}": {"criteria": [{"index": 0}], **({"verdicts": []} if i < 29 else {})} for i in range(32)}
        d, outcome, _ = check(
            Decision.model_validate(dec("judge_trials", trial_ids=["N29", "N30", "N31"])),
            {"trials": trials, "decisions": []},
            Limits(max_trials_judged=30),
        )
        assert d.args["trial_ids"] == ["N29"]
        assert outcome == "accepted"

    def test_41st_step_forced_finish(self) -> None:
        d, outcome, _ = check(Decision.model_validate(dec("rank")), {"steps": 40}, Limits(max_steps=40))
        assert (d.action, outcome) == ("finish", "forced_finish")


class TestTracingModes:
    def test_off_and_cloudwatch_and_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agents.tracing import LogTracer, tracer_from_env

        monkeypatch.setenv("TRACING_MODE", "off")
        assert tracer_from_env() is None
        monkeypatch.setenv("TRACING_MODE", "cloudwatch")
        assert isinstance(tracer_from_env(), LogTracer)
        monkeypatch.setenv("TRACING_MODE", "nope")
        with pytest.raises(ValueError, match="TRACING_MODE"):
            tracer_from_env()

    def test_langfuse_without_keys_is_off(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from agents.tracing import tracer_from_env

        monkeypatch.setenv("TRACING_MODE", "langfuse")
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        assert tracer_from_env() is None

    def test_log_tracer_emits_no_rationale(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from agents.ledger import LedgerEntry
        from agents.tracing import LogTracer

        entry = LedgerEntry(
            run_id="r",
            step=1,
            action="search_trials",
            args={"conditions": "asthma"},
            rationale="HbA1c 8.2% on record",
            state_refs=["Observation/1"],
            confidence=0.9,
            guard_outcome="accepted",
        )
        with caplog.at_level(logging.INFO, logger="progenitor.telemetry"):
            LogTracer().decision(entry)
        assert '"event":"supervisor_decision"' in caplog.text
        assert '"action":"search_trials"' in caplog.text
        assert "8.2" not in caplog.text  # rationale and record values are not logged


def test_search_spec_accepts_lists() -> None:
    from agents.schemas import SearchSpec

    spec = SearchSpec.model_validate(
        {"conditions": ["type 2 diabetes", "CKD"], "terms": ["metformin"], "phases": "PHASE3", "location": ["India"]}
    )
    assert spec.conditions == "type 2 diabetes OR CKD"
    assert spec.terms == "metformin"
    assert spec.phases == ["PHASE3"]
    assert spec.location == "India"


async def test_bad_search_args_do_not_crash_the_run() -> None:
    graph, ledger, _ = make(
        {
            "supervisor": [
                dec("fetch_patient_data"),
                dec("search_trials", conditions="asthma", max_results="many"),
                dec("finish", summary="done"),
            ]
        }
    )
    out = await graph.ainvoke(initial_state("r-bad", "p1"), cfg("t-bad"))
    assert out["status"] == "finished"
    assert out["searches"][0]["error"].startswith("invalid args")
    assert [e.action for e in ledger.entries] == ["fetch_patient_data", "search_trials", "finish"]
