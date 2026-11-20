"""Gate 6: metric functions on toy rankings, text patients, pools, BM25, eval cache, reproducibility."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from agents.guards import Limits
from agents.schemas import ConditionFact, PatientProfile, Ranking
from agents.supervisor import summarize
from agents.workers.common import profile_context
from evals.phase6 import bm25, cohorts, metrics, runner
from evals.phase6.cache import DictStore, EvalLLM, ReplayMiss
from evals.phase6.cohorts import EvalPatient
from evals.phase6.text_patient import age_and_sex, profile_from_text
from tests.fakes.llm import ScriptedLLM

SIGIR = (
    "A 58-year-old African-American woman presents to the ER with episodic pressing/burning anterior chest pain. "
    "She is known to have hypertension and obesity. She currently takes no medications."
)

# ---------------- metrics (known answers) ----------------

QRELS = {"A": 2, "B": 1, "C": 0, "D": 2}


def test_ndcg_perfect_and_worst() -> None:
    assert metrics.ndcg_at_k(["A", "D", "B", "C"], QRELS, 4) == pytest.approx(1.0)
    worst = metrics.ndcg_at_k(["C", "B", "D", "A"], QRELS, 4)
    # DCG = 0/1 + 1/log2(3) + 3/log2(4) + 3/log2(5); IDCG = 3 + 3/log2(3) + 1/log2(4) + 0
    expected = (1 / math.log2(3) + 3 / 2 + 3 / math.log2(5)) / (3 + 3 / math.log2(3) + 1 / 2)
    assert worst == pytest.approx(expected)


def test_ndcg_cutoff_and_unjudged_trials() -> None:
    # Only position 1 counts at k=1; an unjudged trial is rel 0.
    assert metrics.ndcg_at_k(["X", "A"], QRELS, 1) == 0.0
    assert metrics.ndcg_at_k(["A"], QRELS, 1) == pytest.approx(1.0)
    assert metrics.ndcg_at_k(["A"], {"A": 0}, 10) == 0.0  # no relevant trial: defined as 0


def test_precision_and_mrr() -> None:
    assert metrics.precision_at_k(["B", "A", "C", "D"], QRELS, 2) == 0.5
    assert metrics.reciprocal_rank(["B", "C", "D"], QRELS) == pytest.approx(1 / 3)
    assert metrics.reciprocal_rank(["B", "C"], QRELS) == 0.0
    assert metrics.mean([1.0, 0.0]) == 0.5


def test_verdict_metrics() -> None:
    verdicts = [
        {"status": "met", "evidence_refs": ["Note/t#1"]},
        {"status": "not_met", "evidence_refs": ["Observation/x"]},
        {"status": "unknown", "evidence_refs": []},
        {"status": "not_applicable", "evidence_refs": []},
    ]
    assert metrics.unknown_rate(verdicts) == 0.25
    assert metrics.citation_validity(verdicts, {"Note/t#1"}) == 0.5
    assert metrics.citation_validity([], set()) == 1.0
    assert metrics.criterion_accuracy([("met", "met"), ("unknown", "met")]) == 0.5


def test_cost() -> None:
    assert metrics.cost_usd(1_000_000, 500_000, {"input": 2.0, "output": 10.0}) == pytest.approx(7.0)
    assert metrics.cost_usd(10, 10, {"input": None, "output": None}) is None
    assert metrics.cost_usd(10, 10, None) is None


def test_price_lookup_ignores_bedrock_region_prefix() -> None:
    assert runner.price_for("global.anthropic.claude-sonnet-5-5") == runner.price_for("anthropic.claude-sonnet-5-5")
    assert runner.price_for("unknown-model") is None


# ---------------- text patients ----------------


def test_text_patient_profile() -> None:
    assert age_and_sex(SIGIR) == (58, "female")
    assert age_and_sex("A 6-month-old boy with fever.") == (0, "male")
    p = profile_from_text("sigir-1", SIGIR)
    assert [n.resource_id for n in p.notes] == ["Note/sigir-1#1", "Note/sigir-1#2", "Note/sigir-1#3"]
    assert "Note/sigir-1#2" in p.resource_ids()
    text, aliases = profile_context(p)
    assert "N2 | She is known to have hypertension and obesity." in text
    assert aliases["N2"] == "Note/sigir-1#2"


def test_fhir_profiles_render_unchanged() -> None:
    """The notes field is empty for FHIR patients: prompts and the supervisor summary are as before."""
    p = PatientProfile(
        patient_ref="Patient/p1",
        age_years=60,
        sex="male",
        conditions=[ConditionFact(resource_id="Condition/c1", display="Diabetes", code="44054006")],
    )
    text, _ = profile_context(p)
    assert "N1" not in text
    assert "notes" not in json.loads(summarize({"profile": p.model_dump(mode="json")}))["patient"]
    assert (
        "notes" in json.loads(summarize({"profile": profile_from_text("t", SIGIR).model_dump(mode="json")}))["patient"]
    )


# ---------------- cohorts and pools ----------------


def test_pool_is_stratified_and_deterministic() -> None:
    qrels = {f"E{i}": 2 for i in range(3)} | {f"X{i}": 1 for i in range(10)} | {f"N{i}": 0 for i in range(10)}
    p = EvalPatient("sigir", "t1", "", qrels)
    pool = cohorts.build_pool(p, 9, seed=7)
    assert pool == cohorts.build_pool(p, 9, seed=7)
    assert len(pool) == 9
    assert sum(qrels[n] == 2 for n in pool) == 3  # every eligible trial fits: 3 of 9
    assert {qrels[n] for n in pool} == {0, 1, 2}


def test_load_and_sample_cohort(tmp_path: Path) -> None:
    base = tmp_path / "sigir" / "qrels"
    base.mkdir(parents=True)
    (tmp_path / "sigir" / "queries.jsonl").write_text(
        "\n".join(json.dumps({"_id": f"t{i}", "text": SIGIR}) for i in range(5))
    )
    rows = ["query-id\tcorpus-id\tscore"] + [f"t{i}\tNCT0000000{i}\t{2 if i < 3 else 0}" for i in range(5)]
    (base / "test.tsv").write_text("\n".join(rows))
    patients = cohorts.load_cohort("sigir", tmp_path)
    assert [p.topic_id for p in patients] == ["t0", "t1", "t2", "t3", "t4"]
    picked = cohorts.sample(patients, 2, seed=1)
    assert picked == cohorts.sample(patients, 2, seed=1)
    assert all(2 in p.qrels.values() for p in picked)  # only patients with an eligible trial


def test_bm25_ranks_matching_trial_first() -> None:
    docs = {"NCT1": "asthma inhaled steroids children", "NCT2": "hypertension obesity chest pain adults"}
    assert bm25.rank("woman with hypertension, obesity and chest pain", docs) == ["NCT2", "NCT1"]
    assert bm25.rank("unrelated words", {"b": "x", "a": "y"}) == ["a", "b"]  # ties by id


# ---------------- eval cache ----------------


def test_eval_cache_counts_virtual_and_real_tokens() -> None:
    inner = ScriptedLLM({"rank_trials": [{"ranked": []}]})
    real = EvalLLM(inner, DictStore())
    store = real.store

    def bump(**_: Any) -> None:
        inner.usage.tokens_in += 100
        inner.usage.tokens_out += 10

    orig = inner.complete

    def complete(**kw: Any) -> Any:
        bump()
        return orig(**kw)

    inner.complete = complete  # type: ignore[method-assign]
    for _ in range(3):
        real.complete(task="rank_trials", system="s", user="u", schema=Ranking)
    assert (real.spent.calls, real.spent.tokens_in) == (1, 100)  # bought once
    assert (real.usage.calls, real.usage.tokens_in, real.usage.cache_hits) == (3, 300, 2)  # as if run 3 times

    replay = EvalLLM(ScriptedLLM({}), store, replay=True)
    replay.complete(task="rank_trials", system="s", user="u", schema=Ranking)
    with pytest.raises(ReplayMiss):
        replay.complete(task="rank_trials", system="s", user="other", schema=Ranking)


def test_reply_failures_are_recorded_and_replayed() -> None:
    """A deterministic bad reply is cached as a failure: replay reproduces it, a live run tries again."""
    from agents.llm import ReplyError

    class Failing:
        model = ScriptedLLM({}).model  # same cache key space as the replay LLM below

        def __init__(self) -> None:
            self.usage = ScriptedLLM({}).usage
            self.calls = 0

        def complete(self, **_: Any) -> Any:
            self.calls += 1
            self.usage.tokens_out += 32768
            raise ReplyError("parse_criteria: reply failed schema validation twice (cut off)")

    store = DictStore()
    inner = Failing()
    live = EvalLLM(inner, store)
    with pytest.raises(ReplyError):
        live.complete(task="parse_criteria", system="s", user="u", schema=Ranking)
    assert (live.spent.calls, live.spent.tokens_out) == (1, 32768)  # failed calls cost tokens too

    replay = EvalLLM(ScriptedLLM({}), store, replay=True)
    with pytest.raises(ReplyError, match="cut off"):  # not ReplayMiss
        replay.complete(task="parse_criteria", system="s", user="u", schema=Ranking)
    assert replay.usage.tokens_out == 32768

    again = EvalLLM(inner, store)
    with pytest.raises(ReplyError):
        again.complete(task="parse_criteria", system="s", user="u", schema=Ranking)
    assert inner.calls == 2  # live runs retry recorded failures


# ---------------- end to end: three systems, then a free replay ----------------

A, B = "NCT00000001", "NCT00000002"
TRIALS = {
    A: {"summary": {"nct_id": A, "title": "Hypertension trial", "conditions": ["Hypertension"]}, "criteria_text": "x"},
    B: {"summary": {"nct_id": B, "title": "Asthma trial", "conditions": ["Asthma"]}, "criteria_text": "y"},
}
CRIT = {"criteria": [{"index": 0, "kind": "inclusion", "source_text": "Hypertension", "entity": "hypertension"}]}
MET = {"verdicts": [{"criterion_index": 0, "status": "met", "evidence_refs": ["N2"], "reason": "known HTN"}]}


def dec(action: str, **args: Any) -> dict[str, Any]:
    return {"action": action, "args": args, "rationale": "test", "confidence": 1.0}


def script() -> dict[str, list[dict[str, Any]]]:
    rank = {"ranked": [{"nct_id": A, "rank": 1, "explanation": "a"}, {"nct_id": B, "rank": 2, "explanation": "b"}]}
    return {
        "parse_criteria": [CRIT, {"criteria": [{**CRIT["criteria"][0], "source_text": "Asthma", "entity": "asthma"}]}],
        "judge_trial": [MET, MET],
        "rank_trials": [rank, rank],
        "supervisor": [
            dec("search_trials", conditions="hypertension"),
            dec("parse_criteria", trial_ids=[A, B]),
            dec("judge_trials", trial_ids=[A, B]),
            dec("rank"),
            dec("finish", summary="done"),
        ],
    }


async def test_three_systems_then_replay_gives_same_metrics_for_free() -> None:
    patient = EvalPatient("sigir", "sigir-1", SIGIR, {A: 2, B: 0}, pool=[A, B])
    store = DictStore()
    first = EvalLLM(ScriptedLLM(script()), store)
    r1 = await runner.run_patient(first, patient, TRIALS, runner.SYSTEMS, Limits())
    assert {s: r1.systems[s].error for s in runner.SYSTEMS} == {"bm25": "", "fixed": "", "supervisor": ""}
    assert r1.systems["bm25"].ranking == [A, B]
    assert r1.systems["fixed"].ranking == [A, B]
    assert r1.systems["supervisor"].ranking == [A, B]
    assert r1.systems["supervisor"].steps == 5
    # The supervisor reused the fixed pipeline's parse and judge replies: only its own decisions were bought.
    assert first.usage.cache_hits >= 4

    replay = EvalLLM(ScriptedLLM({}), store, replay=True)
    r2 = await runner.run_patient(replay, patient, TRIALS, runner.SYSTEMS, Limits())
    assert replay.spent.calls == 0
    price = {"input": 2.0, "output": 10.0}
    assert runner.score([r1], price) == runner.score([r2], price) or (
        # latency differs between runs; every quality metric must match
        _without_time(runner.score([r1], price)) == _without_time(runner.score([r2], price))
    )


def _without_time(table: dict[str, Any]) -> dict[str, Any]:
    return {
        c: {s: {k: v for k, v in row.items() if k != "seconds_per_patient"} for s, row in sys.items()}
        for c, sys in table.items()
    }


def test_threshold_check() -> None:
    row = {"ndcg@10": 0.5, "cost_per_patient_usd": 0.01, "citation_validity": 1.0}
    table = {"sigir": {"supervisor": dict(row), "bm25": {**row, "ndcg@10": 0.4}, "fixed": {**row, "ndcg@10": 0.5}}}
    assert runner.check_thresholds(table, None)[0].startswith("phase6 thresholds not set")
    assert runner.check_thresholds(table, {"citation_validity": 1.0}) == []
    table["sigir"]["bm25"]["ndcg@10"] = 0.6
    assert "does not beat BM25" in runner.check_thresholds(table, {})[0]


def test_ndcg_ideal_comes_from_the_pool() -> None:
    """A perfect ranking of the pool scores 1.0 even when the patient has more judged trials outside it."""
    res = runner.PatientResult("sigir", "t", [A, B], {A: 2, B: 0, "NCT09999999": 2}, [])
    res.systems["bm25"] = runner.SystemResult("bm25", [A, B])
    assert runner.score([res], None)["sigir"]["bm25"]["ndcg@10"] == 1.0


def test_errored_patients_are_left_out_of_scoring() -> None:
    good = runner.PatientResult("sigir", "t1", [A, B], {A: 2, B: 0}, [])
    good.systems["bm25"] = runner.SystemResult("bm25", [A, B])
    bad = runner.PatientResult("sigir", "t2", [A, B], {A: 0, B: 2}, [])
    bad.systems["bm25"] = runner.SystemResult("bm25", [A, B])
    bad.systems["fixed"] = runner.SystemResult("fixed", [A, B], error="LlmError: boom")
    table = runner.score([good, bad], None)
    assert table["sigir"]["bm25"]["patients"] == 1
    assert table["sigir"]["bm25"]["ndcg@10"] == 1.0


def test_error_detail_has_field_paths_not_values() -> None:
    from pydantic import ValidationError

    from agents.schemas import TrialSummary

    with pytest.raises(ValidationError) as info:
        TrialSummary.model_validate({"nct_id": "N", "title": None})
    detail = runner.error_detail(info.value)
    assert detail.startswith("ValidationError in TrialSummary: title (string_type)")


def test_store_reconnects_once_after_dropped_connection(monkeypatch: pytest.MonkeyPatch) -> None:
    from django.db import OperationalError, connection

    from evals.phase6.cache import with_reconnect

    closed: list[bool] = []
    monkeypatch.setattr(connection, "close", lambda: closed.append(True))
    failures = [OperationalError("server closed the connection unexpectedly")]

    def query() -> str:
        if failures:
            raise failures.pop()
        return "row"

    assert with_reconnect(query) == "row"
    assert closed == [True]
    with pytest.raises(OperationalError):
        with_reconnect(lambda: (_ for _ in ()).throw(OperationalError("down")))


def test_store_closes_worker_thread_connections(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each patient runs in a new event loop with new threads; their connections must not pile up."""
    import threading

    from django.db import connections

    from evals.phase6.cache import with_reconnect

    closed: list[str] = []
    # Patch the class: each thread has its own connection object.
    wrapper = type(connections["default"])
    monkeypatch.setattr(wrapper, "close", lambda self: closed.append(threading.current_thread().name))

    assert with_reconnect(lambda: "main") == "main"
    assert closed == []  # the main thread keeps its connection

    worker = threading.Thread(target=lambda: with_reconnect(lambda: "row"), name="worker")
    worker.start()
    worker.join()
    assert closed == ["worker"]


def test_error_detail_shows_database_message() -> None:
    from django.db import OperationalError

    from evals.phase6.runner import error_detail

    exc = OperationalError(
        'connection failed: connection to server at "127.0.0.1" failed\nFATAL:  sorry, too many clients already'
    )
    assert error_detail(exc) == "OperationalError: FATAL:  sorry, too many clients already"
    assert error_detail(RuntimeError("x")) == "RuntimeError"


def test_recorded_thresholds_pass_on_the_baseline_numbers() -> None:
    """The phase6 thresholds (ADR 0005) applied to the pilot baseline table (2026-11-20, 30 of 30 scored)."""
    from data.loaders.common import DATA_DIR
    from evals.phase6.runner import check_thresholds

    thresholds = json.loads((DATA_DIR / "eval_thresholds.json").read_text())["phase6"]

    def row(ndcg: float) -> dict[str, Any]:
        return {"ndcg@10": ndcg, "citation_validity": 1.0, "cost_per_patient_usd": None}

    table = {
        "sigir": {"bm25": row(0.7097), "fixed": row(0.8727), "supervisor": row(0.8561)},
        "trec_2021": {"bm25": row(0.8008), "fixed": row(0.8895), "supervisor": row(0.8985)},
        "trec_2022": {"bm25": row(0.7334), "fixed": row(0.8809), "supervisor": row(0.8473)},
    }
    assert check_thresholds(table, thresholds) == []
    # A supervisor that drops 0.06 below the fixed pipeline, or no longer beats BM25 by 0.05, fails.
    table["sigir"]["supervisor"] = row(0.8127)
    assert any("below fixed pipeline" in f for f in check_thresholds(table, thresholds))
    table["trec_2021"]["supervisor"] = row(0.84)
    assert any("does not beat BM25" in f for f in check_thresholds(table, thresholds))


def test_system_rerun_after_transient_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A rate-limit failure mid-run changes the supervisor's path; the runner re-runs it so the cache holds
    the error-free path (the one a replay takes)."""
    import asyncio

    from agents.guards import Limits

    llm = EvalLLM(ScriptedLLM({}), DictStore())
    calls: list[str] = []

    async def fake(system: str, llm_: EvalLLM, *args: Any) -> runner.SystemResult:
        calls.append(system)
        if system == "supervisor" and calls.count("supervisor") == 1:
            llm_.transient_errors += 1  # a 429 ran out of retries during this attempt
            return runner.SystemResult("supervisor", ["NCT1"], error="LlmError: failed after 3 attempts: HTTP 429")
        return runner.SystemResult(system, ["NCT1"])

    monkeypatch.setattr(runner, "_run_system", fake)
    monkeypatch.setattr(runner, "available", lambda trials, pool: ["NCT1"])
    p = EvalPatient("sigir", "t1", "60 year old man with asthma.", {"NCT1": 2}, ["NCT1"])
    result = asyncio.run(runner.run_patient(llm, p, {}, ("fixed", "supervisor"), Limits()))
    assert calls == ["fixed", "supervisor", "supervisor"]
    assert result.systems["supervisor"].error == ""  # the clean second attempt counts


def test_eval_cache_single_flight_for_concurrent_duplicates() -> None:
    """Two parallel calls with the same prompt: one live call, both callers get the stored reply."""
    import threading
    import time

    class Slow:
        model = ScriptedLLM({}).model

        def __init__(self) -> None:
            self.usage = ScriptedLLM({}).usage
            self.calls = 0

        def complete(self, **_: Any) -> Any:
            self.calls += 1
            time.sleep(0.05)
            return (
                Ranking(ranked=[])
                if self.calls == 1
                else Ranking.model_validate({"ranked": [{"nct_id": "X", "rank": 1, "explanation": "second"}]})
            )

    inner = Slow()
    llm = EvalLLM(inner, DictStore())
    out: list[Any] = []
    threads = [
        threading.Thread(
            target=lambda: out.append(llm.complete(task="rank_trials", system="s", user="u", schema=Ranking))
        )
        for _ in range(2)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert inner.calls == 1
    assert out[0] == out[1] == Ranking(ranked=[])
