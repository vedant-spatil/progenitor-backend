"""Gate progress runner: step order, stop on first failure, status JSON, report."""

import json
from pathlib import Path

import pytest

from scripts import gate


@pytest.fixture
def tmp_gates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(gate, "GATES_DIR", tmp_path)
    return tmp_path


def test_gate_chain_matches_makefile() -> None:
    names = [s.name for s in gate.steps_for(2)]
    assert names[0] == "install backend deps"
    assert names[-1] == "all unit tests (cassettes only)"
    assert len(names) == len(gate.GATES[0]) + len(gate.GATES[2])  # gate-2 depends on gate-0, not gate-1
    four = gate.steps_for(4)
    assert len(four) == sum(len(gate.GATES[p]) for p in (0, 2, 3, 4))
    assert four[-1].name == "phase 4 supervisor scenarios"


def test_unknown_phase_exits() -> None:
    with pytest.raises(SystemExit):
        gate.steps_for(99)


def test_run_stops_at_first_failure_and_writes_json(tmp_gates: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        gate,
        "GATES",
        {0: [gate.Step("ok", "true"), gate.Step("boom", "echo broken-output; false"), gate.Step("never", "true")]},
    )
    assert gate.run(0) == 1
    last = json.loads((tmp_gates / "last.json").read_text())
    assert last["status"] == "fail"
    assert [s["status"] for s in last["steps"]] == ["pass", "fail", "not_run"]
    assert "broken-output" in gate.tail_of_step(Path(last["log"]), 2)


def test_status_and_report_after_pass(
    tmp_gates: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gate, "PARENT", {0: None})
    monkeypatch.setattr(gate, "GATES", {0: [gate.Step("a", "true"), gate.Step("b", "true")]})
    assert gate.run(0) == 0
    capsys.readouterr()
    assert gate.status() == 0
    out = capsys.readouterr().out
    assert "2/2 done" in out
    assert gate.report() == 0
    assert "| 2 | b | pass |" in capsys.readouterr().out
