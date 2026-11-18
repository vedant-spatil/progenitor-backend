"""LangGraph run state for the supervisor graph.

Parallel judges each update one trial, so `trials` uses a merge reducer keyed by NCT id.
Lists that several nodes extend use `operator.add`.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict


def merge_trials(
    left: dict[str, dict[str, Any]] | None, right: dict[str, dict[str, Any]] | None
) -> dict[str, dict[str, Any]]:
    """Merge per-trial updates: right's fields override left's, per NCT id."""
    out = {k: dict(v) for k, v in (left or {}).items()}
    for nct, update in (right or {}).items():
        out[nct] = {**out.get(nct, {}), **update}
    return out


class RunState(TypedDict, total=False):
    run_id: str
    patient_id: str
    as_of: str  # ISO date the record is judged against
    profile: dict[str, Any] | None
    # nct_id -> {summary, criteria, verdicts, dropped, papers, interventions}
    trials: Annotated[dict[str, dict[str, Any]], merge_trials]
    searches: Annotated[list[dict[str, Any]], operator.add]
    decisions: Annotated[list[dict[str, Any]], operator.add]
    clinician_qa: Annotated[list[dict[str, Any]], operator.add]
    pending: dict[str, Any]
    rejection: str | None  # last guard rejection, shown to the supervisor once
    steps: int
    tokens_used: int
    shortlist: list[dict[str, Any]]
    approved_trial_ids: list[str]  # set by a clinician (Phase 5 API); empty = not approved
    referrals: list[dict[str, Any]]
    status: str
    finish_reason: str
    summary: str
