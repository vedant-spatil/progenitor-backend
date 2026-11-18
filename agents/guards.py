"""Guards validate and limit; they never pick the next worker (project rule).

Phase 3: citation and relevance guards for Judge verdicts.
Phase 4: supervisor guards (budgets, argument and precondition checks, approval gate,
loop detection). The schema re-ask and action allowlist live in the LLM call and the
`Decision.action` Literal type.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from agents.schemas import Criterion, Decision, PatientProfile, Verdict, VerdictStatus
from agents.workers.common import focus_terms, mentions

EVIDENCED = (VerdictStatus.MET, VerdictStatus.NOT_MET, VerdictStatus.NOT_APPLICABLE)


def enforce_citations(verdicts: list[Verdict], valid_refs: set[str]) -> list[Verdict]:
    """A met/not_met verdict must cite at least one resource present in the patient's record.

    Unknown references are dropped. If no valid reference remains, the verdict
    becomes `unknown` and is marked `downgraded`.
    """
    out: list[Verdict] = []
    for v in verdicts:
        refs = [r for r in v.evidence_refs if r in valid_refs]
        if v.status in EVIDENCED and not refs:
            out.append(
                v.model_copy(
                    update={
                        "status": VerdictStatus.UNKNOWN,
                        "evidence_refs": [],
                        "downgraded": True,
                        "reason": f"[no valid evidence cited] {v.reason}".strip(),
                    }
                )
            )
        else:
            out.append(v.model_copy(update={"evidence_refs": refs}))
    return out


def _downgrade(v: Verdict, why: str) -> Verdict:
    return v.model_copy(
        update={
            "status": VerdictStatus.UNKNOWN,
            "evidence_refs": [],
            "downgraded": True,
            "reason": f"[{why}] {v.reason}",
        }
    )


def enforce_relevance(verdicts: list[Verdict], criteria: list[Criterion], profile: PatientProfile) -> list[Verdict]:
    """A not_met verdict that cites only conditions/medications unrelated to the criterion becomes unknown.

    Stops "patient has no X" answers that cite some other diagnosis as evidence (absence of a
    record is not proof). Lab and Patient citations are left to the citation guard.
    """
    facts = {f.resource_id: f.display for f in [*profile.conditions, *profile.medications]}
    by_index = {c.index: c for c in criteria}
    out: list[Verdict] = []
    for v in verdicts:
        crit = by_index.get(v.criterion_index)
        cited = [r for r in v.evidence_refs if r in facts]
        if (
            v.status == VerdictStatus.NOT_MET
            and crit is not None
            and cited
            and len(cited) == len(v.evidence_refs)
            and not any(mentions(facts[r], focus_terms([crit.source_text, crit.entity])) for r in cited)
        ):
            out.append(_downgrade(v, "cited records unrelated to the criterion"))
        else:
            out.append(v)
    return out


# ---------------- Supervisor guards (Phase 4) ----------------
# Guards validate and limit. They may reject a decision (the supervisor re-decides with the
# reason) or force a finish (budget, loop, repeated rejection). They never pick a worker.


class GuardRejection(Exception):
    pass


@dataclass(frozen=True)
class Limits:
    max_steps: int = 40
    max_trials_judged: int = 30
    max_tokens: int = 400_000
    loop_repeat: int = 3

    @classmethod
    def from_env(cls) -> Limits:
        return cls(
            max_steps=int(os.environ.get("AGENT_MAX_STEPS", "40")),
            max_trials_judged=int(os.environ.get("AGENT_MAX_TRIALS_JUDGED", "30")),
            max_tokens=int(os.environ.get("AGENT_MAX_TOKENS", "400000")),
        )


class Outcome:
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    FORCED_FINISH = "forced_finish"
    LOOP_STOPPED = "loop_stopped"


REQUIRED_ARGS: dict[str, tuple[str, ...]] = {
    "search_trials": ("conditions",),
    "refine_search": ("mode",),
    "parse_criteria": ("trial_ids",),
    "judge_trials": ("trial_ids",),
    "resolve_unknowns": ("trial_id", "criterion_indexes"),
    "drop_trials": ("trial_ids", "reason"),
    "gather_evidence": ("trial_ids",),
    "ask_clinician": ("question",),
}
TRIAL_LIST_ACTIONS = {"parse_criteria", "judge_trials", "drop_trials", "gather_evidence"}


def forced_finish(reason: str) -> Decision:
    return Decision(
        action="finish",
        args={"summary": f"Run stopped by guard: {reason}. Partial results kept."},
        rationale=f"guard: {reason}",
        confidence=1.0,
    )


def _decision_key(action: str, args: dict[str, Any]) -> str:
    return f"{action}:{json.dumps(args, sort_keys=True, default=str)}"


def check(d: Decision, state: Mapping[str, Any], limits: Limits | None = None) -> tuple[Decision, str, str]:
    """Return (decision, outcome, detail). Raises GuardRejection for decisions the supervisor must redo."""
    limits = limits or Limits()
    if state.get("steps", 0) >= limits.max_steps:
        return forced_finish(f"step budget {limits.max_steps} reached"), Outcome.FORCED_FINISH, "steps"
    if state.get("tokens_used", 0) >= limits.max_tokens:
        return forced_finish(f"token budget {limits.max_tokens} reached"), Outcome.FORCED_FINISH, "tokens"

    missing = [a for a in REQUIRED_ARGS.get(d.action, ()) if d.args.get(a) in (None, "", [])]
    if missing:
        raise GuardRejection(f"{d.action} needs args: {', '.join(missing)}")

    trials: Mapping[str, Mapping[str, Any]] = state.get("trials") or {}
    if d.action in TRIAL_LIST_ACTIONS:
        ids = d.args["trial_ids"]
        if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
            raise GuardRejection("trial_ids must be a list of NCT ids")
        unknown = [i for i in ids if i not in trials]
        if unknown:
            raise GuardRejection(f"unknown trial ids (not found by any search): {', '.join(unknown[:5])}")
    if d.action == "resolve_unknowns" and d.args["trial_id"] not in trials:
        raise GuardRejection(f"unknown trial id {d.args['trial_id']}")

    if d.action == "judge_trials":
        not_parsed = [i for i in d.args["trial_ids"] if not trials[i].get("criteria")]
        if not_parsed:
            raise GuardRejection(f"criteria not parsed yet for: {', '.join(not_parsed[:5])}")
        # Cap on distinct trials judged per run. Re-judging an already judged trial is free.
        judged = {n for n, t in trials.items() if t.get("verdicts") is not None}
        ids = d.args["trial_ids"]
        new = [i for i in ids if i not in judged]
        room = max(0, limits.max_trials_judged - len(judged))
        if new and room == 0:
            raise GuardRejection(f"judging budget used: {limits.max_trials_judged} trials already judged")
        if len(new) > room:
            d = d.model_copy(update={"args": {**d.args, "trial_ids": [i for i in ids if i in judged] + new[:room]}})

    if d.action == "draft_referrals" and not state.get("approved_trial_ids"):
        raise GuardRejection("draft_referrals requires clinician approval of the shortlist")

    previous = [_decision_key(x["action"], x.get("args", {})) for x in (state.get("decisions") or [])]
    window = previous[-(limits.loop_repeat - 1) :] if limits.loop_repeat > 1 else []
    if len(window) == limits.loop_repeat - 1 and all(k == _decision_key(d.action, d.args) for k in window):
        return forced_finish(f"same decision repeated {limits.loop_repeat} times"), Outcome.LOOP_STOPPED, d.action
    return d, Outcome.ACCEPTED, ""
