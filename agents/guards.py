"""Guards validate and limit; they never pick the next worker (project rule).

Phase 3: citation guard. Phase 4 adds schema re-ask, allowlist, budgets, loop
detection, and the approval gate.
"""

from __future__ import annotations

from agents.schemas import Verdict, VerdictStatus


def enforce_citations(verdicts: list[Verdict], valid_refs: set[str]) -> list[Verdict]:
    """A met/not_met verdict must cite at least one resource present in the patient's record.

    Unknown references are dropped. If no valid reference remains, the verdict
    becomes `unknown` and is marked `downgraded`.
    """
    out: list[Verdict] = []
    for v in verdicts:
        refs = [r for r in v.evidence_refs if r in valid_refs]
        if v.status in (VerdictStatus.MET, VerdictStatus.NOT_MET) and not refs:
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
