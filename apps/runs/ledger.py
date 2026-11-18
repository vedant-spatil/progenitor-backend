"""Decision ledger backed by runs.Decision (one row per supervisor decision attempt)."""

from __future__ import annotations

from agents.ledger import LedgerEntry
from apps.runs.models import Decision


class DjangoLedger:
    """Sync ORM writes; the graph calls it off the event loop (asyncio.to_thread)."""

    def record(self, entry: LedgerEntry) -> None:
        Decision.objects.create(
            run_id=entry.run_id,
            step=entry.step,
            attempt=entry.attempt,
            action=entry.action,
            args=entry.args,
            rationale=entry.rationale,
            state_refs=entry.state_refs,
            confidence=entry.confidence,
            model=entry.model,
            tokens_in=entry.tokens_in,
            tokens_out=entry.tokens_out,
            latency_ms=entry.latency_ms,
            guard_outcome=entry.guard_outcome,
            guard_detail=entry.guard_detail[:255],
        )


class RunLedger(DjangoLedger):
    """Ledger for API runs: the Decision row, a decision audit event, and a live event for SSE."""

    def record(self, entry: LedgerEntry) -> None:
        from apps.audit.services import audit_decision
        from apps.runs.events import publish

        super().record(entry)
        audit_decision(entry)
        publish(
            entry.run_id,
            "decision",
            {
                "step": entry.step,
                "attempt": entry.attempt,
                "action": entry.action,
                "guard_outcome": entry.guard_outcome,
                "state_refs": entry.state_refs,
            },
        )
