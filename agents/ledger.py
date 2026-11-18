"""Decision ledger: every supervisor decision is recorded with rationale, refs, and guard outcome.

`LedgerSink` is the interface; `MemoryLedger` serves tests; apps.runs.ledger.DjangoLedger
writes runs.Decision rows. Tracing to Langfuse is a separate sink (agents.tracing).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class LedgerEntry:
    run_id: str
    step: int
    action: str
    args: dict[str, Any]
    rationale: str
    state_refs: list[str]
    confidence: float
    guard_outcome: str
    guard_detail: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    attempt: int = 0  # 1 for the re-ask after a guard rejection


class LedgerSink(Protocol):
    def record(self, entry: LedgerEntry) -> None: ...


@dataclass
class MemoryLedger:
    entries: list[LedgerEntry] = field(default_factory=list)

    def record(self, entry: LedgerEntry) -> None:
        self.entries.append(entry)
