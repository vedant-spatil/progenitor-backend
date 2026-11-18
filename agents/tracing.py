"""Tracing for supervisor decisions. TRACING_MODE selects the backend:

- langfuse (default, local dev): one span per decision in self-hosted Langfuse
  (needs LANGFUSE_* keys and the `tracing` compose profile).
- cloudwatch (EC2): one structured JSON log line per decision. Docker's awslogs driver
  ships container logs to CloudWatch Logs (docker-compose.ec2.yml); query with Logs Insights.
  No collector, no extra container. The rationale is left out (no record text at INFO).
- off: no tracer. The Postgres decision ledger is written in every mode.
"""

from __future__ import annotations

import logging
import os
from dataclasses import asdict
from typing import Any, Protocol

from agents.ledger import LedgerEntry
from agents.telemetry import emit

log = logging.getLogger(__name__)


class Tracer(Protocol):
    def decision(self, entry: LedgerEntry) -> None: ...


class LangfuseTracer:
    def __init__(self, client: Any) -> None:
        self.client = client

    def decision(self, entry: LedgerEntry) -> None:
        try:
            with self.client.start_as_current_observation(
                name=f"supervisor.{entry.action}",
                as_type="span",
                input={"step": entry.step, "action": entry.action, "args": entry.args},
                output={"guard_outcome": entry.guard_outcome, "guard_detail": entry.guard_detail},
                metadata={
                    "run_id": entry.run_id,
                    **{
                        k: v
                        for k, v in asdict(entry).items()
                        if k in ("model", "tokens_in", "latency_ms", "confidence")
                    },
                },
            ):
                pass
        except Exception as exc:  # noqa: BLE001 - tracing must never break a run
            log.debug("langfuse span failed (%s)", exc.__class__.__name__)


class LogTracer:
    """CloudWatch mode: a JSON telemetry line per decision (ids, action, outcome, counts)."""

    def decision(self, entry: LedgerEntry) -> None:
        emit(
            "supervisor_decision",
            run_id=entry.run_id,
            step=entry.step,
            attempt=entry.attempt,
            action=entry.action,
            arg_keys=sorted(entry.args),
            state_refs=len(entry.state_refs),
            confidence=entry.confidence,
            guard_outcome=entry.guard_outcome,
            guard_detail=entry.guard_detail[:120],
            model=entry.model,
            tokens_in=entry.tokens_in,
            latency_ms=entry.latency_ms,
        )


def tracer_from_env() -> Tracer | None:
    mode = os.environ.get("TRACING_MODE", "langfuse").lower()
    if mode == "off":
        return None
    if mode == "cloudwatch":
        return LogTracer()
    if mode != "langfuse":
        raise ValueError(f"unknown TRACING_MODE {mode!r} (langfuse | cloudwatch | off)")
    public, secret = os.environ.get("LANGFUSE_PUBLIC_KEY"), os.environ.get("LANGFUSE_SECRET_KEY")
    if not (public and secret):
        return None
    from langfuse import Langfuse

    client = Langfuse(
        public_key=public, secret_key=secret, base_url=os.environ.get("LANGFUSE_URL", "http://localhost:3001")
    )
    return LangfuseTracer(client)
