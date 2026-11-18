"""Structured telemetry: one JSON line per event with ids, status, duration, error class. No payloads."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger("progenitor.telemetry")


def emit(event: str, **fields: Any) -> None:
    log.info(json.dumps({"event": event, **fields}, separators=(",", ":"), default=str))


@contextmanager
def timed(event: str, **fields: Any) -> Iterator[None]:
    started = time.perf_counter()
    status, err = "ok", ""
    try:
        yield
    except Exception as exc:
        status, err = "error", type(exc).__name__
        raise
    finally:
        extra = {"error_type": err} if err else {}
        emit(event, status=status, duration_ms=round((time.perf_counter() - started) * 1000), **fields, **extra)
