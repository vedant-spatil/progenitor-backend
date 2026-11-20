"""Shared LLM cache for the eval, with per-call token accounting.

Every system calls the same `EvalLLM`. A reply is bought once and reused by any later call with the
same model, prompt, and schema (the fixed pipeline and the supervisor share parse and judge calls;
a trial is parsed once for all patients). The cache row keeps the tokens the original call used, so:

- `usage`  counts tokens as if nothing were cached: the cost of running a system on its own.
- `spent`  counts tokens actually bought in this process.

With `replay=True` a cache miss raises instead of calling the model: re-runs cost nothing and give
the same metrics.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Callable
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from agents.llm import ReplyError, StructuredLLM, Usage

T = TypeVar("T", bound=BaseModel)


class ReplayMiss(RuntimeError):
    """A replay needed an LLM reply that is not in the cache (inputs changed since the baseline)."""


class Store(Protocol):
    def get(self, key: str) -> dict[str, Any] | None: ...

    def set(self, key: str, value: dict[str, Any], task: str) -> None: ...


class DictStore:
    def __init__(self) -> None:
        self.rows: dict[str, dict[str, Any]] = {}

    def get(self, key: str) -> dict[str, Any] | None:
        return self.rows.get(key)

    def set(self, key: str, value: dict[str, Any], task: str) -> None:
        self.rows[key] = value


class DjangoStore:
    """Rows in runs.LlmCache under their own key space (task 'phase6:<task>')."""

    def get(self, key: str) -> dict[str, Any] | None:
        from apps.runs.models import LlmCache

        row = with_reconnect(lambda: LlmCache.objects.filter(key=key).only("response").first())
        return row.response if row else None

    def set(self, key: str, value: dict[str, Any], task: str) -> None:
        from apps.runs.models import LlmCache

        with_reconnect(
            lambda: LlmCache.objects.update_or_create(
                key=key, defaults={"response": value, "task": f"phase6:{task}"[:64]}
            )
        )


def with_reconnect[R](fn: Callable[[], R]) -> R:
    """Run a DB call; on a dropped connection, reconnect once and retry.

    The eval calls the store from worker threads (asyncio.to_thread) for hours. Django only manages
    connections around HTTP requests, so:
    - a connection the server dropped stays broken in its thread until it is closed (hence the retry);
    - each worker thread opens its own connection and never closes it. Every patient runs in a new event
      loop with new threads, so the connections pile up until Postgres refuses more ("too many clients").
      On a worker thread the connection is therefore closed after each call.
    """
    from django.db import OperationalError, connection

    try:
        try:
            return fn()
        except OperationalError:
            connection.close()  # this thread's connection; the next query opens a new one
            return fn()
    finally:
        if threading.current_thread() is not threading.main_thread():
            connection.close()


def eval_key(model: str, task: str, system: str, user: str, schema: type[BaseModel]) -> str:
    raw = json.dumps(["phase6", model, task, system, user, schema.model_json_schema()], sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()


class EvalLLM:
    def __init__(self, inner: StructuredLLM, store: Store, *, replay: bool = False) -> None:
        self.inner = inner
        self.store = store
        self.replay = replay
        self.model: str = inner.model
        self.usage = Usage()  # virtual: as if every call were made
        self.spent = Usage()  # real: calls made by this process
        self._lock = threading.Lock()  # one live call at a time: exact per-call token deltas, gentler on rate limits

    def complete(self, *, task: str, system: str, user: str, schema: type[T]) -> T:
        key = eval_key(self.model, task, system, user, schema)
        hit = self.store.get(key)
        if hit is not None and "error" in hit and not self.replay:
            hit = None  # a recorded failure: live runs try again (a fix may have landed since)
        if hit is None:
            if self.replay:
                raise ReplayMiss(f"no cached reply for {task} (key {key[:12]})")
            with self._lock:
                before = (self.inner.usage.tokens_in, self.inner.usage.tokens_out)
                failure: ReplyError | None = None
                try:
                    result = self.inner.complete(task=task, system=system, user=user, schema=schema)
                except ReplyError as exc:
                    failure = exc
                t_in = self.inner.usage.tokens_in - before[0]
                t_out = self.inner.usage.tokens_out - before[1]
                entry: dict[str, Any] = {"tokens_in": t_in, "tokens_out": t_out}
                if failure is None:
                    entry["reply"] = result.model_dump(mode="json")
                else:
                    entry["error"] = str(failure)  # replay reproduces it; a live run tries again
                self.store.set(key, entry, task)
                self.spent.calls += 1
                self.spent.tokens_in += t_in
                self.spent.tokens_out += t_out
            hit = entry
            if failure is not None:
                self._count(hit)
                raise failure
            reply: BaseModel = result
        else:
            with self._lock:
                self.usage.cache_hits += 1
            if "error" in hit:
                self._count(hit)
                raise ReplyError(str(hit["error"]))
            reply = schema.model_validate(hit["reply"])
        self._count(hit)
        return reply  # type: ignore[return-value]  # validated against `schema` above

    def _count(self, hit: dict[str, Any]) -> None:
        """Virtual usage: every call counts, cached or not, failed or not (failed calls cost tokens too)."""
        with self._lock:
            self.usage.calls += 1
            self.usage.tokens_in += int(hit["tokens_in"])
            self.usage.tokens_out += int(hit["tokens_out"])
