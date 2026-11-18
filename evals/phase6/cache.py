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
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel

from agents.llm import StructuredLLM, Usage

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

        row = LlmCache.objects.filter(key=key).only("response").first()
        return row.response if row else None

    def set(self, key: str, value: dict[str, Any], task: str) -> None:
        from apps.runs.models import LlmCache

        LlmCache.objects.update_or_create(key=key, defaults={"response": value, "task": f"phase6:{task}"[:64]})


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
        if hit is None:
            if self.replay:
                raise ReplayMiss(f"no cached reply for {task} (key {key[:12]})")
            with self._lock:
                before = (self.inner.usage.tokens_in, self.inner.usage.tokens_out)
                result = self.inner.complete(task=task, system=system, user=user, schema=schema)
                t_in = self.inner.usage.tokens_in - before[0]
                t_out = self.inner.usage.tokens_out - before[1]
                self.store.set(
                    key, {"reply": result.model_dump(mode="json"), "tokens_in": t_in, "tokens_out": t_out}, task
                )
                self.spent.calls += 1
                self.spent.tokens_in += t_in
                self.spent.tokens_out += t_out
            hit = {"tokens_in": t_in, "tokens_out": t_out}
            reply: BaseModel = result
        else:
            reply = schema.model_validate(hit["reply"])
            with self._lock:
                self.usage.cache_hits += 1
        with self._lock:
            self.usage.calls += 1
            self.usage.tokens_in += int(hit["tokens_in"])
            self.usage.tokens_out += int(hit["tokens_out"])
        return reply  # type: ignore[return-value]  # validated against `schema` above
