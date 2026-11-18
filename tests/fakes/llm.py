"""Scripted fake LLM for logic tests (no live LLM in unit tests)."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, TypeVar

from pydantic import BaseModel

from agents.llm import Usage

T = TypeVar("T", bound=BaseModel)


@dataclass
class Call:
    task: str
    system: str
    user: str
    schema: str


@dataclass
class ScriptedLLM:
    """Returns scripted replies per task, in order. Raises if a task is called more often than scripted."""

    replies: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    model: str = "scripted"
    calls: list[Call] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        self._queues: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
        for task, items in self.replies.items():
            self._queues[task].extend(items)

    def complete(self, *, task: str, system: str, user: str, schema: type[T]) -> T:
        self.calls.append(Call(task, system, user, schema.__name__))
        if not self._queues[task]:
            raise AssertionError(f"LLM task {task!r} called more times than scripted")
        return schema.model_validate(self._queues[task].popleft())
