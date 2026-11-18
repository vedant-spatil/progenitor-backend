"""Postgres-backed LLM response cache (agents.llm.LlmCache protocol)."""

from __future__ import annotations

from typing import Any

from apps.runs.models import LlmCache


class DjangoLlmCache:
    def __init__(self, task: str = "") -> None:
        self.task = task

    def get(self, key: str) -> dict[str, Any] | None:
        row = LlmCache.objects.filter(key=key).only("response").first()
        return row.response if row else None

    def set(self, key: str, value: dict[str, Any]) -> None:
        LlmCache.objects.update_or_create(key=key, defaults={"response": value, "task": self.task})
