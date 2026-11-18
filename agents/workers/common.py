"""Helpers shared by workers: calling MCP tools and rendering compact prompt context."""

from __future__ import annotations

import json
import os
from collections.abc import Awaitable, Callable, Mapping
from datetime import date
from typing import Any

from agents.schemas import Criterion, PatientProfile

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]


class ToolCallError(RuntimeError):
    pass


def tool_caller(tools: Mapping[str, Any]) -> ToolCaller:
    """Wrap loaded LangChain MCP tools as `await call(name, args) -> dict`."""

    async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
        result = await tools[name].ainvoke(args)
        text = result[0]["text"] if isinstance(result, list) and result else result
        try:
            data = json.loads(text) if isinstance(text, str) else text
        except json.JSONDecodeError as exc:
            raise ToolCallError(f"{name}: {str(text)[:200]}") from exc
        if not isinstance(data, dict):
            raise ToolCallError(f"{name}: unexpected result")
        return data

    return call


def _short(text: str, limit: int = 70) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def profile_context(
    profile: PatientProfile, as_of: date | None = None, max_chars: int | None = None
) -> tuple[str, dict[str, str]]:
    """Compact, citable profile text for prompts plus {short id: FHIR ref}.

    Lines start with short ids (P, C1, M1, L1) instead of FHIR UUID refs, which keeps
    a full record under small per-request token limits. Facts are deduplicated by code.
    Over `max_chars` (env LLM_PROFILE_MAX_CHARS, default 10000), the oldest low-value facts
    are dropped first (stopped meds, resolved conditions, oldest labs) and the count is stated.
    """
    max_chars = max_chars or int(os.environ.get("LLM_PROFILE_MAX_CHARS", "10000"))
    aliases: dict[str, str] = {"P": profile.patient_ref}
    age = "unknown" if profile.age_years is None else f"{profile.age_years}{'+' if profile.age_years == 90 else ''}"
    lines = [f"Profile date: {(as_of or date.today()).isoformat()}", f"P | age {age} | sex {profile.sex or 'unknown'}"]

    def alias(prefix: str, ref: str) -> str:
        n = sum(k.startswith(prefix) for k in aliases) + 1
        aliases[f"{prefix}{n}"] = ref
        return f"{prefix}{n}"

    seen: set[tuple[str, str]] = set()
    for c in sorted(
        profile.conditions,
        key=lambda c: (c.clinical_status != "active", -(c.onset.toordinal() if c.onset else 0)),
    ):
        if (c.code, c.clinical_status) in seen:
            continue
        seen.add((c.code, c.clinical_status))
        onset = f" | onset {c.onset}" if c.onset else ""
        system = c.system.rsplit("/", 1)[-1]
        lines.append(
            f"{alias('C', c.resource_id)} | {_short(c.display)} [{system} {c.code}] | {c.clinical_status}{onset}"
        )
    seen.clear()
    for m in profile.medications:  # newest first from the profiler
        if (m.code, m.status) in seen:
            continue
        seen.add((m.code, m.status))
        lines.append(
            f"{alias('M', m.resource_id)} | {_short(m.display)} [rxnorm {m.code}] | {m.status} | {m.authored_on}"
        )
    for lab in profile.labs:
        value = f"{lab.value:g} {lab.unit}".strip() if lab.value is not None else lab.value_text
        lines.append(
            f"{alias('L', lab.resource_id)} | {_short(lab.display)} [loinc {lab.code}] = {value} | {lab.effective}"
        )
    return _fit(lines, max_chars), aliases


def _fit(lines: list[str], max_chars: int) -> str:
    """Drop facts until the text fits: stopped meds, then resolved conditions, then oldest labs."""
    text = "\n".join(lines)
    if len(text) <= max_chars:
        return text

    def date_of(line: str) -> str:
        return line.rsplit("| ", 1)[-1].strip()

    order = (
        sorted((ln for ln in lines if ln.startswith("M") and "| active |" not in ln), key=date_of)
        + sorted((ln for ln in lines if ln.startswith("C") and "| active" not in ln), key=date_of)
        + sorted((ln for ln in lines if ln.startswith("L")), key=date_of)
    )
    keep = list(lines)
    dropped = 0
    for line in order:
        if len("\n".join(keep)) <= max_chars - 80:
            break
        keep.remove(line)
        dropped += 1
    keep.append(f"[{dropped} older facts omitted for length; treat them as not shown]")
    return "\n".join(keep)


def profile_lines(profile: PatientProfile, as_of: date | None = None) -> str:
    return profile_context(profile, as_of)[0]


def criteria_lines(criteria: list[Criterion]) -> str:
    return "\n".join(f"[{c.index}] ({c.kind}) {c.source_text}" for c in criteria)
