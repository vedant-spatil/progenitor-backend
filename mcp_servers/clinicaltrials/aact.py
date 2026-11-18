"""AACT fallback for the ClinicalTrials.gov MCP server.

Used only when the live API is unavailable and AACT_DATABASE_URL points at a
restored AACT snapshot (see data/loaders/aact_restore.sh). Returns studies in
the same JSON shape as API v2 so the server code paths stay the same.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Protocol

STUDY_SQL = """
SELECT s.nct_id, s.brief_title, s.overall_status, s.phase, e.criteria, e.minimum_age,
       e.maximum_age, e.gender, e.healthy_volunteers,
       (SELECT array_agg(c.name) FROM ctgov.conditions c WHERE c.nct_id = s.nct_id) AS conditions
FROM ctgov.studies s LEFT JOIN ctgov.eligibilities e ON e.nct_id = s.nct_id
WHERE s.nct_id = %s
"""

SEARCH_SQL = """
SELECT DISTINCT s.nct_id, s.brief_title, s.overall_status, s.phase
FROM ctgov.studies s JOIN ctgov.conditions c ON c.nct_id = s.nct_id
WHERE c.downcase_name LIKE %s AND upper(s.overall_status) = ANY(%s)
ORDER BY s.nct_id LIMIT %s
"""


class AactBackend(Protocol):
    async def study(self, nct_id: str) -> dict[str, Any] | None: ...
    async def search(self, conditions: str, status: list[str], limit: int) -> list[dict[str, Any]]: ...


def _phases(value: str | None) -> list[str]:
    # AACT stores e.g. "Phase 2/Phase 3"; API v2 uses ["PHASE2", "PHASE3"].
    if not value or value.upper() in {"N/A", "NA"}:
        return []
    return [p.strip().upper().replace(" ", "") for p in value.split("/")]


class PostgresAact:
    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    def _query(self, sql: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        import psycopg

        with psycopg.connect(self.dsn, connect_timeout=5) as conn:
            return list(conn.execute(sql, params).fetchall())

    async def study(self, nct_id: str) -> dict[str, Any] | None:
        rows = await asyncio.to_thread(self._query, STUDY_SQL, (nct_id,))
        if not rows:
            return None
        nct, title, status, phase, criteria, min_age, max_age, sex, healthy, conds = rows[0]
        return {
            "protocolSection": {
                "identificationModule": {"nctId": nct, "briefTitle": title},
                "statusModule": {"overallStatus": (status or "").upper()},
                "designModule": {"phases": _phases(phase)},
                "conditionsModule": {"conditions": list(conds or [])},
                "eligibilityModule": {
                    "eligibilityCriteria": criteria or "",
                    "minimumAge": min_age,
                    "maximumAge": max_age,
                    "sex": (sex or "").upper(),
                    "healthyVolunteers": None if healthy is None else str(healthy).lower().startswith("accepts"),
                },
            }
        }

    async def search(self, conditions: str, status: list[str], limit: int) -> list[dict[str, Any]]:
        rows = await asyncio.to_thread(
            self._query, SEARCH_SQL, (f"%{conditions.lower()}%", [s.upper() for s in status], limit)
        )
        return [{"nct_id": r[0], "title": r[1], "status": (r[2] or "").upper(), "phases": _phases(r[3])} for r in rows]


def aact_from_env() -> AactBackend | None:
    dsn = os.environ.get("AACT_DATABASE_URL")
    return PostgresAact(dsn) if dsn else None
