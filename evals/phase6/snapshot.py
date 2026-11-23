"""Frozen trial snapshot: registry summary + eligibility text for every pooled trial, fetched once.

Lives in data/external/eval/phase6/snapshot.json (gitignored). Its sha256 goes in the report, so a
result can be tied to the exact trial texts it was computed on. Trials the registry no longer serves
are recorded as missing and left out of every system's pool.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.loaders.eval_data import EVAL_DIR
from mcp_servers.clinicaltrials.server import BASE_URL, DESTINATION, eligibility, summarize
from mcp_servers.common.http import OutboundClient, UpstreamError, UpstreamErrorKind

log = logging.getLogger(__name__)

PHASE6_DIR = EVAL_DIR / "phase6"
SNAPSHOT_PATH = PHASE6_DIR / "snapshot.json"

Snapshot = dict[str, dict[str, Any]]  # nct_id -> {"summary": {...}, "criteria_text": str} | {"missing": reason}


def load(path: Path = SNAPSHOT_PATH) -> Snapshot:
    if not path.exists():
        return {}
    data: Snapshot = json.loads(path.read_text())["trials"]
    return data


def save(trials: Snapshot, path: Path = SNAPSHOT_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"created": datetime.now(UTC).isoformat(timespec="seconds"), "trials": dict(sorted(trials.items()))}
    path.write_text(json.dumps(body, indent=1))


def digest(trials: Snapshot) -> str:
    return hashlib.sha256(json.dumps(trials, sort_keys=True).encode()).hexdigest()[:16]


def available(trials: Snapshot, pool: list[str]) -> list[str]:
    return [n for n in pool if n in trials and "missing" not in trials[n]]


def trial_text(entry: dict[str, Any]) -> str:
    s = entry["summary"]
    return " ".join([s.get("title") or "", " ".join(s.get("conditions") or []), entry.get("criteria_text") or ""])


async def build(nct_ids: list[str], path: Path = SNAPSHOT_PATH, client: OutboundClient | None = None) -> Snapshot:
    """Fetch the trials not yet in the snapshot. NCT ids only leave the system (public registry ids)."""
    trials = load(path)
    todo = sorted(set(nct_ids) - set(trials))
    if not todo:
        return trials
    own = client is None
    client = client or OutboundClient(DESTINATION, BASE_URL, rate_per_min=50)
    try:
        for i, nct in enumerate(todo, start=1):
            try:
                study = await client.get(f"/studies/{nct}", {"format": "json"})
                trials[nct] = {"summary": summarize(study), "criteria_text": eligibility(study)["criteria_text"] or ""}
            except UpstreamError as exc:
                if exc.kind != UpstreamErrorKind.NOT_FOUND:
                    raise
                trials[nct] = {"missing": "not_found"}
            if i % 25 == 0:
                save(trials, path)  # keep progress if the fetch is interrupted
                log.info("snapshot_progress fetched=%d of=%d", i, len(todo))
    finally:
        save(trials, path)
        if own:
            await client.aclose()
    return trials


def build_sync(nct_ids: list[str], path: Path = SNAPSHOT_PATH) -> Snapshot:
    return asyncio.run(build(nct_ids, path))
