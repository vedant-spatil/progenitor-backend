"""Offline tool layer for the eval: the agents see the frozen snapshot instead of live MCP servers.

- clinicaltrials_search_trials returns the patient's fixed candidate pool, whatever the query
  (the eval compares rankers on equal input; search quality is out of scope here).
- get_eligibility / get_trial read the snapshot.
- terminology lookups return no codes and PubMed returns no papers, so runs make no network calls
  and replay exactly.
"""

from __future__ import annotations

from typing import Any

from agents.fhir_reader import Resource
from agents.workers.common import ToolCallError
from evals.phase6.snapshot import Snapshot


class SnapshotTools:
    def __init__(self, trials: Snapshot, pool: list[str]) -> None:
        self.trials = trials
        self.pool = pool
        self.calls: list[str] = []

    async def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        if name == "clinicaltrials_search_trials":
            studies = [self.trials[n]["summary"] for n in self.pool]
            return {"studies": studies, "total": len(studies), "next_page_token": None}  # nosec B105 - pagination token, not a secret
        if name in ("clinicaltrials_get_eligibility", "clinicaltrials_get_trial"):
            nct = str(args.get("nct_id", ""))
            if nct not in self.pool:
                raise ToolCallError(f"{name}: {nct} is not in this patient's pool")
            entry = self.trials[nct]
            if name == "clinicaltrials_get_trial":
                return dict(entry["summary"])
            return {"nct_id": nct, "criteria_text": entry.get("criteria_text", "")}
        if name.startswith("terminology_"):
            return {"matches": [], "candidates": [], "preferred": None}
        if name == "pubmed_search_pubmed":
            return {"articles": []}
        raise ToolCallError(f"{name} is not available in the offline eval")


class TextPatientReader:
    """FHIR reader for text patients: the profile is preloaded, so a fetch adds nothing."""

    def __init__(self, topic_id: str) -> None:
        self.topic_id = topic_id

    async def read(self, resource_type: str, resource_id: str) -> Resource:
        return {"resourceType": "Patient", "id": self.topic_id}

    async def search(self, resource_type: str, params: dict[str, str]) -> tuple[list[Resource], list[Resource]]:
        return [], []
