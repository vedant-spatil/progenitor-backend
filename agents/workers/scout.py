"""Trial Scout: runs the supervisor's search spec against the ClinicalTrials.gov MCP server.

No LLM: the supervisor decides what to search (conditions, terms, phases); this
worker executes it and pages until max_results.
"""

from __future__ import annotations

from typing import Any

from agents.schemas import SearchSpec, TrialSummary
from agents.telemetry import timed
from agents.workers.common import ToolCaller


async def search_trials(spec: SearchSpec, call_tool: ToolCaller) -> tuple[list[TrialSummary], int]:
    """Returns (trials, total matching count reported by the registry)."""
    trials: list[TrialSummary] = []
    total = 0
    token: str | None = None
    with timed("worker", worker="scout"):
        while len(trials) < spec.max_results:
            args: dict[str, Any] = {
                "conditions": spec.conditions,
                "terms": spec.terms,
                "interventions": spec.interventions,
                "phases": spec.phases or None,
                "location": spec.location,
                "page_size": min(100, spec.max_results - len(trials)),
            }
            if token:
                args["page_token"] = token
            page = await call_tool("clinicaltrials_search_trials", {k: v for k, v in args.items() if v})
            total = int(page.get("total") or 0)
            trials += [
                TrialSummary.model_validate(
                    {k: s.get(k) for k in TrialSummary.model_fields if k in s} | {"nct_id": s["nct_id"]}
                )
                for s in page.get("studies", [])
            ]
            token = page.get("next_page_token")
            if not token or not page.get("studies"):
                break
    return trials[: spec.max_results], total
