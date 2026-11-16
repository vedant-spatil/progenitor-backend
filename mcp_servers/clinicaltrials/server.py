"""ClinicalTrials.gov MCP server (API v2), with Redis cache, rate limit, and AACT fallback.

Tools: search_trials, get_trial, get_eligibility. Free-text search params pass the
PHI egress filter; NCT ids are validated by pattern.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError

from mcp_servers.clinicaltrials.aact import AactBackend, aact_from_env
from mcp_servers.common.http import OutboundClient, UpstreamError, UpstreamErrorKind, redis_cache_from_env
from mcp_servers.common.server import add_health_route, require, run, tool_errors

BASE_URL = "https://clinicaltrials.gov/api/v2"
DESTINATION = "clinicaltrials.gov"
NCT = r"NCT\d{8}"
STATUSES = {
    "RECRUITING",
    "NOT_YET_RECRUITING",
    "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
    "COMPLETED",
    "SUSPENDED",
    "TERMINATED",
    "WITHDRAWN",
}
PHASES = {"EARLY_PHASE1", "PHASE1", "PHASE2", "PHASE3", "PHASE4", "NA"}
SEARCH_FIELDS = "NCTId,BriefTitle,OverallStatus,Phase,Condition,MinimumAge,MaximumAge,Sex,LastUpdatePostDate"
# Fallback only on availability problems, never on bad input or PHI blocks.
FALLBACK_KINDS = {UpstreamErrorKind.TIMEOUT, UpstreamErrorKind.UNAVAILABLE, UpstreamErrorKind.HTTP_ERROR}


def summarize(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection", {})
    ident, status = p.get("identificationModule", {}), p.get("statusModule", {})
    elig = p.get("eligibilityModule", {})
    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "status": status.get("overallStatus"),
        "phases": p.get("designModule", {}).get("phases", []),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "minimum_age": elig.get("minimumAge"),
        "maximum_age": elig.get("maximumAge"),
        "sex": elig.get("sex"),
        "last_update": status.get("lastUpdatePostDateStruct", {}).get("date"),
    }


def eligibility(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection", {})
    m = p.get("eligibilityModule", {})
    return {
        "nct_id": p.get("identificationModule", {}).get("nctId"),
        "criteria_text": m.get("eligibilityCriteria", ""),
        "minimum_age": m.get("minimumAge"),
        "maximum_age": m.get("maximumAge"),
        "sex": m.get("sex"),
        "healthy_volunteers": m.get("healthyVolunteers"),
    }


def locations(study: dict[str, Any]) -> list[dict[str, Any]]:
    """Trial sites (public registry data, not patient data)."""
    locs = study.get("protocolSection", {}).get("contactsLocationsModule", {}).get("locations", [])
    return [{k: loc.get(k) for k in ("facility", "city", "state", "country", "status")} for loc in locs]


def build_server(client: OutboundClient | None = None, aact: AactBackend | None = None) -> FastMCP:
    client = client or OutboundClient(
        DESTINATION,
        BASE_URL,
        rate_per_min=float(os.environ.get("CTGOV_RATE_PER_MIN", "50")),
        cache=redis_cache_from_env(),
    )
    aact = aact if aact is not None else aact_from_env()
    mcp = FastMCP("clinicaltrials", instructions="Search and read ClinicalTrials.gov studies (API v2).")
    add_health_route(mcp)

    async def fetch_study(nct_id: str) -> dict[str, Any]:
        try:
            study: dict[str, Any] = await client.get(f"/studies/{nct_id}", {"format": "json"})
        except UpstreamError as exc:
            if exc.kind == UpstreamErrorKind.NOT_FOUND:
                raise ToolError(f"{nct_id} not found") from exc
            if aact and exc.kind in FALLBACK_KINDS:
                fallback = await aact.study(nct_id)
                if fallback is not None:
                    return fallback
            raise
        if not study.get("protocolSection"):
            raise UpstreamError(UpstreamErrorKind.MALFORMED, DESTINATION, "no protocolSection")
        return study

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def search_trials(
        conditions: str,
        terms: str = "",
        interventions: str = "",
        status: list[str] | None = None,
        phases: list[str] | None = None,
        location: str = "",
        page_size: int = 20,
        page_token: str | None = None,
    ) -> dict[str, Any]:
        """Search ClinicalTrials.gov. Use condition and drug terms only, never patient identifiers.

        status defaults to ["RECRUITING"]; phases e.g. ["PHASE2", "PHASE3"]; location is a
        state or country. Returns studies, total count, and next_page_token.
        """
        status = status or ["RECRUITING"]
        if bad := set(status) - STATUSES:
            raise ToolError(f"invalid status: {sorted(bad)}")
        if phases and (bad := set(phases) - PHASES):
            raise ToolError(f"invalid phases: {sorted(bad)}")
        if page_token is not None:
            require(r"[A-Za-z0-9_-]{1,200}", page_token, "page_token")
        params: dict[str, Any] = {
            "query.cond": conditions,
            "query.term": terms,
            "query.intr": interventions,
            "query.locn": location,
            "filter.overallStatus": ",".join(status),
            "filter.advanced": f"AREA[Phase]({' OR '.join(phases)})" if phases else None,
            "pageSize": max(1, min(page_size, 100)),
            "pageToken": page_token,
            "countTotal": "true",
            "fields": SEARCH_FIELDS,
            "format": "json",
        }
        try:
            data = await client.get(
                "/studies", params, free_text=("query.cond", "query.term", "query.intr", "query.locn")
            )
        except UpstreamError as exc:
            if aact and exc.kind in FALLBACK_KINDS:
                studies = await aact.search(conditions, status, page_size)
                # "next_page_token" is a paging cursor, not a credential.
                return {  # nosec B105
                    "studies": studies,
                    "total": len(studies),
                    "next_page_token": None,
                    "source": "aact",
                }
            raise
        if not isinstance(data, dict) or "studies" not in data:
            raise UpstreamError(UpstreamErrorKind.MALFORMED, DESTINATION, "no studies key")
        return {
            "studies": [summarize(s) for s in data["studies"]],
            "total": data.get("totalCount"),
            "next_page_token": data.get("nextPageToken"),
            "source": "ctgov",
        }

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def get_trial(nct_id: str) -> dict[str, Any]:
        """Summary and site locations for one study (e.g. NCT04280705)."""
        nct_id = require(NCT, nct_id.upper(), "nct_id")
        study = await fetch_study(nct_id)
        return {**summarize(study), "locations": locations(study)}

    @mcp.tool(annotations={"readOnlyHint": True, "openWorldHint": True})
    @tool_errors
    async def get_eligibility(nct_id: str) -> dict[str, Any]:
        """Eligibility criteria text plus age, sex, and healthy-volunteer limits for one study."""
        nct_id = require(NCT, nct_id.upper(), "nct_id")
        return eligibility(await fetch_study(nct_id))

    return mcp


if __name__ == "__main__":
    run(build_server(), default_port=8101)
