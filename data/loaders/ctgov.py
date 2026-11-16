"""Map ClinicalTrials.gov API v2 study JSON onto the Trial cache table.

Fetching happens in the clinicaltrials MCP server (Phase 2); this module only
turns a study document into a Trial row.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from django.utils import timezone

from apps.trials.models import Trial, TrialSource


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def trial_fields(study: dict[str, Any]) -> dict[str, Any]:
    p = study.get("protocolSection", {})
    ident = p.get("identificationModule", {})
    status = p.get("statusModule", {})
    elig = p.get("eligibilityModule", {})
    return {
        "nct_id": ident["nctId"],
        "title": ident.get("briefTitle") or ident.get("officialTitle", ""),
        "overall_status": status.get("overallStatus", ""),
        "phases": p.get("designModule", {}).get("phases", []),
        "conditions": p.get("conditionsModule", {}).get("conditions", []),
        "eligibility_text": elig.get("eligibilityCriteria", ""),
        "minimum_age": elig.get("minimumAge", ""),
        "maximum_age": elig.get("maximumAge", ""),
        "sex": elig.get("sex", ""),
        "healthy_volunteers": elig.get("healthyVolunteers"),
        "source_last_updated": _parse_date(status.get("lastUpdatePostDateStruct", {}).get("date")),
    }


def upsert_trial(
    study: dict[str, Any], source: str = TrialSource.CTGOV_API, fetched_at: datetime | None = None
) -> tuple[Trial, bool]:
    fields = trial_fields(study)
    nct_id = fields.pop("nct_id")
    return Trial.objects.update_or_create(
        nct_id=nct_id,
        defaults={**fields, "raw": study, "source": source, "fetched_at": fetched_at or timezone.now()},
    )
