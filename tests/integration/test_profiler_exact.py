"""Gate 3: the Profiler extracts every condition, medication, and lab present for 10 seeded patients.

Ground truth comes from direct HAPI queries, not from Profiler code. "Labs present" means
every (system, code) with a value in laboratory/vital-signs observations, including panel components.
"""

import os
from datetime import date
from typing import Any

import httpx
import pytest

from agents.fhir_reader import HttpFhirReader
from agents.workers.profiler import build_profile
from data.loaders.common import SOURCE_TAG_SYSTEM

pytestmark = pytest.mark.data

BASE = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")


def _all(client: httpx.Client, rtype: str, params: dict[str, str]) -> list[dict[str, Any]]:
    """Every match, following HAPI's own paging links (independent of the Profiler's date paging)."""
    out: list[dict[str, Any]] = []
    body = client.get(f"/{rtype}", params={**params, "_count": "500"}).json()
    while True:
        out += [e["resource"] for e in body.get("entry", [])]
        nxt = next((link["url"] for link in body.get("link", []) if link["relation"] == "next"), None)
        if not nxt:
            return out
        body = client.get(nxt.replace("http://hapi-fhir:8080/fhir", BASE)).json()


def _patients() -> list[str]:
    with httpx.Client(base_url=BASE, timeout=60, headers={"Accept": "application/fhir+json"}) as c:
        ids = sorted(p["id"] for p in _all(c, "Patient", {"_tag": f"{SOURCE_TAG_SYSTEM}|synthea", "_elements": "id"}))
    sample = ids[::20][:10]
    # Plus patients with more than 1000 MedicationRequests (found during label review).
    return sample + [p for p in _heavy_patients() if p not in sample]


def _heavy_patients() -> list[str]:
    import json
    from pathlib import Path

    labels = json.loads((Path(__file__).parents[2] / "data" / "handlabel" / "handlabel_v1.json").read_text())
    return sorted({p["patient_id"] for p in labels["pairs"] if p["id"] in {"hl-048", "hl-050"}})


def _truth(pid: str) -> dict[str, set[Any]]:
    with httpx.Client(base_url=BASE, timeout=120, headers={"Accept": "application/fhir+json"}) as c:
        conditions = {f"Condition/{r['id']}" for r in _all(c, "Condition", {"patient": pid})}
        meds = {f"MedicationRequest/{r['id']}" for r in _all(c, "MedicationRequest", {"patient": pid})}
        lab_codes: set[tuple[str, str]] = set()
        for cat in ("laboratory", "vital-signs"):
            for o in _all(c, "Observation", {"patient": pid, "category": cat}):
                for holder in [o, *o.get("component", [])]:
                    coding = (holder.get("code", {}).get("coding") or [{}])[0]
                    has_value = any(k in holder for k in ("valueQuantity", "valueCodeableConcept", "valueString"))
                    if coding.get("code") and has_value:
                        lab_codes.add((coding.get("system", ""), coding["code"]))
    return {"conditions": conditions, "meds": meds, "labs": lab_codes}


@pytest.mark.parametrize("pid", _patients())
async def test_profile_matches_record_exactly(pid: str) -> None:
    reader = HttpFhirReader(BASE)
    try:
        profile = await build_profile(reader, pid, as_of=date(2026, 11, 16))
    finally:
        await reader.aclose()
    truth = _truth(pid)
    assert not profile.truncated
    assert {c.resource_id for c in profile.conditions} == truth["conditions"]
    assert {m.resource_id for m in profile.medications} == truth["meds"]
    assert {(lab.system, lab.code) for lab in profile.labs} == truth["labs"]
    assert profile.age_years is not None
    assert profile.sex in {"male", "female"}
