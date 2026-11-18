"""Profiler: FHIR record -> PatientProfile. Deterministic (no LLM): every fact keeps its resource id.

Identifiers are dropped here: the profile has age in years (capped at 90) and
sex, never names, birth date, address, or MRN.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import date
from typing import Any

from agents.fhir_reader import FhirReader, Resource
from agents.schemas import ConditionFact, LabFact, MedicationFact, PatientProfile
from agents.telemetry import timed

PAGE = 1000  # HAPI returns up to 1000 resources per page
OBS_CATEGORIES = ("laboratory", "vital-signs")
ALL_TYPES = ("Condition", "MedicationRequest", "Observation")


def _date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value[:10])
    except ValueError:
        return None


def age_years(birth_date: str | None, as_of: date) -> int | None:
    born = _date(birth_date)
    if born is None:
        return None
    years = as_of.year - born.year - ((as_of.month, as_of.day) < (born.month, born.day))
    return min(years, 90)


def _coding(concept: dict[str, Any] | None) -> tuple[str, str, str]:
    concept = concept or {}
    codings = concept.get("coding") or [{}]
    first = codings[0]
    return first.get("system", ""), first.get("code", ""), first.get("display") or concept.get("text", "")


def _status(concept: dict[str, Any] | None) -> str:
    return _coding(concept)[1]


def condition_facts(resources: Iterable[Resource]) -> list[ConditionFact]:
    facts = []
    for r in resources:
        system, code, display = _coding(r.get("code"))
        facts.append(
            ConditionFact(
                resource_id=f"Condition/{r['id']}",
                system=system,
                code=code,
                display=display,
                clinical_status=_status(r.get("clinicalStatus")),
                onset=_date(r.get("onsetDateTime") or (r.get("onsetPeriod") or {}).get("start")),
                abatement=_date(r.get("abatementDateTime")),
            )
        )
    return facts


def medication_facts(requests: Iterable[Resource], included: Iterable[Resource]) -> list[MedicationFact]:
    meds = {f"Medication/{m['id']}": m for m in included if m.get("resourceType") == "Medication"}
    facts = []
    for r in requests:
        concept = r.get("medicationCodeableConcept")
        if concept is None and (ref := (r.get("medicationReference") or {}).get("reference")):
            concept = (meds.get(ref) or {}).get("code")
        system, code, display = _coding(concept)
        facts.append(
            MedicationFact(
                resource_id=f"MedicationRequest/{r['id']}",
                system=system,
                code=code,
                display=display,
                status=r.get("status", ""),
                authored_on=_date(r.get("authoredOn")),
            )
        )
    return facts


def _value(obs: Resource) -> tuple[float | None, str, str]:
    if q := obs.get("valueQuantity"):
        return q.get("value"), "", q.get("unit") or q.get("code", "")
    if cc := obs.get("valueCodeableConcept"):
        return None, _coding(cc)[2], ""
    if (s := obs.get("valueString")) is not None:
        return None, str(s), ""
    return None, "", ""


def lab_facts(observations: Iterable[Resource]) -> list[LabFact]:
    """Latest value per (system, code). Observations must arrive newest first.

    Panel components (e.g. blood pressure) become separate facts that cite the panel resource.
    """
    latest: dict[tuple[str, str], LabFact] = {}
    for obs in observations:
        ref = f"Observation/{obs['id']}"
        effective = _date(obs.get("effectiveDateTime") or (obs.get("effectivePeriod") or {}).get("start"))
        category = next(
            (c.get("code", "") for cat in obs.get("category", []) or [] for c in cat.get("coding", []) or []),
            "",
        )
        parts: list[tuple[dict[str, Any] | None, Resource]] = [(obs.get("code"), obs)]
        parts += [(comp.get("code"), comp) for comp in obs.get("component", []) or []]
        for concept, holder in parts:
            system, code, display = _coding(concept)
            value, text, unit = _value(holder)
            if not code or (value is None and not text):
                continue
            key = (system, code)
            if key not in latest:
                latest[key] = LabFact(
                    resource_id=ref,
                    system=system,
                    code=code,
                    display=display,
                    value=value,
                    value_text=text,
                    unit=unit,
                    effective=effective,
                    category=category,
                )
    return list(latest.values())


MAX_PAGES = 20


def _effective(o: Resource) -> str:
    return str(o.get("effectiveDateTime") or (o.get("effectivePeriod") or {}).get("start") or "")


async def _search_all_by_date(
    reader: FhirReader,
    resource_type: str,
    params: dict[str, str],
    *,
    date_param: str = "date",
    date_of: Callable[[Resource], str] = _effective,
) -> tuple[list[Resource], list[Resource], bool]:
    """All matches, newest first, plus included resources.

    The FHIR MCP tool cannot follow paging links, so pages are walked by date: each
    request asks for <date_param> <= oldest seen so far; ids already seen are skipped.
    `params` must sort newest first. Returns (matches, included, truncated).
    """
    seen: dict[str, Resource] = {}
    included: dict[str, Resource] = {}
    query = dict(params)
    for _ in range(MAX_PAGES):
        found, inc = await reader.search(resource_type, query)
        included.update((f"{r.get('resourceType')}/{r.get('id')}", r) for r in inc)
        new = [r for r in found if r["id"] not in seen]
        seen.update((r["id"], r) for r in new)
        if len(found) < PAGE:
            return list(seen.values()), list(included.values()), False
        oldest = min((date_of(r) for r in found if date_of(r)), default="")
        if not new or not oldest:
            break  # a full page with nothing new: more than PAGE resources share one timestamp
        query = {**params, date_param: f"le{oldest}"}
    return list(seen.values()), list(included.values()), True


async def build_profile(
    reader: FhirReader,
    patient_id: str,
    *,
    resource_types: Iterable[str] = ALL_TYPES,
    loinc_codes: Iterable[str] = (),
    as_of: date | None = None,
) -> PatientProfile:
    """Read one patient's record. `loinc_codes` narrows Observation fetches (supervisor 'fetch_patient_data')."""
    as_of = as_of or date.today()
    types = set(resource_types)
    with timed("worker", worker="profiler"):
        patient = await reader.read("Patient", patient_id)
        profile = PatientProfile(
            patient_ref=f"Patient/{patient_id}",
            age_years=age_years(patient.get("birthDate"), as_of),
            sex=patient.get("gender", ""),
        )
        base = {"patient": patient_id, "_count": str(PAGE)}
        if "Condition" in types:
            found, _ = await reader.search("Condition", base)
            profile.conditions = condition_facts(found)
            profile.truncated |= len(found) >= PAGE
        if "MedicationRequest" in types:
            found, included, truncated = await _search_all_by_date(
                reader,
                "MedicationRequest",
                {**base, "_include": "MedicationRequest:medication", "_sort": "-authoredon"},
                date_param="authoredon",
                date_of=lambda r: str(r.get("authoredOn") or ""),
            )
            profile.medications = medication_facts(found, included)
            profile.truncated |= truncated
        if "Observation" in types:
            obs: list[Resource] = []
            codes = ",".join(f"http://loinc.org|{c}" for c in loinc_codes)
            for category in OBS_CATEGORIES if not codes else ("",):
                params = {**base, "_sort": "-date"}
                if category:
                    params["category"] = category
                if codes:
                    params["code"] = codes
                found, _, truncated = await _search_all_by_date(reader, "Observation", params)
                obs += found
                profile.truncated |= truncated
            obs.sort(key=_effective, reverse=True)
            profile.labs = lab_facts(obs)
    return profile
