"""Profiler: FHIR -> PatientProfile, deterministic, no identifiers."""

from datetime import date
from typing import Any

from agents.schemas import PatientProfile
from agents.workers.profiler import PAGE, age_years, build_profile

PATIENT = {
    "resourceType": "Patient",
    "id": "p1",
    "name": [{"family": "Okafor", "given": ["Adaeze"]}],
    "birthDate": "1961-07-19",
    "gender": "female",
    "address": [{"city": "Somerville"}],
}
CONDITIONS = [
    {
        "resourceType": "Condition",
        "id": "c1",
        "code": {"coding": [{"system": "http://snomed.info/sct", "code": "44054006", "display": "Diabetes"}]},
        "clinicalStatus": {"coding": [{"code": "active"}]},
        "onsetDateTime": "2015-03-01T10:00:00Z",
    }
]
MEDS = [
    {
        "resourceType": "MedicationRequest",
        "id": "m1",
        "status": "active",
        "authoredOn": "2024-01-02",
        "medicationCodeableConcept": {"coding": [{"system": "rxnorm", "code": "860975", "display": "metformin"}]},
    },
    {
        "resourceType": "MedicationRequest",
        "id": "m2",
        "status": "stopped",
        "medicationReference": {"reference": "Medication/x"},
    },
]
MED_INCLUDED = [
    {
        "resourceType": "Medication",
        "id": "x",
        "code": {"coding": [{"system": "rxnorm", "code": "5856", "display": "insulin"}]},
    }
]


def obs(oid: str, code: str, value: float, when: str, category: str = "laboratory", **extra: Any) -> dict[str, Any]:
    return {
        "resourceType": "Observation",
        "id": oid,
        "category": [{"coding": [{"code": category}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": code, "display": code}]},
        "valueQuantity": {"value": value, "unit": "%"},
        "effectiveDateTime": when,
        **extra,
    }


LABS = [obs("o-old", "4548-4", 6.1, "2020-01-01"), obs("o-new", "4548-4", 8.2, "2025-06-01")]
VITALS = [
    {
        "resourceType": "Observation",
        "id": "bp1",
        "category": [{"coding": [{"code": "vital-signs"}]}],
        "code": {"coding": [{"system": "http://loinc.org", "code": "85354-9", "display": "BP panel"}]},
        "effectiveDateTime": "2025-06-01",
        "component": [
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "8480-6"}]},
                "valueQuantity": {"value": 142, "unit": "mm[Hg]"},
            },
            {
                "code": {"coding": [{"system": "http://loinc.org", "code": "8462-4"}]},
                "valueQuantity": {"value": 91, "unit": "mm[Hg]"},
            },
        ],
    }
]


class FakeReader:
    def __init__(self, sizes: dict[str, int] | None = None) -> None:
        self.searches: list[tuple[str, dict[str, str]]] = []
        self.sizes = sizes or {}

    async def read(self, resource_type: str, resource_id: str) -> dict[str, Any]:
        return PATIENT

    async def search(
        self, resource_type: str, params: dict[str, str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        self.searches.append((resource_type, params))
        if resource_type == "Condition":
            return CONDITIONS, []
        if resource_type == "MedicationRequest":
            return MEDS, MED_INCLUDED
        if params.get("category") == "vital-signs":
            return VITALS, []
        return [{"id": str(i), "resourceType": "Observation"} for i in range(self.sizes.get("lab", 0))] or LABS[
            ::-1
        ], []


async def profile(reader: FakeReader | None = None, **kw: Any) -> PatientProfile:
    return await build_profile(reader or FakeReader(), "p1", as_of=date(2026, 11, 1), **kw)


async def test_demographics_without_identifiers() -> None:
    p = await profile()
    assert (p.patient_ref, p.age_years, p.sex) == ("Patient/p1", 65, "female")
    dumped = p.model_dump_json()
    for leaked in ("Okafor", "Adaeze", "1961-07-19", "Somerville"):
        assert leaked not in dumped


def test_age_cap_at_90() -> None:
    assert age_years("1920-01-01", date(2026, 1, 1)) == 90
    assert age_years("2000-12-31", date(2026, 12, 30)) == 25
    assert age_years(None, date(2026, 1, 1)) is None


async def test_latest_lab_per_code_and_panel_components() -> None:
    p = await profile()
    labs = {lab.code: lab for lab in p.labs}
    assert labs["4548-4"].value == 8.2
    assert labs["4548-4"].resource_id == "Observation/o-new"
    assert labs["8480-6"].value == 142
    assert labs["8480-6"].resource_id == "Observation/bp1"
    assert "85354-9" not in labs  # panel has no own value


async def test_conditions_and_medications_with_included_medication() -> None:
    p = await profile()
    assert p.conditions[0].resource_id == "Condition/c1"
    assert p.conditions[0].onset == date(2015, 3, 1)
    meds = {m.resource_id: m for m in p.medications}
    assert meds["MedicationRequest/m1"].code == "860975"
    assert meds["MedicationRequest/m2"].display == "insulin"


async def test_resource_ids_cover_every_fact() -> None:
    p = await profile()
    assert p.resource_ids() == {
        "Patient/p1",
        "Condition/c1",
        "MedicationRequest/m1",
        "MedicationRequest/m2",
        "Observation/o-new",
        "Observation/bp1",
    }


async def test_loinc_filter_and_resource_types() -> None:
    reader = FakeReader()
    await profile(reader, resource_types=["Observation"], loinc_codes=["4548-4"])
    assert [s[0] for s in reader.searches] == ["Observation"]
    assert reader.searches[0][1]["code"] == "http://loinc.org|4548-4"


async def test_truncated_flag_at_page_limit() -> None:
    p = await profile(FakeReader(sizes={"lab": PAGE}))
    assert p.truncated


class PagingReader(FakeReader):
    """Serves 1500 lab observations, PAGE at a time, honoring date=le<ts>."""

    def __init__(self) -> None:
        super().__init__()
        self.all = [obs(f"o{i}", f"c{i % 40}", 1.0, f"2025-01-01T00:{i // 60:02d}:{i % 60:02d}Z") for i in range(1500)]
        self.all.sort(key=lambda o: o["effectiveDateTime"], reverse=True)

    async def search(
        self, resource_type: str, params: dict[str, str]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if resource_type != "Observation" or params.get("category") != "laboratory":
            return [], []
        self.searches.append((resource_type, params))
        rows = self.all
        if (d := params.get("date", "")).startswith("le"):
            rows = [o for o in rows if o["effectiveDateTime"] <= d[2:]]
        return rows[:PAGE], []


async def test_date_paging_reads_past_page_limit() -> None:
    reader = PagingReader()
    p = await profile(reader)
    assert not p.truncated
    assert len(p.labs) == 40
    assert len(reader.searches) == 2
    assert reader.searches[1][1]["date"].startswith("le")
