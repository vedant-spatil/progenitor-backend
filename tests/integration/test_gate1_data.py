"""Gate 1: data foundation.

Needs the compose stack and loaded data (`make data`). Runs against real HAPI and
Postgres; nothing is mocked.
"""

from collections.abc import Iterator

import pytest

from data.loaders import eval_data, mimic, synthea
from data.loaders.common import sha256_file
from data.loaders.fhir import FhirClient
from data.loaders.inventory import DEFAULT_REPORT

pytestmark = pytest.mark.data

IDEMPOTENCE_TYPES = ["Patient", "Encounter", "Condition", "Observation", "MedicationRequest", "Procedure"]


@pytest.fixture(scope="module")
def fhir() -> Iterator[FhirClient]:
    with FhirClient() as client:
        yield client


def test_hapi_has_at_least_200_synthea_patients(fhir: FhirClient) -> None:
    assert fhir.count_by_source("Patient", synthea.SOURCE) >= 200


def test_hapi_has_100_mimic_patients(fhir: FhirClient) -> None:
    assert fhir.count_by_source("Patient", mimic.SOURCE) == mimic.EXPECTED_PATIENTS


def test_synthea_cohort_covers_all_six_conditions(fhir: FhirClient) -> None:
    """Every condition run kept patients with that condition active (keep modules)."""
    from data.synthea.generate import OUTPUT_DIR, load_cohort, patient_bundles

    for cond in load_cohort()["conditions"]:
        assert len(patient_bundles(OUTPUT_DIR / cond["name"])) >= cond["patients"], cond["name"]
        codes = ",".join(f"http://snomed.info/sct|{c['code']}" for c in cond["codes"])
        with_condition = fhir.count(
            "Condition", {"code": codes, "_tag": f"https://progenitor.dev/fhir/source|{synthea.SOURCE}"}
        )
        assert with_condition >= cond["patients"], cond["name"]


@pytest.mark.django_db
def test_loaders_are_idempotent(fhir: FhirClient) -> None:
    """A second full run of both loaders creates 0 FHIR resources and changes no counts."""
    before = {(t, s): fhir.count_by_source(t, s) for t in IDEMPOTENCE_TYPES for s in (synthea.SOURCE, mimic.SOURCE)}

    synthea_report = synthea.load(client=fhir)
    mimic_report = mimic.load(client=fhir)

    assert synthea_report.result.created == 0, synthea_report.result.statuses
    assert mimic_report.result.created == 0, mimic_report.result.statuses
    after = {(t, s): fhir.count_by_source(t, s) for t in IDEMPOTENCE_TYPES for s in (synthea.SOURCE, mimic.SOURCE)}
    assert after == before


@pytest.mark.django_db
def test_patient_registration_is_idempotent() -> None:
    from apps.patients.models import Patient
    from data.loaders.registry import register_patients

    ids = [("p-1", "Synthea test 001"), ("p-2", "Synthea test 002")]
    assert register_patients("synthea", ids) == (2, 0)
    assert register_patients("synthea", ids) == (0, 2)
    assert Patient.objects.filter(source="synthea").count() == 2


class TestEvalData:
    def test_trialgpt_files_match_pinned_checksums(self) -> None:
        manifest = eval_data.load_manifest()
        for cohort, spec in manifest["trialgpt"]["cohorts"].items():
            for rel, digest in spec["files"].items():
                assert sha256_file(eval_data.trialgpt_dir(cohort) / rel) == digest, f"{cohort}/{rel}"

    @pytest.mark.parametrize(("cohort", "expected"), [("sigir", 59), ("trec_2021", 75), ("trec_2022", 50)])
    def test_trialgpt_patient_counts(self, cohort: str, expected: int) -> None:
        stats = eval_data.cohort_stats(cohort, expected)
        assert stats.patients == expected
        assert set(stats.labels) <= {"0", "1", "2"}

    def test_chia_checksum_and_1000_trials(self) -> None:
        manifest = eval_data.load_manifest()
        path = eval_data.EVAL_DIR / "chia" / manifest["chia"]["file"]
        assert sha256_file(path) == manifest["chia"]["sha256"]
        assert eval_data.chia_stats(path, 1000).trials == 1000


def test_data_inventory_recorded() -> None:
    text = DEFAULT_REPORT.read_text()
    assert "# Data inventory" in text
    for heading in ("## HAPI FHIR", "## Synthea cohort", "## App tables", "## Eval datasets"):
        assert heading in text
