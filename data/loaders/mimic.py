"""Load the MIMIC-IV Clinical Database Demo on FHIR (v2.1.0) into HAPI.

Open access, ODbL v1.0, de-identified. Files are verified against PhysioNet's
SHA256SUMS.txt. Resources are written with PUT <type>/<id> in dependency order,
so a rerun creates 0 resources.

The default "core" set covers what the Profiler needs (patients, encounters,
conditions, meds, labs, procedures). `full=True` adds the large ICU chart
events, medication administrations, dispenses, and microbiology files.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from data.loaders.common import EXTERNAL_DIR, USER_AGENT, ensure_file
from data.loaders.fhir import FhirClient, TransactionResult, add_source_tag, put_entry

log = logging.getLogger(__name__)

SOURCE = "mimic"
VERSION = "2.1.0"
BASE_URL = f"https://physionet.org/files/mimic-iv-fhir-demo/{VERSION}"
MIMIC_DIR = EXTERNAL_DIR / "mimic-iv-fhir-demo" / VERSION
EXPECTED_PATIENTS = 100

# Dependency order: referenced resources load before the resources that point to them.
CORE_FILES = [
    "MimicOrganization",
    "MimicLocation",
    "MimicPatient",
    "MimicMedication",
    "MimicMedicationMix",
    "MimicEncounter",
    "MimicEncounterED",
    "MimicEncounterICU",
    "MimicCondition",
    "MimicConditionED",
    # ED observations point to procedures (Observation.partOf), so procedures load first.
    "MimicProcedure",
    "MimicProcedureED",
    "MimicProcedureICU",
    "MimicSpecimen",
    "MimicSpecimenLab",
    "MimicObservationLabevents",
    "MimicObservationED",
    "MimicObservationVitalSignsED",
    "MimicMedicationRequest",
    "MimicMedicationStatementED",
]
EXTRA_FILES = [
    "MimicObservationMicroTest",
    "MimicObservationMicroOrg",
    "MimicObservationMicroSusc",
    "MimicMedicationDispense",
    "MimicMedicationDispenseED",
    "MimicMedicationAdministration",
    "MimicMedicationAdministrationICU",
    "MimicObservationDatetimeevents",
    "MimicObservationOutputevents",
    "MimicObservationChartevents",
]


@dataclass
class MimicLoadReport:
    files: dict[str, int] = field(default_factory=dict)  # file -> resources sent
    result: TransactionResult = field(default_factory=TransactionResult)
    patient_ids: list[str] = field(default_factory=list)


def fetch_checksums() -> dict[str, str]:
    """PhysioNet's published SHA256 sums, keyed by path (e.g. 'fhir/MimicPatient.ndjson.gz')."""
    sums_path = MIMIC_DIR / "SHA256SUMS.txt"
    if not sums_path.exists():
        sums_path.parent.mkdir(parents=True, exist_ok=True)
        resp = httpx.get(f"{BASE_URL}/SHA256SUMS.txt", headers={"User-Agent": USER_AGENT}, timeout=60)
        resp.raise_for_status()
        sums_path.write_text(resp.text)
    sums: dict[str, str] = {}
    for line in sums_path.read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            sums[name.strip()] = digest
    return sums


def ensure_files(names: list[str]) -> dict[str, Path]:
    sums = fetch_checksums()
    paths = {}
    for name in names:
        rel = f"fhir/{name}.ndjson.gz"
        paths[name] = ensure_file(f"{BASE_URL}/{rel}", MIMIC_DIR / rel, sha256=sums[rel])
    return paths


def read_ndjson(path: Path) -> Iterator[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _chunks(items: Iterator[dict[str, Any]], size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def load(client: FhirClient | None = None, full: bool = False, batch_size: int = 500) -> MimicLoadReport:
    names = CORE_FILES + (EXTRA_FILES if full else [])
    paths = ensure_files(names)
    report = MimicLoadReport()
    own_client = client is None
    client = client or FhirClient()
    try:
        for name in names:
            sent = 0
            for batch in _chunks(read_ndjson(paths[name]), batch_size):
                entries = []
                for resource in batch:
                    add_source_tag(resource, SOURCE)
                    entries.append(put_entry(resource))
                    if resource["resourceType"] == "Patient":
                        report.patient_ids.append(resource["id"])
                report.result.add(client.transaction(entries))
                sent += len(entries)
            report.files[name] = sent
            log.info("mimic %s: %d resources", name, sent)
        return report
    finally:
        if own_client:
            client.close()
