"""Data inventory report for Gate 1: what is loaded where, with counts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from apps.patients.models import Patient
from apps.trials.models import Trial
from data.loaders import mimic, synthea
from data.loaders.common import BACKEND_DIR
from data.loaders.eval_data import EVAL_DIR, chia_stats, cohort_stats, load_manifest
from data.loaders.fhir import FhirClient
from data.synthea.generate import OUTPUT_DIR, load_cohort, patient_bundles

DEFAULT_REPORT = BACKEND_DIR / "docs" / "eval-reports" / "data-inventory.md"

FHIR_TYPES = [
    "Patient",
    "Encounter",
    "Condition",
    "Observation",
    "MedicationRequest",
    "Procedure",
    "Organization",
    "Practitioner",
]


def _fhir_section(client: FhirClient) -> list[str]:
    lines = [
        "## HAPI FHIR (local)",
        "",
        f"Server: `{client.base_url}`. Counts by source tag.",
        "",
        "| Resource | Synthea | MIMIC | Server total |",
        "|---|---:|---:|---:|",
    ]
    for rtype in FHIR_TYPES:
        s = client.count_by_source(rtype, synthea.SOURCE)
        m = client.count_by_source(rtype, mimic.SOURCE)
        lines.append(f"| {rtype} | {s:,} | {m:,} | {client.count(rtype):,} |")
    return lines


def _synthea_section() -> list[str]:
    cohort = load_cohort()
    lines = [
        "## Synthea cohort",
        "",
        f"Synthea {cohort['synthea_version']}, reference date {cohort['reference_date']}, "
        f"{cohort['years_of_history']} years of history, seeds in `data/synthea/cohort.json`.",
        "",
        "| Condition | Requested | Generated bundles |",
        "|---|---:|---:|",
    ]
    for cond in cohort["conditions"]:
        lines.append(f"| {cond['name']} | {cond['patients']} | {len(patient_bundles(OUTPUT_DIR / cond['name']))} |")
    return lines


def _app_section() -> list[str]:
    lines = ["## App tables", "", "| Table | Rows |", "|---|---:|"]
    for source in (synthea.SOURCE, mimic.SOURCE):
        lines.append(f"| patients_patient (source={source}) | {Patient.objects.filter(source=source).count():,} |")
    lines.append(f"| trials_trial | {Trial.objects.count():,} |")
    return lines


def _eval_section() -> list[str]:
    manifest = load_manifest()
    lines = [
        "## Eval datasets",
        "",
        f"TrialGPT commit `{manifest['trialgpt']['commit']}`; files verified by SHA256 (`data/eval_manifest.json`).",
        "",
        "| Cohort | Patients | Expected | Patients with qrels | Qrels rows | Trials | Labels (0/1/2) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for cohort, spec in manifest["trialgpt"]["cohorts"].items():
        c = cohort_stats(cohort, spec["expected_patients"])
        labels = " / ".join(f"{c.labels.get(k, 0):,}" for k in ("0", "1", "2"))
        lines.append(
            f"| {cohort} | {c.patients} | {c.expected_patients} | {c.qrels_patients} | "
            f"{c.qrels_rows:,} | {c.trials:,} | {labels} |"
        )
    chia = chia_stats(EVAL_DIR / "chia" / manifest["chia"]["file"], manifest["chia"]["expected_trials"])
    lines += [
        "",
        f"Chia ({manifest['chia']['license']}, DOI {manifest['chia']['doi']}): "
        f"{chia.trials:,} trials (expected {chia.expected_trials:,}), "
        f"{chia.entities:,} entity annotations, {chia.relations:,} relation annotations "
        "(with-scope release; paper reports 41,487 entities and 25,017 relations).",
        "",
        "Notes:",
        "- SIGIR: one of the 59 topics has no qrels in the TrialGPT release; ranking metrics "
        "average over topics with qrels.",
        "- TREC 2021/2022 full trial corpora are not downloaded. Phase 6 builds a frozen trial "
        "snapshot from the qrels NCT ids.",
        "- TrialGPT repo declares no license: files stay in `data/external/`, never committed.",
    ]
    return lines


def write_inventory(out: str | Path = DEFAULT_REPORT) -> Path:
    path = Path(out)
    with FhirClient() as client:
        sections = [
            [
                "# Data inventory",
                "",
                f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `python manage.py data_inventory`. "
                "Synthetic (Synthea) and de-identified (MIMIC-IV demo) data only.",
            ],
            _fhir_section(client),
            _synthea_section(),
            _app_section(),
            _eval_section(),
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n\n".join("\n".join(s) for s in sections) + "\n")
    return path
