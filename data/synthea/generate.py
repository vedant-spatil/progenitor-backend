"""Generate the Synthea dev cohort (200 patients across 6 conditions).

Downloads the pinned Synthea jar (checksum verified), writes one keep module per
condition, and runs Synthea once per condition into
data/external/synthea/output/<condition>/fhir/. Output is synthetic.

A condition whose output already holds the requested number of patient bundles
is skipped, so reruns are cheap. Use force=True to regenerate.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess  # nosec B404
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from data.loaders.common import EXTERNAL_DIR, download, sha256_file

log = logging.getLogger(__name__)

COHORT_FILE = Path(__file__).with_name("cohort.json")
SYNTHEA_DIR = EXTERNAL_DIR / "synthea"
OUTPUT_DIR = SYNTHEA_DIR / "output"


@dataclass(frozen=True)
class ConditionResult:
    name: str
    patients: int
    skipped: bool


def load_cohort() -> dict[str, Any]:
    cohort: dict[str, Any] = json.loads(COHORT_FILE.read_text())
    return cohort


def patient_bundles(condition_dir: Path) -> list[Path]:
    """Patient bundles only (Synthea also writes hospital/practitioner info bundles)."""
    fhir_dir = condition_dir / "fhir"
    if not fhir_dir.is_dir():
        return []
    return sorted(
        p
        for p in fhir_dir.glob("*.json")
        if not p.name.startswith(("hospitalInformation", "practitionerInformation"))
    )


def ensure_jar(cohort: dict[str, Any]) -> Path:
    jar = EXTERNAL_DIR / "tools" / f"synthea-{cohort['synthea_version']}.jar"
    if not jar.exists():
        download(cohort["synthea_jar_url"], jar)
    actual = sha256_file(jar)
    if actual != cohort["synthea_jar_sha256"]:
        raise RuntimeError(f"Synthea jar checksum mismatch: {actual}")
    return jar


def keep_module(codes: list[dict[str, str]]) -> dict[str, Any]:
    """Synthea keep module: keep the patient if any of `codes` is an active condition."""
    return {
        "name": "Progenitor keep module",
        "gmf_version": 2,
        "states": {
            "Initial": {
                "type": "Initial",
                "conditional_transition": [
                    {
                        "transition": "Keep",
                        "condition": {"condition_type": "Active Condition", "codes": codes},
                    },
                    {"transition": "Terminal"},
                ],
            },
            "Terminal": {"type": "Terminal"},
            "Keep": {"type": "Terminal"},
        },
    }


def synthea_command(
    jar: Path, cohort: dict[str, Any], cond: dict[str, Any], out: Path, keep: Path
) -> list[str]:
    cmd = [
        "java",
        "-jar",
        str(jar),
        "-s",
        str(cond["seed"]),
        "-cs",
        str(cond["seed"]),
        "-p",
        str(cond["patients"]),
        "-r",
        cohort["reference_date"],
        "-a",
        cond["age"],
        "-k",
        str(keep),
        f"--exporter.baseDirectory={out}/",
        f"--exporter.years_of_history={cohort['years_of_history']}",
        "--exporter.fhir.export=true",
        "--exporter.fhir.transaction_bundle=true",
        "--exporter.hospital.fhir.export=true",
        "--exporter.practitioner.fhir.export=true",
        "--exporter.csv.export=false",
        "--exporter.ccda.export=false",
        "--exporter.text.export=false",
        "--generate.only_alive_patients=true",
        # Rare conditions (lung cancer) need many tries per kept patient; default 1000 is too low.
        f"--generate.max_attempts_to_keep_patient={cohort.get('max_attempts_to_keep_patient', 1000)}",
    ]
    if cond.get("gender"):
        cmd += ["-g", cond["gender"]]
    cmd.append(cohort["state"])
    return cmd


def generate(force: bool = False, only: list[str] | None = None) -> list[ConditionResult]:
    cohort = load_cohort()
    jar = ensure_jar(cohort)
    keep_dir = SYNTHEA_DIR / "keep"
    keep_dir.mkdir(parents=True, exist_ok=True)
    results: list[ConditionResult] = []

    for cond in cohort["conditions"]:
        name = cond["name"]
        if only and name not in only:
            continue
        out = OUTPUT_DIR / name
        existing = patient_bundles(out)
        if len(existing) >= cond["patients"] and not force:
            log.info("synthea %s: %d bundles present, skipping", name, len(existing))
            results.append(ConditionResult(name, len(existing), skipped=True))
            continue
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        keep = keep_dir / f"keep_{name}.json"
        keep.write_text(json.dumps(keep_module(cond["codes"]), indent=2))

        log.info("synthea %s: generating %d patients", name, cond["patients"])
        subprocess.run(  # noqa: S603  # nosec B603
            synthea_command(jar, cohort, cond, out, keep),
            cwd=SYNTHEA_DIR,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        produced = patient_bundles(out)
        if len(produced) < cond["patients"]:
            raise RuntimeError(f"synthea {name}: expected {cond['patients']} bundles, got {len(produced)}")
        results.append(ConditionResult(name, len(produced), skipped=False))
    return results
