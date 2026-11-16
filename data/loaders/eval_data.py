"""Download and verify eval datasets: TrialGPT cohorts (SIGIR, TREC 2021, TREC 2022) and Chia.

Everything is pinned in data/eval_manifest.json (commit, URLs, checksums,
expected counts) and lands in data/external/eval/ (gitignored).

Not downloaded here: the full TREC 2021/2022 trial corpora (large, separate
license). The Phase 6 eval builds its frozen trial snapshot from the qrels' NCT
ids instead.
"""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from data.loaders.common import DATA_DIR, EXTERNAL_DIR, ensure_file, sha256_file

MANIFEST_PATH = DATA_DIR / "eval_manifest.json"
EVAL_DIR = EXTERNAL_DIR / "eval"


@dataclass
class CohortStats:
    name: str
    patients: int
    expected_patients: int
    qrels_rows: int
    qrels_patients: int
    trials: int
    labels: dict[str, int]


@dataclass
class ChiaStats:
    trials: int
    expected_trials: int
    entities: int
    relations: int


@dataclass
class EvalDataReport:
    cohorts: list[CohortStats] = field(default_factory=list)
    chia: ChiaStats | None = None


def load_manifest() -> dict[str, Any]:
    manifest: dict[str, Any] = json.loads(MANIFEST_PATH.read_text())
    return manifest


def trialgpt_dir(cohort: str) -> Path:
    return EVAL_DIR / "trialgpt" / cohort


def fetch_trialgpt(manifest: dict[str, Any]) -> None:
    tg = manifest["trialgpt"]
    base = f"https://raw.githubusercontent.com/{tg['repo']}/{tg['commit']}/dataset"
    for cohort, spec in tg["cohorts"].items():
        for rel, digest in spec["files"].items():
            ensure_file(f"{base}/{cohort}/{rel}", trialgpt_dir(cohort) / rel, sha256=digest)


def fetch_chia(manifest: dict[str, Any]) -> Path:
    spec = manifest["chia"]
    path = ensure_file(spec["url"], EVAL_DIR / "chia" / spec["file"], md5=spec["md5"])
    if sha256_file(path) != spec["sha256"]:
        raise RuntimeError("chia zip sha256 mismatch")
    return path


def cohort_stats(cohort: str, expected: int) -> CohortStats:
    root = trialgpt_dir(cohort)
    with (root / "queries.jsonl").open() as fh:
        patients = [json.loads(line)["_id"] for line in fh if line.strip()]
    with (root / "qrels" / "test.tsv").open() as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    labels: dict[str, int] = {}
    for r in rows:
        labels[r["score"]] = labels.get(r["score"], 0) + 1
    return CohortStats(
        name=cohort,
        patients=len(set(patients)),
        expected_patients=expected,
        qrels_rows=len(rows),
        qrels_patients=len({r["query-id"] for r in rows}),
        trials=len({r["corpus-id"] for r in rows}),
        labels=dict(sorted(labels.items())),
    )


def chia_stats(path: Path, expected: int) -> ChiaStats:
    trials: set[str] = set()
    entities = relations = 0
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            stem = Path(name).name
            if stem.startswith("NCT") and stem.endswith((".txt", ".ann")):
                trials.add(stem.split("_")[0])
            if stem.endswith(".ann"):
                for line in zf.read(name).decode("utf-8", errors="replace").splitlines():
                    if line.startswith("T"):
                        entities += 1
                    elif line.startswith("R"):
                        relations += 1
    return ChiaStats(trials=len(trials), expected_trials=expected, entities=entities, relations=relations)


def fetch_all() -> EvalDataReport:
    manifest = load_manifest()
    fetch_trialgpt(manifest)
    chia_zip = fetch_chia(manifest)
    report = EvalDataReport()
    for cohort, spec in manifest["trialgpt"]["cohorts"].items():
        report.cohorts.append(cohort_stats(cohort, spec["expected_patients"]))
    report.chia = chia_stats(chia_zip, manifest["chia"]["expected_trials"])
    return report
