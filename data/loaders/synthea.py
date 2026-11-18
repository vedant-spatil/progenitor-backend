"""Upload Synthea transaction bundles to HAPI, idempotently.

Synthea bundles POST resources with `urn:uuid:` fullUrls and refer to shared
Organization/Location/Practitioner resources by conditional URL
(`Practitioner?identifier=system|value`). POSTs would duplicate resources on a
rerun, so each entry becomes `PUT <type>/<id>` and every reference is rewritten
to a literal `<type>/<id>`. A second upload then creates 0 resources.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from data.loaders.fhir import FhirClient, TransactionResult, add_source_tag, put_entry
from data.synthea.generate import OUTPUT_DIR, load_cohort, patient_bundles

log = logging.getLogger(__name__)

SOURCE = "synthea"
CONDITIONAL_REF = re.compile(r"^(?P<type>[A-Za-z]+)\?identifier=(?P<ident>.+)$")
INFO_PREFIXES = ("hospitalInformation", "practitionerInformation")


class UnresolvedReferenceError(RuntimeError):
    pass


@dataclass
class SyntheaLoadReport:
    bundles: int = 0
    result: TransactionResult = field(default_factory=TransactionResult)
    patients: dict[str, list[str]] = field(default_factory=dict)  # condition -> patient ids


def identifier_index(bundles: Iterable[dict[str, Any]]) -> dict[tuple[str, str], str]:
    """Map (resourceType, 'system|value') -> 'Type/id' for every identifier in the bundles."""
    index: dict[tuple[str, str], str] = {}
    for bundle in bundles:
        for entry in bundle.get("entry", []):
            res = entry["resource"]
            for ident in res.get("identifier", []):
                if "system" in ident and "value" in ident:
                    key = (res["resourceType"], f"{ident['system']}|{ident['value']}")
                    index[key] = f"{res['resourceType']}/{res['id']}"
    return index


def _rewrite(node: Any, urn_map: dict[str, str], ident_index: dict[tuple[str, str], str]) -> Any:
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "reference" and isinstance(value, str):
                out[key] = _resolve(value, urn_map, ident_index)
            else:
                out[key] = _rewrite(value, urn_map, ident_index)
        return out
    if isinstance(node, list):
        return [_rewrite(v, urn_map, ident_index) for v in node]
    return node


def _resolve(ref: str, urn_map: dict[str, str], ident_index: dict[tuple[str, str], str]) -> str:
    if ref.startswith("urn:uuid:"):
        if ref not in urn_map:
            raise UnresolvedReferenceError(ref)
        return urn_map[ref]
    match = CONDITIONAL_REF.match(ref)
    if match:
        key = (match["type"], unquote(match["ident"]))
        if key not in ident_index:
            raise UnresolvedReferenceError(ref)
        return ident_index[key]
    return ref


def to_put_entries(bundle: dict[str, Any], ident_index: dict[tuple[str, str], str]) -> list[dict[str, Any]]:
    urn_map = {
        e["fullUrl"]: f"{e['resource']['resourceType']}/{e['resource']['id']}"
        for e in bundle.get("entry", [])
        if str(e.get("fullUrl", "")).startswith("urn:uuid:")
    }
    entries = []
    for entry in bundle.get("entry", []):
        resource = _rewrite(entry["resource"], urn_map, ident_index)
        add_source_tag(resource, SOURCE)
        entries.append(put_entry(resource))
    return entries


def _read(path: Path) -> dict[str, Any]:
    bundle: dict[str, Any] = json.loads(path.read_text())
    return bundle


def _patient_id(bundle: dict[str, Any]) -> str:
    for entry in bundle.get("entry", []):
        if entry["resource"]["resourceType"] == "Patient":
            return str(entry["resource"]["id"])
    raise ValueError("bundle has no Patient")


def load(client: FhirClient | None = None, workers: int = 1) -> SyntheaLoadReport:
    cohort = load_cohort()
    report = SyntheaLoadReport()
    own_client = client is None
    client = client or FhirClient()
    try:
        cond_dirs = [OUTPUT_DIR / c["name"] for c in cohort["conditions"]]
        info_files = sorted(
            p for d in cond_dirs for p in (d / "fhir").glob("*.json") if p.name.startswith(INFO_PREFIXES)
        )
        if not info_files:
            raise RuntimeError(f"no Synthea output under {OUTPUT_DIR}; run generate_synthea first")
        info_bundles = [_read(p) for p in info_files]
        ident_index = identifier_index(info_bundles)

        # Shared resources first (organizations, locations, practitioners): patients refer to them.
        for bundle in info_bundles:
            report.result.add(client.transaction(to_put_entries(bundle, ident_index)))
            report.bundles += 1

        jobs: list[tuple[str, Path]] = [
            (c["name"], p) for c, d in zip(cohort["conditions"], cond_dirs, strict=True) for p in patient_bundles(d)
        ]

        def upload(job: tuple[str, Path]) -> tuple[str, str, TransactionResult]:
            name, path = job
            bundle = _read(path)
            return name, _patient_id(bundle), client.transaction(to_put_entries(bundle, ident_index))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            for name, pid, result in pool.map(upload, jobs):
                report.result.add(result)
                report.bundles += 1
                report.patients.setdefault(name, []).append(pid)
        return report
    finally:
        if own_client:
            client.close()
