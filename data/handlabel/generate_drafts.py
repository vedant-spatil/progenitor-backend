"""Draft the 50 hand-labelled patient-criterion pairs (Gate 3) from Synthea data in HAPI.

Truth is computed straight from raw FHIR resources (not via the Profiler), then
reviewed by a person before the file counts (status "draft" -> "reviewed").

Label policy (matches the system's citation rule):
- met / not_met need a resource id that shows it (lab value, condition, prescription, Patient for age/sex).
- When the record has nothing on the criterion, the label is "unknown": absence of a
  record is not evidence.

    uv run python -m data.handlabel.generate_drafts
"""

from __future__ import annotations

import csv
import json
import os
import random
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from data.loaders.common import SOURCE_TAG_SYSTEM

OUT_DIR = Path(__file__).parent
AS_OF = date(2026, 11, 16)
SEED = 3
N_PAIRS = 50
TARGET = {"met": 20, "not_met": 18, "unknown": 12}


@dataclass(frozen=True)
class LabRule:
    text: str
    code: str
    op: str
    threshold: float
    component_of: str | None = None  # panel code when the value is a component
    alt_codes: tuple[str, ...] = ()  # equivalent codes (Synthea uses more than one for some labs)


LAB_RULES = [
    LabRule("HbA1c of 7.0% or higher on the most recent measurement", "4548-4", ">=", 7.0),
    LabRule("Most recent eGFR below 60 mL/min/1.73 m2", "33914-3", "<", 60),
    LabRule("Body mass index of 30 kg/m2 or higher", "39156-5", ">=", 30),
    # Synthea codes creatinine as 2160-0 (serum/plasma) or 38483-4 (blood) depending on the module.
    LabRule("Most recent serum creatinine above 1.5 mg/dL", "2160-0", ">", 1.5, alt_codes=("38483-4",)),
    LabRule("Most recent LDL cholesterol of 100 mg/dL or higher", "18262-6", ">=", 100),
    LabRule("Platelet count of at least 100 x10^3/uL on the latest blood count", "777-3", ">=", 100),
    LabRule("Systolic blood pressure of 140 mmHg or higher at the most recent visit", "8480-6", ">=", 140, "85354-9"),
]
CONDITION_RULES = {
    "Active diagnosis of type 2 diabetes mellitus": {"44054006"},
    "Active diagnosis of essential hypertension": {"59621000"},
    "Active diagnosis of chronic kidney disease (any stage)": {
        "431855005",
        "431856006",
        "433144002",
        "431857002",
        "46177005",
        "127013003",
    },
    "Active diagnosis of asthma": {"195967001", "233678006"},
}
MED_RULES = {"Currently prescribed metformin": "metformin", "Currently prescribed insulin": "insulin"}
UNKNOWN_ONLY = ["Negative serum pregnancy test within 14 days before enrollment", "ECOG performance status 0 or 1"]


def _cmp(value: float, op: str, threshold: float) -> bool:
    return {">=": value >= threshold, ">": value > threshold, "<": value < threshold, "<=": value <= threshold}[op]


class Fhir:
    def __init__(self, base: str) -> None:
        self.http = httpx.Client(base_url=base, timeout=120, headers={"Accept": "application/fhir+json"})

    def search(self, rtype: str, params: dict[str, str]) -> list[dict[str, Any]]:
        """Every match: follows HAPI paging links (some patients have >1000 records of a type)."""
        out: list[dict[str, Any]] = []
        body = self.http.get(f"/{rtype}", params={**params, "_count": "500"}).json()
        while True:
            out += [e["resource"] for e in body.get("entry", [])]
            nxt = next((link["url"] for link in body.get("link", []) if link["relation"] == "next"), None)
            if not nxt:
                return out
            body = self.http.get(nxt.replace("http://hapi-fhir:8080/fhir", str(self.http.base_url).rstrip("/"))).json()


def latest_value(obs: list[dict[str, Any]], rule: LabRule) -> tuple[float, str] | None:
    rows = []
    for o in obs:
        when = o.get("effectiveDateTime") or ""
        holders = [o] if not rule.component_of else o.get("component", [])
        for h in holders:
            codes = {c.get("code") for c in h.get("code", {}).get("coding", [])}
            if ({rule.code, *rule.alt_codes} & codes) and (q := h.get("valueQuantity")) and q.get("value") is not None:
                rows.append((when, float(q["value"]), f"Observation/{o['id']}"))
    if not rows:
        return None
    rows.sort(reverse=True)
    return rows[0][1], rows[0][2]


def age(birth: str) -> int:
    b = date.fromisoformat(birth)
    return AS_OF.year - b.year - ((AS_OF.month, AS_OF.day) < (b.month, b.day))


def candidates_for(fhir: Fhir, pid: str) -> list[dict[str, Any]]:
    patient = fhir.http.get(f"/Patient/{pid}").json()
    pref = f"Patient/{pid}"
    out: list[dict[str, Any]] = []

    def add(text: str, status: str, evidence: list[str], why: str, family: str) -> None:
        out.append(
            {
                "criterion": {"kind": "inclusion", "text": text},
                "expected_status": status,
                "expected_evidence": evidence,
                "rationale": why,
                "family": family,
            }
        )

    years = age(patient["birthDate"])
    add("Age 65 years or older", "met" if years >= 65 else "not_met", [pref], f"age {years}", "age")
    add("Age between 18 and 75 years", "met" if 18 <= years <= 75 else "not_met", [pref], f"age {years}", "age")
    add(
        "Female participants only",
        "met" if patient.get("gender") == "female" else "not_met",
        [pref],
        f"sex {patient.get('gender')}",
        "sex",
    )

    labs = fhir.search("Observation", {"patient": pid, "category": "laboratory"})
    vitals = fhir.search("Observation", {"patient": pid, "category": "vital-signs"})
    for rule in LAB_RULES:
        found = latest_value(vitals if rule.code in ("39156-5", "8480-6") else labs, rule)
        if found is None:
            add(rule.text, "unknown", [], f"no {rule.code} on record", "lab")
        else:
            value, ref = found
            add(
                rule.text,
                "met" if _cmp(value, rule.op, rule.threshold) else "not_met",
                [ref],
                f"latest {rule.code} = {value:g}",
                "lab",
            )

    conditions = fhir.search("Condition", {"patient": pid})
    for text, codes in CONDITION_RULES.items():
        hits = [c for c in conditions if {x.get("code") for x in c["code"].get("coding", [])} & codes]
        active = [c for c in hits if c.get("clinicalStatus", {}).get("coding", [{}])[0].get("code") == "active"]
        if active:
            add(text, "met", [f"Condition/{active[0]['id']}"], "active condition", "condition")
        elif hits:
            add(text, "not_met", [f"Condition/{hits[0]['id']}"], "condition resolved, not active", "condition")
        else:
            add(text, "unknown", [], "no matching condition on record", "condition")

    meds = fhir.search("MedicationRequest", {"patient": pid})
    for text, word in MED_RULES.items():
        hits = [m for m in meds if word in json.dumps(m.get("medicationCodeableConcept", {})).lower()]
        active = [m for m in hits if m.get("status") == "active"]
        if active:
            add(text, "met", [f"MedicationRequest/{active[0]['id']}"], "active prescription", "medication")
        elif hits:
            add(
                text,
                "not_met",
                [f"MedicationRequest/{hits[0]['id']}"],
                f"prescription {hits[0].get('status')}",
                "medication",
            )
    for text in UNKNOWN_ONLY:
        if "pregnancy" in text.lower() and patient.get("gender") == "male":
            add(text, "not_applicable", [pref], "pregnancy rule; patient is male", "unrecorded")
        else:
            add(text, "unknown", [], "not recorded in Synthea data", "unrecorded")
    return out


def main() -> None:
    fhir = Fhir(os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir"))
    tag = f"{SOURCE_TAG_SYSTEM}|synthea"
    pids = sorted(p["id"] for p in fhir.search("Patient", {"_tag": tag, "_elements": "id"}))
    rng = random.Random(SEED)  # noqa: S311  # nosec B311 - reproducible sampling, not security
    rng.shuffle(pids)

    need = dict(TARGET)
    families_used: dict[str, int] = {}
    pairs: list[dict[str, Any]] = []
    for pid in pids:
        if len(pairs) >= N_PAIRS:
            break
        cands = candidates_for(fhir, pid)
        rng.shuffle(cands)
        taken = 0
        for c in cands:
            status, fam = c["expected_status"], c["family"]
            # Spread across statuses and criterion families; at most 2 pairs per patient.
            if need[status] <= 0 or families_used.get(fam + status, 0) >= 5 or taken >= 2:
                continue
            need[status] -= 1
            families_used[fam + status] = families_used.get(fam + status, 0) + 1
            taken += 1
            pairs.append({"id": f"hl-{len(pairs) + 1:03d}", "patient_id": pid, **c})
            if len(pairs) >= N_PAIRS:
                break

    doc = {
        "version": 1,
        "status": "draft",
        "reviewed_by": None,
        "as_of": AS_OF.isoformat(),
        "generated_by": "data/handlabel/generate_drafts.py (computed from HAPI; needs human review)",
        "label_policy": __doc__.split("Label policy")[1].split("uv run")[0].strip(),
        "pairs": pairs,
    }
    (OUT_DIR / "handlabel_v1.json").write_text(json.dumps(doc, indent=2) + "\n")
    with (OUT_DIR / "review_sheet.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["id", "patient_id", "criterion", "expected_status", "evidence", "rationale", "review (ok / fix: ...)"]
        )
        for p in pairs:
            w.writerow(
                [
                    p["id"],
                    p["patient_id"],
                    p["criterion"]["text"],
                    p["expected_status"],
                    " ".join(p["expected_evidence"]),
                    p["rationale"],
                    "",
                ]
            )
    counts = {s: sum(p["expected_status"] == s for p in pairs) for s in TARGET}
    print(f"wrote {len(pairs)} pairs {counts} to {OUT_DIR}")


if __name__ == "__main__":
    main()
