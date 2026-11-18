"""Independent verification of handlabel_v1.json against raw HAPI records.

Re-derives every label without the draft generator's code: follows all paging links, checks units,
latest dates, comparison direction, active status, and absence for `unknown`. One line per pair:
OK or MISMATCH | id | expected | verified | criterion | what was checked.

    uv run python data/handlabel/verify_labels.py data/handlabel/handlabel_v1.json   # set FHIR_BASE_URL
"""

import json
import os
import sys
from datetime import date
from typing import Any

import httpx

BASE = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")
H = httpx.Client(base_url=BASE, timeout=120, headers={"Accept": "application/fhir+json"})
doc = json.load(open(sys.argv[1]))
AS_OF = date.fromisoformat(doc["as_of"])


def get(ref: str) -> dict[str, Any]:
    body: dict[str, Any] = H.get("/" + ref).json()
    return body


def allres(rtype: str, params: dict[str, str]) -> list[dict[str, Any]]:
    out = []
    body = H.get(f"/{rtype}", params={**params, "_count": "500"}).json()
    while True:
        out += [e["resource"] for e in body.get("entry", [])]
        nxt = next((link["url"] for link in body.get("link", []) if link["relation"] == "next"), None)
        if not nxt:
            return out
        body = H.get(nxt.replace("http://hapi-fhir:8080/fhir", BASE)).json()


def codes(cc: dict[str, Any] | None) -> set[str]:
    return {c.get("code") for c in (cc or {}).get("coding", [])}


def disp(cc: dict[str, Any] | None) -> str:
    return " / ".join(c.get("display", "") for c in (cc or {}).get("coding", []))


LAB = {
    "HbA1c": ("4548-4", ">=", 7.0, "%"),
    "eGFR": ("33914-3", "<", 60, "mL/min"),
    "Body mass index": ("39156-5", ">=", 30, "kg/m2"),
    "creatinine": ("2160-0", ">", 1.5, "mg/dL"),
    "LDL": ("18262-6", ">=", 100, "mg/dL"),
    "Platelet": ("777-3", ">=", 100, "10*3/uL"),
    "Systolic": ("8480-6", ">=", 140, "mm[Hg]"),
}
COND = {
    "type 2 diabetes": {"44054006"},
    "essential hypertension": {"59621000"},
    "chronic kidney disease": {"431855005", "431856006", "433144002", "431857002", "46177005", "127013003"},
    "asthma": {"195967001", "233678006"},
}


ALT_CODES: dict[str, tuple[str, ...]] = {"2160-0": ("38483-4",)}


def cmp(v: float, op: str, t: float) -> bool:
    return {">=": v >= t, ">": v > t, "<": v < t}[op]


for p in doc["pairs"]:
    pid, text, exp, ev = p["patient_id"], p["criterion"]["text"], p["expected_status"], p["expected_evidence"]
    verdict, note = None, ""
    if text.startswith("Age"):
        pat = get(f"Patient/{pid}")
        b = date.fromisoformat(pat["birthDate"])
        age = AS_OF.year - b.year - ((AS_OF.month, AS_OF.day) < (b.month, b.day))
        ok = age >= 65 if "65" in text else 18 <= age <= 75
        verdict = "met" if ok else "not_met"
        note = f"age {age} as of {AS_OF}"
    elif text.startswith("Female"):
        g = get(f"Patient/{pid}")["gender"]
        verdict = "met" if g == "female" else "not_met"
        note = f"gender {g}"
    elif any(k in text for k in LAB):
        k = next(k for k in LAB if k in text)
        code, op, t, unit = LAB[k]
        # Equivalent codes: Synthea records creatinine as 2160-0 (serum/plasma) or 38483-4 (blood).
        accepted = {code, *ALT_CODES.get(code, ())}
        rows = []
        for o in allres(
            "Observation",
            {
                "patient": pid,
                "code" if code != "8480-6" else "component-code": ",".join(f"http://loinc.org|{c}" for c in accepted),
            },
        ):
            for h in [o] if code != "8480-6" else o.get("component", []):
                if accepted & codes(h.get("code")) and (q := h.get("valueQuantity")):
                    rows.append(
                        (
                            o.get("effectiveDateTime", ""),
                            q["value"],
                            q.get("unit") or q.get("code"),
                            f"Observation/{o['id']}",
                        )
                    )
        if not rows:
            verdict, note = "unknown", f"no {code} records"
        else:
            rows.sort(reverse=True)
            when, v, u, ref = rows[0]
            same_time = [r for r in rows if r[0] == when]
            verdict = "met" if cmp(v, op, t) else "not_met"
            note = f"latest {code} {v:g} {u} on {when[:10]} ({len(rows)} readings)" + (
                f"; {len(same_time)} readings at same time" if len(same_time) > 1 else ""
            )
            if ev and ev[0] != ref and not any(r[3] == ev[0] for r in same_time):
                note += f"; evidence {ev[0]} is NOT the latest"
            if unit.split("/")[0].lower() not in (u or "").lower():
                note += f"; UNIT CHECK rule {unit} vs record {u}"
    elif text.startswith("Active diagnosis"):
        k = next(k for k in COND if k in text)
        cs = allres("Condition", {"patient": pid})
        hits = [c for c in cs if codes(c.get("code")) & COND[k]]
        act = [c for c in hits if "active" in codes(c.get("clinicalStatus"))]
        verdict = "met" if act else "not_met" if hits else "unknown"
        note = (
            (f"{len(act)} active / {len(hits)} total; " + "; ".join(sorted({disp(c["code"]) for c in hits}))[:120])
            if hits
            else f"no matching Condition among {len(cs)}"
        )
        if act and ev and ev[0] not in {f"Condition/{c['id']}" for c in act}:
            note += "; evidence not an active match"
    elif text.startswith("Currently prescribed"):
        word = "metformin" if "metformin" in text else "insulin"
        ms = allres("MedicationRequest", {"patient": pid})
        hits = [m for m in ms if word in json.dumps(m.get("medicationCodeableConcept", {})).lower()]
        act = [m for m in hits if m.get("status") == "active"]
        verdict = "met" if act else "not_met" if hits else "unknown"
        note = (
            f"{len(act)} active / {len(hits)} total; "
            + ("; ".join(sorted({disp(m["medicationCodeableConcept"])[:60] for m in hits}))[:140])
        )
    else:
        gender = get(f"Patient/{pid}")["gender"]
        if "pregnancy" in text.lower() and gender == "male":
            verdict, note = "not_applicable", "pregnancy rule; patient is male"
        else:
            verdict, note = "unknown", "criterion not representable in Synthea FHIR (no such resource type)"
    flag = "OK " if verdict == exp else "MISMATCH"
    print(f"{flag}|{p['id']}|{exp}|{verdict}|{text[:60]}|{note}")
