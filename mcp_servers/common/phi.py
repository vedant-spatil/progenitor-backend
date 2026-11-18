"""PHI egress filter for outbound calls (ClinicalTrials.gov, PubMed, RxNorm, NLM tables).

Detects the 18 HIPAA Safe Harbor identifier types in text that is about to leave
the system. Two layers:

1. Generic patterns that need no patient context (SSN, phone, email, full dates,
   ZIP codes, URLs, IPs, labelled record/plan/account/licence/device numbers...).
2. Known identifiers taken from the patient's FHIR record (names, birth date,
   MRNs, address parts, phone, email, other identifier values), which catch
   things no pattern can, such as a bare name or city.

Outbound queries carry condition and drug terms only, so the patterns lean
towards blocking. Blocked calls are logged with identifier types and the
destination only, never the text.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

log = logging.getLogger(__name__)


class IdentifierType(StrEnum):
    """HIPAA Safe Harbor identifiers, 45 CFR 164.514(b)(2)(i)(A)-(R)."""

    NAME = "name"
    GEOGRAPHIC = "geographic"  # smaller than a state: street, city, ZIP
    DATE = "date"  # date elements other than year; ages over 89
    PHONE = "phone"
    FAX = "fax"
    EMAIL = "email"
    SSN = "ssn"
    MRN = "mrn"
    HEALTH_PLAN = "health_plan"
    ACCOUNT = "account"
    LICENSE = "license"
    VEHICLE = "vehicle"
    DEVICE = "device"
    URL = "url"
    IP = "ip"
    BIOMETRIC = "biometric"
    PHOTO = "photo"
    OTHER_UNIQUE = "other_unique"


class PhiBlockedError(Exception):
    """Outbound text contains identifiers. `types` lists what was found (never the values)."""

    def __init__(self, types: Iterable[IdentifierType], destination: str = "") -> None:
        self.types = sorted(set(types))
        self.destination = destination
        super().__init__(f"PHI egress blocked ({destination or 'outbound'}): {', '.join(self.types)}")


_I = re.IGNORECASE
_MONTHS = (
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
_ID = r"[A-Z0-9][A-Z0-9-]{3,}"  # an identifier value after a label

PATTERNS: dict[IdentifierType, list[re.Pattern[str]]] = {
    IdentifierType.NAME: [
        re.compile(r"\b(?i:mr|mrs|ms|miss|mx)\.?\s+[A-Z][a-z]+"),
        re.compile(r"\b(?:patient|pt)\s+name\s*[:=]", _I),
    ],
    IdentifierType.GEOGRAPHIC: [
        re.compile(
            r"\b\d{1,6}\s+(?:[A-Z][A-Za-z]+\s+){1,4}"
            r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|way|place|pl"
            r"|terrace|circle|cir|highway|hwy|parkway|pkwy)\b\.?",
            _I,
        ),
        re.compile(r"(?<![\d.-])\d{5}(?:-\d{4})?(?![\d.])"),  # US ZIP / ZIP+4
        re.compile(r"\b(?:zip(?:\s*code)?|postal\s*code|address)\s*[:=#]", _I),
        re.compile(r"\b(?:p\.?\s*o\.?\s*box)\s*\d+", _I),
    ],
    IdentifierType.DATE: [
        re.compile(r"\b\d{4}-\d{1,2}-\d{1,2}\b"),  # 2021-03-04
        re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b"),  # 03/04/2021, 4.3.21
        re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}(?:st|nd|rd|th)?,?\s+\d{{2,4}}\b", _I),  # March 3, 2021
        re.compile(rf"\b\d{{1,2}}(?:st|nd|rd|th)?\s+{_MONTHS}\.?,?\s+\d{{2,4}}\b", _I),  # 3 March 2021
        re.compile(rf"\b{_MONTHS}\.?\s+(?:19|20)\d{{2}}\b", _I),  # March 2021
        re.compile(r"\b(?:dob|date\s+of\s+birth|birth\s*date|born\s+on|admitted\s+on|discharged\s+on)\b", _I),
        re.compile(r"\b(?:9\d|1[0-4]\d)\s*-?\s*(?:years?|yrs?|y/?o)(?:\s*-?\s*old)?\b", _I),  # ages > 89
        re.compile(r"\bage[ds]?\s*[:=]?\s*(?:9\d|1[0-4]\d)\b", _I),
    ],
    IdentifierType.PHONE: [
        re.compile(r"(?<![\w.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w.])"),
        re.compile(r"\b(?:phone|tel|telephone|mobile|cell)\s*(?:no\.?|number|#)?\s*[:=]?\s*\+?\d[\d\s().-]{6,}", _I),
    ],
    IdentifierType.FAX: [
        re.compile(r"\bfax\s*(?:no\.?|number|#)?\s*[:=]?\s*\+?[\d(][\d\s().-]{6,}", _I),
    ],
    IdentifierType.EMAIL: [
        re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    ],
    IdentifierType.SSN: [
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        re.compile(r"\b(?:ssn|social\s+security(?:\s+(?:no\.?|number|#))?)\s*[:=#]?\s*\d{3}\s?\d{2}\s?\d{4}\b", _I),
    ],
    IdentifierType.MRN: [
        re.compile(
            rf"\b(?:mrn|medical\s+record(?:\s+(?:no\.?|number|#))?|chart\s+(?:no\.?|number|#))\s*[:=#]?\s*{_ID}",
            _I,
        ),
    ],
    IdentifierType.HEALTH_PLAN: [
        re.compile(
            rf"\b(?:member|subscriber|beneficiary|policy|insurance|medicaid|medicare|health\s+plan|group)"
            rf"\s*(?:id|no\.?|number|#)\s*[:=#]?\s*{_ID}",
            _I,
        ),
        # Medicare Beneficiary Identifier, e.g. 1EG4-TE5-MK73
        re.compile(
            r"\b[1-9][AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d-?[AC-HJKMNP-RT-Y][AC-HJKMNP-RT-Y0-9]\d-?"
            r"[AC-HJKMNP-RT-Y]{2}\d{2}\b"
        ),
    ],
    IdentifierType.ACCOUNT: [
        re.compile(r"\b(?:account|acct|billing|invoice)\.?\s*(?:no\.?|number|#)?\s*[:=#]?\s*\d{4,}", _I),
    ],
    IdentifierType.LICENSE: [
        re.compile(
            rf"\b(?:licen[cs]e|certificate|cert|dea|driver'?s?\s+licen[cs]e|passport)"
            rf"\s*(?:no\.?|number|#)?\s*[:=#]?\s*{_ID}",
            _I,
        ),
        re.compile(r"\b[A-Z]{2}\d{7}\b"),  # DEA registration format
    ],
    IdentifierType.VEHICLE: [
        re.compile(r"\b(?=[A-HJ-NPR-Z0-9]*\d)(?=[A-HJ-NPR-Z0-9]*[A-HJ-NPR-Z])[A-HJ-NPR-Z0-9]{17}\b"),  # VIN
        re.compile(r"\b(?:vin|license\s+plate|plate\s+(?:no\.?|number|#))\s*[:=#]?\s*[A-Z0-9-]{2,}", _I),
    ],
    IdentifierType.DEVICE: [
        re.compile(
            r"\b(?:serial\s*(?:no\.?|number|#)|s/n|device\s+(?:id|identifier|serial)|udi|imei)"
            r"\s*[:=#]?\s*[A-Z0-9()-]{4,}",
            _I,
        ),
        re.compile(r"\(01\)\d{14}"),  # GS1 UDI device identifier
    ],
    IdentifierType.URL: [
        re.compile(r"\b(?:https?|ftp)://\S+", _I),
        re.compile(r"\bwww\.[\w-]+\.\S+", _I),
    ],
    IdentifierType.IP: [
        re.compile(r"\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b"),
        re.compile(r"(?<![\w:])(?:[0-9a-f]{1,4}:){3,7}(?::|[0-9a-f]{1,4})(?![\w:])", _I),  # IPv6
    ],
    IdentifierType.BIOMETRIC: [
        re.compile(
            r"\b(?:finger\s?prints?|retina(?:l)?\s+scans?|iris\s+scans?|voice\s?prints?|face\s?prints?"
            r"|palm\s?prints?|biometric\s+(?:id|identifier|template|data))\b",
            _I,
        ),
    ],
    IdentifierType.PHOTO: [
        re.compile(r"data:image/", _I),
        re.compile(r"\b[\w-]+\.(?:jpe?g|png|gif|bmp|heic|tiff?|webp)\b", _I),
        re.compile(r"\b(?:photo(?:graph)?s?|selfies?|headshots?|face\s+images?)\b", _I),
    ],
    IdentifierType.OTHER_UNIQUE: [
        re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", _I),  # UUID (FHIR ids)
        re.compile(r"(?<![\w.])\d{9,}(?![\w.])"),  # long bare digit runs
        re.compile(r"\b(?:patient|person|record|unique)\s*(?:id|identifier)\s*[:=#]?\s*[A-Z0-9-]{3,}", _I),
    ],
}


@dataclass(frozen=True)
class KnownIdentifiers:
    """Identifiers from one patient's record, checked by exact (case-insensitive) match."""

    values: Mapping[IdentifierType, frozenset[str]] = field(default_factory=dict)

    @staticmethod
    def _clean(items: Iterable[str], min_len: int = 3) -> frozenset[str]:
        return frozenset(s.strip().lower() for s in items if s and len(s.strip()) >= min_len)

    @classmethod
    def from_fhir_patient(cls, patient: Mapping[str, Any]) -> KnownIdentifiers:
        """Build from a FHIR R4 Patient resource."""
        names: list[str] = []
        for n in patient.get("name", []) or []:
            names += [str(g) for g in n.get("given", []) or []]
            if n.get("family"):
                names.append(str(n["family"]))
            if n.get("text"):
                names.append(str(n["text"]))
        phones: list[str] = []
        emails: list[str] = []
        faxes: list[str] = []
        for t in patient.get("telecom", []) or []:
            value = str(t.get("value", ""))
            system = t.get("system")
            (emails if system == "email" else faxes if system == "fax" else phones).append(value)
        geo: list[str] = []
        for a in patient.get("address", []) or []:
            geo += [str(x) for x in a.get("line", []) or []]
            geo += [str(a[k]) for k in ("city", "district", "postalCode") if a.get(k)]
        mrns: list[str] = []
        others: list[str] = []
        for ident in patient.get("identifier", []) or []:
            value = str(ident.get("value", ""))
            codes = {c.get("code") for c in (ident.get("type", {}) or {}).get("coding", []) or []}
            (mrns if "MR" in codes else others).append(value)
        dates = [str(patient["birthDate"])] if patient.get("birthDate") else []
        if patient.get("id"):
            others.append(str(patient["id"]))
        return cls(
            values={
                IdentifierType.NAME: cls._clean(names, min_len=2),
                IdentifierType.PHONE: cls._clean(phones),
                IdentifierType.FAX: cls._clean(faxes),
                IdentifierType.EMAIL: cls._clean(emails),
                IdentifierType.GEOGRAPHIC: cls._clean(geo),
                IdentifierType.MRN: cls._clean(mrns),
                IdentifierType.DATE: cls._clean(dates),
                IdentifierType.OTHER_UNIQUE: cls._clean(others),
            }
        )


def _contains_value(text_lower: str, value: str) -> bool:
    # Whole-token match so the name "Al" does not match "albumin".
    return re.search(rf"(?<![\w]){re.escape(value)}(?![\w])", text_lower) is not None


def find_phi(text: str, known: KnownIdentifiers | None = None) -> set[IdentifierType]:
    """Identifier types present in `text`."""
    found = {kind for kind, patterns in PATTERNS.items() if any(p.search(text) for p in patterns)}
    if known:
        lowered = text.lower()
        for kind, values in known.values.items():
            if kind not in found and any(_contains_value(lowered, v) for v in values):
                found.add(kind)
    return found


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from _strings(v)
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _strings(v)


def check_outbound(payload: Any, *, destination: str, known: KnownIdentifiers | None = None) -> None:
    """Raise PhiBlockedError if any string in `payload` carries identifiers. Logs types only."""
    found: set[IdentifierType] = set()
    for text in _strings(payload):
        found |= find_phi(text, known)
    if found:
        err = PhiBlockedError(found, destination)
        log.warning("phi_egress_blocked destination=%s types=%s", destination, ",".join(err.types))
        raise err
