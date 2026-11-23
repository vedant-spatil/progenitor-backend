"""Text-patient adapter: a TREC/SIGIR topic becomes a PatientProfile the existing workers can judge.

Each sentence is a citable note (`Note/<topic>#<n>`, shown to the judge as N1, N2, ...). Age and sex
come from the first "<n>-year-old <man|woman|...>" phrase. Topics are synthetic case descriptions
(TREC/SIGIR), not patient records.
"""

from __future__ import annotations

import re

from agents.schemas import NoteFact, PatientProfile

_AGE = re.compile(r"\b(\d{1,3})[- ](?:year|yr)s?[- ]old\b", re.IGNORECASE)
_MONTHS = re.compile(r"\b(\d{1,2})[- ]month[- ]old\b", re.IGNORECASE)
_FEMALE = re.compile(r"\b(woman|female|girl|lady|mother|she|her)\b", re.IGNORECASE)
_MALE = re.compile(r"\b(man|male|boy|gentleman|father|he|his)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(text.strip()) if s.strip()]


def age_and_sex(text: str) -> tuple[int | None, str]:
    age: int | None = None
    if m := _AGE.search(text):
        age = min(int(m.group(1)), 90)  # 90 means 90 or older, as in the FHIR profiles
    elif _MONTHS.search(text):
        age = 0
    female, male = len(_FEMALE.findall(text)), len(_MALE.findall(text))
    sex = "female" if female > male else "male" if male > female else ""
    return age, sex


def profile_from_text(topic_id: str, text: str) -> PatientProfile:
    age, sex = age_and_sex(text)
    notes = [NoteFact(resource_id=f"Note/{topic_id}#{i}", text=s) for i, s in enumerate(sentences(text), start=1)]
    return PatientProfile(patient_ref=f"Patient/{topic_id}", age_years=age, sex=sex, notes=notes)


def patient_query(text: str) -> str:
    """Text for the BM25 baseline."""
    return text
