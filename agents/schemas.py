"""Pydantic models for worker I/O. Change schemas here, not inline (project rule).

Profiles carry no direct identifiers: no names, birth dates, addresses, or MRNs.
Age is in whole years, capped at 90 (Safe Harbor groups ages over 89).
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------- Patient profile (Profiler) ----------------


class CodedFact(Model):
    resource_id: str = Field(description="FHIR reference, e.g. 'Condition/abc'. Cite this as evidence.")
    system: str = ""
    code: str = ""
    display: str = ""


class ConditionFact(CodedFact):
    clinical_status: str = ""
    onset: date | None = None
    abatement: date | None = None


class MedicationFact(CodedFact):
    status: str = ""
    authored_on: date | None = None


class LabFact(CodedFact):
    value: float | None = None
    value_text: str = ""
    unit: str = ""
    effective: date | None = None
    category: str = ""


class PatientProfile(Model):
    patient_ref: str = Field(description="'Patient/<id>'. Cite for age and sex.")
    age_years: int | None = Field(default=None, description="Whole years; 90 means 90 or older.")
    sex: str = ""
    conditions: list[ConditionFact] = Field(default_factory=list)
    medications: list[MedicationFact] = Field(default_factory=list)
    labs: list[LabFact] = Field(default_factory=list, description="Latest value per code.")
    truncated: bool = Field(default=False, description="True if a resource type hit the page limit.")

    def resource_ids(self) -> set[str]:
        ids = {self.patient_ref}
        for group in (self.conditions, self.medications, self.labs):
            ids |= {f.resource_id for f in group}
        return ids


# ---------------- Trials (Trial Scout) ----------------


class SearchSpec(Model):
    conditions: str
    terms: str = ""
    interventions: str = ""
    phases: list[str] = Field(default_factory=list)
    location: str = ""
    max_results: int = 20


class TrialSummary(Model):
    nct_id: str
    title: str = ""
    status: str = ""
    phases: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    minimum_age: str | None = None
    maximum_age: str | None = None
    sex: str | None = None


# ---------------- Criteria (Criteria Parser) ----------------


class CriterionKind(StrEnum):
    INCLUSION = "inclusion"
    EXCLUSION = "exclusion"


class EntityType(StrEnum):
    CONDITION = "condition"
    DRUG = "drug"
    LAB = "lab"
    PROCEDURE = "procedure"
    DEMOGRAPHIC = "demographic"  # age, sex
    OBSERVATION = "observation"  # vitals, scores, findings
    OTHER = "other"  # consent, pregnancy, lifestyle, ...


class ParsedEntity(Model):
    text: str = Field(description="Span copied from the criterion text.")
    type: EntityType
    codes: list[dict[str, str]] = Field(default_factory=list, description="Filled by terminology lookup.")


class Criterion(Model):
    index: int = Field(description="0-based position in the trial's criteria list.")
    kind: CriterionKind
    source_text: str
    entities: list[ParsedEntity] = Field(default_factory=list)
    entity: str = Field(default="", description="Main entity, e.g. 'HbA1c'.")
    operator: Literal["", "present", "absent", ">", ">=", "<", "<=", "=", "between"] = ""
    value: str = ""
    unit: str = ""
    temporal_window: str = Field(default="", description="e.g. 'last 3 months'.")


class ParsedCriteria(Model):
    criteria: list[Criterion]


# ---------------- Verdicts (Eligibility Judge) ----------------


class VerdictStatus(StrEnum):
    MET = "met"
    NOT_MET = "not_met"
    UNKNOWN = "unknown"


class Verdict(Model):
    criterion_index: int
    status: VerdictStatus
    evidence_refs: list[str] = Field(default_factory=list, description="FHIR refs from the profile only.")
    reason: str = ""
    downgraded: bool = Field(default=False, description="Set by the citation guard, never by the model.")


class JudgeOutput(Model):
    verdicts: list[Verdict]


# ---------------- Evidence, ranking, referral ----------------


class Paper(Model):
    pmid: str
    title: str = ""
    year: str | None = None
    relevance: str = Field(default="", description="One sentence: why this paper matters for the trial.")


class EvidenceOutput(Model):
    papers: list[Paper]


class TrialAssessment(Model):
    """Ranker input for one trial; counts are computed by code (display only)."""

    trial: TrialSummary
    verdicts: list[Verdict]
    criteria: list[Criterion]
    papers: list[Paper] = Field(default_factory=list)


class RankedTrial(Model):
    nct_id: str
    rank: int
    explanation: str


class Ranking(Model):
    ranked: list[RankedTrial]


class ReferralNote(Model):
    nct_id: str
    body: str
