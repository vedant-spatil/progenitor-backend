"""Gate 2: the PHI egress filter blocks all 18 HIPAA identifier types and passes clinical terms."""

import logging

import pytest

from mcp_servers.common.phi import (
    IdentifierType,
    KnownIdentifiers,
    PhiBlockedError,
    check_outbound,
    find_phi,
)

T = IdentifierType

# At least two samples per identifier type. Each sample must be detected as its own type.
SAMPLES: dict[IdentifierType, list[str]] = {
    T.NAME: ["diabetes trial for Mrs. Hartley", "patient name: J Doe, type 2 diabetes"],
    T.GEOGRAPHIC: [
        "lives at 42 Maple Grove Avenue, asthma",
        "lung cancer near 02139",
        "ZIP code: 94110",
        "Boston 02139-4307",
        "PO Box 1234",
    ],
    T.DATE: [
        "diagnosed 2021-03-04",
        "seen 03/04/2021 for CKD",
        "admitted on March 3, 2021",
        "since 3 March 2021",
        "onset Jan 2019",
        "DOB 1950",
        "92-year-old woman with hypertension",
        "age 95 with heart failure",
    ],
    T.PHONE: ["call 617-555-0142 about trial", "contact (617) 555 0142", "phone: +1 617 555 0142"],
    T.FAX: ["fax 617-555-0199", "Fax #: (617) 555-0199"],
    T.EMAIL: ["email jane.doe@example.org", "j.smith+trials@mail.example.com"],
    T.SSN: ["SSN 123-45-6789", "social security number 123 45 6789"],
    T.MRN: ["MRN: 00412345", "medical record number A1234567", "chart no. 778899"],
    T.HEALTH_PLAN: ["member id XYZ123456", "Medicare number: 1EG4-TE5-MK73", "policy number 55-AB-9921"],
    T.ACCOUNT: ["account number 9988776655", "acct # 44556677"],
    T.LICENSE: ["driver's license D1234567", "DEA AB1234563", "license no. MA-445566"],
    T.VEHICLE: ["VIN 1HGCM82633A004352", "license plate 7ABC123", "car 1HGCM82633A004352"],
    T.DEVICE: ["pump serial number SN-778812", "device id PM-99812", "UDI (01)00884838038410"],
    T.URL: ["see https://portal.example.org/patient/123", "www.myhealthrecord.com/me"],
    T.IP: ["login from 192.168.10.24", "host 2001:db8:85a3:0:0:8a2e:370:7334"],
    T.BIOMETRIC: ["fingerprint on file", "retina scan template", "voiceprint enrolled"],
    T.PHOTO: ["attached photo of patient", "scan_portrait.jpg", "data:image/png;base64,iVBORw0KGgo="],
    T.OTHER_UNIQUE: [
        "fhir id 5fe73798-ce2f-73a3-83d7-f9cff2947e19",
        "reference 1234567890123",
        "patient id: PX-4471",
    ],
}

CLINICAL_TERMS = [
    "type 2 diabetes mellitus",
    "HbA1c > 7.5%",
    "eGFR < 30 mL/min/1.73 m2",
    "non-small cell lung cancer stage III",
    "HER2-positive breast cancer",
    "EGFR exon 19 deletion",
    "PD-L1 TPS >= 50%",
    "metformin 500 mg twice daily",
    "no insulin in the last 3 months",
    "blood pressure 140/90 mmHg",
    "LOINC 4548-4",
    "LOINC 38483-4",
    "33914-3",
    "ICD-10 E11.9",
    "NCT04280705",
    "age 18 to 75 years",
    "aged 65 or older",
    "BMI 30 kg/m2",
    "platelets >= 100000/uL",
    "COVID-19",
    "IL-6 inhibitor",
    "chronic kidney disease stage 3",
    "asthma with FEV1 < 80%",
    "pregnancy in 2024",
    "photodynamic therapy",
    "Massachusetts",
]


def test_every_identifier_type_has_samples() -> None:
    """100% coverage of the 18 identifier types (Gate 2)."""
    assert len(IdentifierType) == 18
    assert set(SAMPLES) == set(IdentifierType)
    assert all(len(v) >= 2 for v in SAMPLES.values())


@pytest.mark.parametrize(
    ("kind", "text"), [(k, s) for k, samples in SAMPLES.items() for s in samples], ids=lambda x: str(x)[:40]
)
def test_identifier_detected(kind: IdentifierType, text: str) -> None:
    assert kind in find_phi(text)
    with pytest.raises(PhiBlockedError):
        check_outbound({"query": text}, destination="test")


@pytest.mark.parametrize("text", CLINICAL_TERMS)
def test_clinical_terms_pass(text: str) -> None:
    assert find_phi(text) == set()
    check_outbound({"query.cond": text}, destination="test")


PATIENT = {
    "resourceType": "Patient",
    "id": "pat-001",
    "name": [{"family": "Okafor", "given": ["Adaeze", "Ngozi"]}],
    "birthDate": "1961-07-19",
    "telecom": [{"system": "phone", "value": "555-0142"}, {"system": "email", "value": "adaeze@example.net"}],
    "address": [{"line": ["17 Quincy Row"], "city": "Somerville", "postalCode": "02143", "state": "MA"}],
    "identifier": [
        {"type": {"coding": [{"code": "MR"}]}, "value": "MRN-77812"},
        {"type": {"coding": [{"code": "SS"}]}, "value": "999-55-1234"},
    ],
}


class TestKnownIdentifiers:
    known = KnownIdentifiers.from_fhir_patient(PATIENT)

    @pytest.mark.parametrize(
        ("text", "kind"),
        [
            ("breast cancer trials for okafor", T.NAME),
            ("Adaeze diabetes", T.NAME),
            ("trials in Somerville", T.GEOGRAPHIC),
            ("born 1961-07-19", T.DATE),
            ("MRN-77812 lung cancer", T.MRN),
            ("ref pat-001", T.OTHER_UNIQUE),
        ],
    )
    def test_known_values_blocked(self, text: str, kind: IdentifierType) -> None:
        assert kind in find_phi(text, self.known)

    def test_whole_token_match_only(self) -> None:
        known = KnownIdentifiers(values={T.NAME: frozenset({"al"})})
        assert find_phi("serum albumin low", known) == set()
        assert T.NAME in find_phi("trial for al", known)

    def test_state_is_not_blocked(self) -> None:
        """States are allowed by Safe Harbor; only smaller geographic units are identifiers."""
        assert find_phi("trials in Massachusetts", self.known) == set()


def test_nested_payload_and_log_has_types_only(caplog: pytest.LogCaptureFixture) -> None:
    payload = {"conditions": "asthma", "extra": [{"note": "call 617-555-0142"}]}
    with caplog.at_level(logging.WARNING), pytest.raises(PhiBlockedError) as err:
        check_outbound(payload, destination="pubmed")
    assert err.value.types == [T.PHONE]
    assert "617" not in caplog.text
    assert "phone" in caplog.text
    assert "destination=pubmed" in caplog.text
