"""Register loaded FHIR patients as app-side Patient rows (ids and synthetic labels only)."""

from __future__ import annotations

from collections.abc import Mapping

from django.db import transaction

from apps.audit.models import AuditEvent, AuditKind
from apps.patients.models import Patient
from data.loaders.fhir import default_base_url


@transaction.atomic
def register_patients(source: str, labelled_ids: list[tuple[str, str]]) -> tuple[int, int]:
    """Upsert (source, fhir_id) rows. `labelled_ids` is [(fhir_id, label)]. Returns (created, existing)."""
    created = 0
    base = default_base_url()
    for fhir_id, label in labelled_ids:
        _, was_created = Patient.objects.update_or_create(
            source=source, fhir_id=fhir_id, defaults={"label": label, "fhir_base_url": base}
        )
        created += was_created
    return created, len(labelled_ids) - created


def record_load(node: str, payload: Mapping[str, object]) -> None:
    """Audit a loader run. Payload holds counts only."""
    AuditEvent.objects.create(kind=AuditKind.DATA_LOAD, node=node, payload=dict(payload))
