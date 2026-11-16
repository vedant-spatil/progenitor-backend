"""Gate 1: AuditEvent is append-only at the database level (Postgres trigger).

The model/queryset guards are covered in unit tests. Here raw SQL goes around
them, so only the trigger can stop it. Requires the test database to be Postgres
(the Makefile sets DATABASE_URL for integration runs).
"""

import pytest
from django.db import DatabaseError, connection, transaction

from apps.audit.models import AuditEvent, AuditKind

pytestmark = [pytest.mark.integration, pytest.mark.django_db]


@pytest.fixture
def event_id() -> int:
    assert connection.vendor == "postgresql", "integration tests must run against Postgres"
    return AuditEvent.objects.create(kind=AuditKind.DATA_LOAD, payload={"n": 1}).pk


@pytest.mark.parametrize(
    "sql",
    [
        "UPDATE audit_auditevent SET kind = 'tampered' WHERE id = %s",
        "DELETE FROM audit_auditevent WHERE id = %s",
    ],
)
def test_trigger_blocks_update_and_delete(event_id: int, sql: str) -> None:
    # psycopg 3 surfaces PL/pgSQL RAISE as ProgrammingError; match the trigger's message, not the class.
    with pytest.raises(DatabaseError, match="append-only"), transaction.atomic(), connection.cursor() as cur:
        cur.execute(sql, [event_id])
    assert AuditEvent.objects.get(pk=event_id).kind == AuditKind.DATA_LOAD


def test_trigger_blocks_truncate(event_id: int) -> None:
    with transaction.atomic(), connection.cursor() as cur:
        # The fixture's insert queued deferred FK checks; flush them so TRUNCATE reaches our trigger.
        cur.execute("SET CONSTRAINTS ALL IMMEDIATE")
        with pytest.raises(DatabaseError, match="append-only"), transaction.atomic():
            cur.execute("TRUNCATE audit_auditevent CASCADE")
    assert AuditEvent.objects.filter(pk=event_id).exists()
