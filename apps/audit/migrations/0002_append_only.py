"""Postgres trigger that makes audit_auditevent append-only (UPDATE, DELETE, TRUNCATE).

Postgres only. On other databases (sqlite unit tests) the model and queryset
guards in apps/audit/models.py enforce the same rule.
"""

from django.db import migrations
from django.db.backends.base.schema import BaseDatabaseSchemaEditor
from django.db.migrations.state import StateApps

SQL = """
CREATE OR REPLACE FUNCTION audit_block_change() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit_auditevent is append-only: % is not allowed', TG_OP;
END;
$$ LANGUAGE plpgsql;
CREATE TRIGGER audit_no_update_delete BEFORE UPDATE OR DELETE ON audit_auditevent
    FOR EACH ROW EXECUTE FUNCTION audit_block_change();
CREATE TRIGGER audit_no_truncate BEFORE TRUNCATE ON audit_auditevent
    FOR EACH STATEMENT EXECUTE FUNCTION audit_block_change();
"""

REVERSE = """
DROP TRIGGER IF EXISTS audit_no_truncate ON audit_auditevent;
DROP TRIGGER IF EXISTS audit_no_update_delete ON audit_auditevent;
DROP FUNCTION IF EXISTS audit_block_change();
"""


def forwards(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(SQL, params=None)


def backwards(apps: StateApps, schema_editor: BaseDatabaseSchemaEditor) -> None:
    if schema_editor.connection.vendor == "postgresql":
        schema_editor.execute(REVERSE, params=None)


class Migration(migrations.Migration):
    dependencies = [("audit", "0001_initial")]
    operations = [migrations.RunPython(forwards, backwards)]
