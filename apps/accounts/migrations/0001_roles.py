"""Create the role groups: coordinator, clinician, admin (apps/accounts/roles.py)."""

from typing import Any

from django.db import migrations

ROLES = ("coordinator", "clinician", "admin")


def create_roles(apps: Any, schema_editor: Any) -> None:
    group = apps.get_model("auth", "Group")
    for name in ROLES:
        group.objects.get_or_create(name=name)


def remove_roles(apps: Any, schema_editor: Any) -> None:
    apps.get_model("auth", "Group").objects.filter(name__in=ROLES).delete()


class Migration(migrations.Migration):
    dependencies = [("auth", "0012_alter_user_first_name_max_length")]
    operations = [migrations.RunPython(create_roles, remove_roles)]
