"""Create the demo sign-in users: `coordinator` and `clinician` (one role each). Idempotent.

    DEMO_USER_PASSWORD=... uv run python manage.py seed_demo_users

The password comes from the environment, never from code. Existing users keep their password unless
--reset-password is given.
"""

from __future__ import annotations

import os
from typing import Any

from django.contrib.auth.models import Group, User
from django.core.management.base import BaseCommand, CommandError, CommandParser

from apps.accounts.roles import CLINICIAN, COORDINATOR

DEMO_USERS = {"coordinator": COORDINATOR, "clinician": CLINICIAN}


class Command(BaseCommand):
    help = "Create demo users 'coordinator' and 'clinician' (password from DEMO_USER_PASSWORD)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--reset-password", action="store_true", help="Also reset existing users' passwords.")

    def handle(self, *args: Any, **opts: Any) -> None:
        password = os.environ.get("DEMO_USER_PASSWORD", "")
        if len(password) < 12:
            raise CommandError("set DEMO_USER_PASSWORD (12+ characters) in the environment")
        for username, role in DEMO_USERS.items():
            user, created = User.objects.get_or_create(username=username)
            if created or opts["reset_password"]:
                user.set_password(password)
                user.save()
            user.groups.add(Group.objects.get(name=role))
            self.stdout.write(f"{username}: {'created' if created else 'exists'} (role {role})")
