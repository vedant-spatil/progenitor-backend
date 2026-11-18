"""User roles (Django groups) and the DRF permissions built on them.

- coordinator: starts runs and reads them.
- clinician: answers clinician questions, reviews shortlists, reads runs.
- admin: manages users (Django admin). Admins do not read run data unless they also hold another role
  (minimum necessary access to patient-derived content).
"""

from __future__ import annotations

from typing import Any

from rest_framework.permissions import BasePermission

COORDINATOR = "coordinator"
CLINICIAN = "clinician"
ADMIN = "admin"
ROLES = (COORDINATOR, CLINICIAN, ADMIN)


def has_role(user: Any, *roles: str) -> bool:
    if not getattr(user, "is_authenticated", False):
        return False
    return bool(user.groups.filter(name__in=roles).exists())


class _RolePermission(BasePermission):
    roles: tuple[str, ...] = ()

    def has_permission(self, request: Any, view: Any) -> bool:
        return has_role(request.user, *self.roles)


class IsCoordinator(_RolePermission):
    roles = (COORDINATOR,)


class IsClinician(_RolePermission):
    roles = (CLINICIAN,)


class CanReadRuns(_RolePermission):
    roles = (COORDINATOR, CLINICIAN)
