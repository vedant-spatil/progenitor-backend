from typing import Any

from django.contrib import admin
from django.http import HttpRequest

from .models import AuditEvent


@admin.register(AuditEvent)
class AuditEventAdmin(admin.ModelAdmin[AuditEvent]):
    """Read-only: audit events are append-only."""

    list_display = ("created_at", "kind", "node", "run", "actor")
    list_filter = ("kind",)
    readonly_fields = ("created_at", "kind", "node", "run", "actor", "payload")

    def has_add_permission(self, request: HttpRequest) -> bool:
        return False

    def has_change_permission(self, request: HttpRequest, obj: Any = None) -> bool:
        return False

    def has_delete_permission(self, request: HttpRequest, obj: Any = None) -> bool:
        return False
