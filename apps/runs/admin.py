from django.contrib import admin

from .models import Decision, Run, ShortlistItem, Verdict


class DecisionInline(admin.TabularInline[Decision, Run]):
    model = Decision
    extra = 0
    fields = ("step", "action", "guard_outcome", "confidence", "rationale")
    readonly_fields = fields
    can_delete = False


@admin.register(Run)
class RunAdmin(admin.ModelAdmin[Run]):
    list_display = ("id", "patient", "status", "steps_used", "created_by", "created_at")
    list_filter = ("status",)
    inlines = [DecisionInline]


@admin.register(Decision)
class DecisionAdmin(admin.ModelAdmin[Decision]):
    list_display = ("run", "step", "action", "guard_outcome", "confidence", "latency_ms")
    list_filter = ("action", "guard_outcome")


@admin.register(Verdict)
class VerdictAdmin(admin.ModelAdmin[Verdict]):
    list_display = ("run", "trial", "criterion", "status", "downgraded")
    list_filter = ("status", "downgraded")


@admin.register(ShortlistItem)
class ShortlistItemAdmin(admin.ModelAdmin[ShortlistItem]):
    list_display = ("run", "rank", "trial", "status")
    list_filter = ("status",)
