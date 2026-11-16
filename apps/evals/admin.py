from django.contrib import admin

from .models import EvalRun


@admin.register(EvalRun)
class EvalRunAdmin(admin.ModelAdmin[EvalRun]):
    list_display = ("id", "system", "dataset", "status", "cost_usd", "started_at")
    list_filter = ("system", "dataset", "status")
