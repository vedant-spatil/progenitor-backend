from django.contrib import admin

from .models import Criterion, Trial


class CriterionInline(admin.TabularInline[Criterion, Trial]):
    model = Criterion
    extra = 0
    fields = ("position", "kind", "entity", "operator", "value", "unit", "temporal_window")


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin[Trial]):
    list_display = ("nct_id", "title", "overall_status", "source", "fetched_at")
    list_filter = ("overall_status", "source")
    search_fields = ("nct_id", "title")
    inlines = [CriterionInline]


@admin.register(Criterion)
class CriterionAdmin(admin.ModelAdmin[Criterion]):
    list_display = ("trial", "position", "kind", "entity", "operator", "value", "unit")
    list_filter = ("kind",)
    search_fields = ("trial__nct_id", "entity", "source_text")
