from django.contrib import admin

from .models import Patient


@admin.register(Patient)
class PatientAdmin(admin.ModelAdmin[Patient]):
    list_display = ("id", "label", "source", "fhir_id", "profile_cached_at")
    list_filter = ("source",)
    search_fields = ("label", "fhir_id")
