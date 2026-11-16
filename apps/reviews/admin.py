from django.contrib import admin

from .models import ReferralDraft, Review


@admin.register(Review)
class ReviewAdmin(admin.ModelAdmin[Review]):
    list_display = ("shortlist_item", "decision", "reviewer", "created_at")
    list_filter = ("decision",)


@admin.register(ReferralDraft)
class ReferralDraftAdmin(admin.ModelAdmin[ReferralDraft]):
    list_display = ("id", "shortlist_item", "status", "updated_at")
    list_filter = ("status",)
