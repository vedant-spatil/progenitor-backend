from django.conf import settings
from django.db import models


class ReviewDecision(models.TextChoices):
    APPROVE = "approve", "Approve"
    REJECT = "reject", "Reject"
    EDIT = "edit", "Approve with edits"


class Review(models.Model):
    """A clinician's decision on one shortlist item."""

    run = models.ForeignKey("runs.Run", on_delete=models.CASCADE, related_name="reviews")
    shortlist_item = models.ForeignKey("runs.ShortlistItem", on_delete=models.CASCADE, related_name="reviews")
    reviewer = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.PROTECT, related_name="reviews")
    decision = models.CharField(max_length=16, choices=ReviewDecision.choices)
    comment = models.TextField(blank=True)
    edits = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.shortlist_item_id}: {self.decision}"


class ReferralStatus(models.TextChoices):
    DRAFT = "draft", "Draft"
    EDITED = "edited", "Edited by clinician"
    DISCARDED = "discarded", "Discarded"


class ReferralDraft(models.Model):
    """Draft referral note for an approved shortlist item. Never sent anywhere."""

    shortlist_item = models.ForeignKey("runs.ShortlistItem", on_delete=models.CASCADE, related_name="referrals")
    review = models.ForeignKey(Review, on_delete=models.PROTECT, related_name="referrals")
    body = models.TextField()
    status = models.CharField(max_length=16, choices=ReferralStatus.choices, default=ReferralStatus.DRAFT)
    model = models.CharField(max_length=128, blank=True)
    edited_by = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        null=True,
        blank=True,
        on_delete=models.PROTECT,
        related_name="edited_referrals",
    )
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"referral {self.pk} ({self.status})"
