"""API shapes for runs. The frontend mirrors these in lib/api/schemas.ts (Zod); change both together."""

from __future__ import annotations

from typing import Any

from rest_framework import serializers

from apps.patients.models import Patient
from apps.reviews.models import ReferralDraft, ReviewDecision
from apps.runs.models import Decision, Run, ShortlistItem, Verdict


class RunCreateSerializer(serializers.Serializer[Any]):
    patient_id = serializers.PrimaryKeyRelatedField(queryset=Patient.objects.all(), source="patient")
    as_of = serializers.DateField(required=False, help_text="Evaluate the record as of this date (default today).")


class DecisionSerializer(serializers.ModelSerializer[Decision]):
    class Meta:
        model = Decision
        fields = [
            "step",
            "attempt",
            "action",
            "args",
            "rationale",
            "state_refs",
            "confidence",
            "guard_outcome",
            "guard_detail",
            "created_at",
        ]


class RunSerializer(serializers.ModelSerializer[Run]):
    patient_id = serializers.IntegerField(source="patient.pk", read_only=True)
    patient_label = serializers.CharField(source="patient.label", read_only=True)
    decisions = DecisionSerializer(many=True, read_only=True)

    class Meta:
        model = Run
        fields = [
            "id",
            "patient_id",
            "patient_label",
            "status",
            "pending_question",
            "summary",
            "error",
            "steps_used",
            "tokens_in",
            "tokens_out",
            "created_at",
            "started_at",
            "finished_at",
            "decisions",
        ]


class VerdictSerializer(serializers.ModelSerializer[Verdict]):
    criterion_index = serializers.IntegerField(source="criterion.position")
    kind = serializers.CharField(source="criterion.kind")
    criterion_text = serializers.CharField(source="criterion.source_text")

    class Meta:
        model = Verdict
        fields = ["criterion_index", "kind", "criterion_text", "status", "evidence_refs", "reason"]


class ShortlistItemSerializer(serializers.ModelSerializer[ShortlistItem]):
    nct_id = serializers.CharField(source="trial.nct_id")
    title = serializers.CharField(source="trial.title")
    verdicts = serializers.SerializerMethodField()

    class Meta:
        model = ShortlistItem
        fields = ["nct_id", "title", "rank", "explanation", "counts", "status", "verdicts"]

    def get_verdicts(self, item: ShortlistItem) -> list[dict[str, Any]]:
        rows = Verdict.objects.filter(run_id=item.run_id, trial_id=item.trial_id).select_related("criterion")
        return list(VerdictSerializer(rows.order_by("criterion__position"), many=True).data)


class AnswerSerializer(serializers.Serializer[Any]):
    answer = serializers.CharField(max_length=2000, trim_whitespace=True)


class ReviewItemSerializer(serializers.Serializer[Any]):
    nct_id = serializers.CharField(max_length=16)
    decision = serializers.ChoiceField(choices=ReviewDecision.choices)
    comment = serializers.CharField(required=False, allow_blank=True, max_length=2000)
    edits = serializers.DictField(required=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if attrs["decision"] == ReviewDecision.EDIT and not attrs.get("edits"):
            raise serializers.ValidationError({"edits": "required when decision is 'edit'"})
        return attrs


class ReviewSerializer(serializers.Serializer[Any]):
    items = ReviewItemSerializer(many=True, allow_empty=False)

    def validate_items(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ids = [i["nct_id"] for i in items]
        if len(ids) != len(set(ids)):
            raise serializers.ValidationError("each trial may appear once")
        return items


class RunListSerializer(serializers.ModelSerializer[Run]):
    patient_id = serializers.IntegerField(source="patient.pk", read_only=True)
    patient_label = serializers.CharField(source="patient.label", read_only=True)

    class Meta:
        model = Run
        fields = ["id", "patient_id", "patient_label", "status", "created_at", "finished_at"]


class PatientSerializer(serializers.ModelSerializer[Patient]):
    """Synthetic display label and source only: no FHIR id, no profile."""

    class Meta:
        model = Patient
        fields = ["id", "label", "source"]


class ReferralUpdateSerializer(serializers.Serializer[Any]):
    body = serializers.CharField(required=False, max_length=20000, trim_whitespace=False)
    discard = serializers.BooleanField(required=False, default=False)

    def validate(self, attrs: dict[str, Any]) -> dict[str, Any]:
        if not attrs.get("discard") and not attrs.get("body", "").strip():
            raise serializers.ValidationError({"body": "required unless discarding"})
        return attrs


class ReferralSerializer(serializers.ModelSerializer[ReferralDraft]):
    nct_id = serializers.CharField(source="shortlist_item.trial.nct_id")
    approved_by = serializers.IntegerField(source="review.reviewer_id")

    class Meta:
        model = ReferralDraft
        fields = ["id", "nct_id", "body", "status", "approved_by", "created_at"]
