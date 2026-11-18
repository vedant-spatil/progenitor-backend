"""Run API (PLAN.md Phase 5). Roles: coordinator starts runs; clinician answers and reviews; both read."""

from __future__ import annotations

from typing import Any

from asgiref.sync import sync_to_async
from django.http import HttpRequest, HttpResponseBase, JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from rest_framework import status
from rest_framework.exceptions import AuthenticationFailed
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework_simplejwt.authentication import JWTAuthentication

from apps.accounts.roles import CLINICIAN, COORDINATOR, CanReadRuns, IsClinician, IsCoordinator, has_role
from apps.audit.models import AuditKind
from apps.audit.services import audit
from apps.reviews.models import ReferralDraft
from apps.runs import events, services
from apps.runs.models import Run
from apps.runs.serializers import (
    AnswerSerializer,
    ReferralSerializer,
    ReviewSerializer,
    RunCreateSerializer,
    RunSerializer,
    ShortlistItemSerializer,
)


def _conflict(exc: Exception) -> Response:
    return Response({"detail": str(exc)}, status=status.HTTP_409_CONFLICT)


class RunCreateView(APIView):
    permission_classes = [IsCoordinator]

    def post(self, request: Request) -> Response:
        ser = RunCreateSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        config = {"as_of": ser.validated_data["as_of"].isoformat()} if ser.validated_data.get("as_of") else {}
        run = services.start(ser.validated_data["patient"], request.user, config)
        return Response({"id": str(run.id), "status": run.status}, status=status.HTTP_201_CREATED)


class RunDetailView(APIView):
    permission_classes = [CanReadRuns]

    def get(self, request: Request, run_id: str) -> Response:
        run = get_object_or_404(Run.objects.select_related("patient").prefetch_related("decisions"), pk=run_id)
        return Response(RunSerializer(run).data)


class ShortlistView(APIView):
    permission_classes = [CanReadRuns]

    def get(self, request: Request, run_id: str) -> Response:
        run = get_object_or_404(Run, pk=run_id)
        items = run.shortlist.select_related("trial").order_by("rank")
        # Verdict reasons can restate record facts: reading them is a PHI read.
        audit(AuditKind.PHI_READ, run_id=str(run.id), actor_id=request.user.id, node="api_shortlist")
        return Response({"status": run.status, "items": ShortlistItemSerializer(items, many=True).data})


class AnswerView(APIView):
    permission_classes = [IsClinician]

    def post(self, request: Request, run_id: str) -> Response:
        get_object_or_404(Run, pk=run_id)
        ser = AnswerSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        try:
            run = services.answer(run_id, request.user, ser.validated_data["answer"])
        except services.RunStateError as exc:
            return _conflict(exc)
        return Response({"id": str(run.id), "status": run.status}, status=status.HTTP_202_ACCEPTED)


class ReviewView(APIView):
    permission_classes = [IsClinician]

    def post(self, request: Request, run_id: str) -> Response:
        get_object_or_404(Run, pk=run_id)
        ser = ReviewSerializer(data=request.data)
        ser.is_valid(raise_exception=True)
        try:
            run = services.review(run_id, request.user, ser.validated_data["items"])
        except services.RunStateError as exc:
            return _conflict(exc)
        except ValueError as exc:
            return Response({"items": [str(exc)]}, status=status.HTTP_400_BAD_REQUEST)
        return Response({"id": str(run.id), "status": run.status}, status=status.HTTP_202_ACCEPTED)


class ReferralsView(APIView):
    permission_classes = [CanReadRuns]

    def get(self, request: Request, run_id: str) -> Response:
        run = get_object_or_404(Run, pk=run_id)
        rows = ReferralDraft.objects.filter(shortlist_item__run=run).select_related("shortlist_item__trial", "review")
        audit(AuditKind.PHI_READ, run_id=str(run.id), actor_id=request.user.id, node="api_referrals")
        return Response({"status": run.status, "items": ReferralSerializer(rows, many=True).data})


# ---------------- SSE ----------------


def _authenticate(request: HttpRequest) -> Any:
    """JWT from the Authorization header (fetch-based SSE client; tokens never go in the URL)."""
    try:
        result = JWTAuthentication().authenticate(Request(request))
    except AuthenticationFailed:
        return None
    return result[0] if result else None


def _snapshot(request: HttpRequest, run_id: str) -> tuple[int, dict[str, Any]]:
    user = _authenticate(request)
    if user is None:
        return 401, {"detail": "Authentication credentials were not provided."}
    if not has_role(user, COORDINATOR, CLINICIAN):
        return 403, {"detail": "You do not have permission to perform this action."}
    run = Run.objects.filter(pk=run_id).first()
    if run is None:
        return 404, {"detail": "Not found."}
    return 200, {
        "status": run.status,
        "steps_used": run.steps_used,
        "pending_question": run.pending_question,
        "last_step": run.decisions.order_by("-step").values_list("step", flat=True).first(),
    }


async def run_events(request: HttpRequest, run_id: str) -> HttpResponseBase:
    """GET /api/runs/{id}/events: text/event-stream. Snapshot first, then live events until the run pauses or ends."""
    if request.method != "GET":
        return JsonResponse({"detail": "Method not allowed."}, status=405)
    code, body = await sync_to_async(_snapshot)(request, run_id)
    if code != 200:
        return JsonResponse(body, status=code)
    resp = StreamingHttpResponse(events.stream(run_id, body), content_type="text/event-stream")
    resp["Cache-Control"] = "no-cache"
    resp["X-Accel-Buffering"] = "no"
    return resp
