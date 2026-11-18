from unittest import mock

import pytest
from django.apps import apps
from django.urls import reverse
from rest_framework.test import APIClient

pytestmark = pytest.mark.smoke

EXPECTED_APPS = ["accounts", "patients", "trials", "runs", "reviews", "audit", "evals"]


def test_all_domain_apps_installed() -> None:
    labels = {cfg.label for cfg in apps.get_app_configs()}
    assert set(EXPECTED_APPS) <= labels


@pytest.mark.django_db
def test_health_ok_when_dependencies_up() -> None:
    with mock.patch("progenitor.health.check_redis", return_value=True):
        resp = APIClient().get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "checks": {"database": True, "redis": True}}


@pytest.mark.django_db
def test_health_degraded_when_redis_down() -> None:
    with mock.patch("progenitor.health.check_redis", return_value=False):
        resp = APIClient().get("/api/health")
    assert resp.status_code == 503
    assert resp.json()["status"] == "degraded"


def test_api_requires_auth_by_default() -> None:
    from django.conf import settings

    assert settings.REST_FRAMEWORK["DEFAULT_PERMISSION_CLASSES"] == ["rest_framework.permissions.IsAuthenticated"]


@pytest.mark.django_db
def test_jwt_token_flow() -> None:
    from django.contrib.auth import get_user_model

    get_user_model().objects.create_user(username="coord", password="a-long-test-password")
    client = APIClient()
    resp = client.post(
        reverse("token_obtain_pair"),
        {"username": "coord", "password": "a-long-test-password"},
        format="json",
    )
    assert resp.status_code == 200
    body = resp.json()
    assert {"access", "refresh"} <= body.keys()

    bad = client.post(reverse("token_obtain_pair"), {"username": "coord", "password": "wrong"}, format="json")
    assert bad.status_code == 401


def test_celery_ping_task_eager() -> None:
    from progenitor.tasks import ping

    assert ping.apply().get(timeout=5) == "pong"


def test_celery_worker_registers_ping() -> None:
    """The worker must find the task on its own. Run in a fresh interpreter so an
    import done by another test cannot hide a missing registration."""
    import subprocess
    import sys
    from pathlib import Path

    code = (
        "from progenitor import celery_app; "
        "celery_app.loader.import_default_modules(); "
        "print('progenitor.ping' in celery_app.tasks)"
    )
    out = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert out.stdout.strip().splitlines()[-1] == "True", out.stderr
