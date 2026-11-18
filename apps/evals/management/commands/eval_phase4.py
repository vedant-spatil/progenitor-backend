import asyncio
import os
from typing import Any

import httpx
from django.core.management.base import BaseCommand, CommandError, CommandParser

from agents.llm import get_llm
from data.loaders.common import SOURCE_TAG_SYSTEM
from evals.phase4 import PASS_RATE_GATE, load_profile, run_scenarios, write_report


def seeded_diabetes_patient(fhir: str) -> str:
    """A Synthea patient with type 2 diabetes and an HbA1c on record."""
    obs = httpx.get(
        f"{fhir}/Observation",
        params={"code": "http://loinc.org|4548-4", "_tag": f"{SOURCE_TAG_SYSTEM}|synthea", "_count": "1"},
        headers={"Accept": "application/fhir+json"},
        timeout=60,
    ).json()
    return str(obs["entry"][0]["resource"]["subject"]["reference"].split("/")[1])


class Command(BaseCommand):
    help = "Gate 4 live scenario suite: LLM supervisor decisions (each scenario N times, majority counts)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--runs", type=int, default=3)
        parser.add_argument("--no-fail", action="store_true")

    def handle(self, *args: Any, **opts: Any) -> None:
        fhir = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir").rstrip("/")
        loaded = asyncio.run(load_profile(fhir, seeded_diabetes_patient(fhir)))
        llm = get_llm(cache=None)  # no cache: repeated runs must be independent samples
        results = run_scenarios(llm, loaded["profile"], loaded["patient"], runs=opts["runs"])
        path, rate, phi = write_report(llm.model, results, vars(llm.usage))
        self.stdout.write(open(path).read())
        failures = []
        if rate < PASS_RATE_GATE:
            failures.append(f"pass rate {rate:.2f} < {PASS_RATE_GATE}")
        if phi:
            failures.append(f"{phi} PHI violations in decision args")
        if failures and not opts["no_fail"]:
            raise CommandError("; ".join(failures))
