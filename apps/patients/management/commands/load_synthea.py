from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from data.loaders import synthea
from data.loaders.registry import record_load, register_patients


class Command(BaseCommand):
    help = "Upload generated Synthea bundles to HAPI (idempotent PUTs) and register Patient rows."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--workers",
            type=int,
            default=1,
            help="Parallel bundle uploads. Large bundles need HAPI heap; keep low.",
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        report = synthea.load(workers=opts["workers"])
        labelled = [
            (pid, f"Synthea {name} {i:03d}")
            for name, ids in sorted(report.patients.items())
            for i, pid in enumerate(sorted(ids), start=1)
        ]
        created, existing = register_patients(synthea.SOURCE, labelled)
        payload = {
            "bundles": report.bundles,
            "fhir_created": report.result.created,
            "fhir_updated": report.result.updated,
            "patients": len(labelled),
            "patient_rows_created": created,
        }
        record_load("load_synthea", payload)
        self.stdout.write(
            f"synthea: {report.bundles} bundles, FHIR created={report.result.created} "
            f"updated={report.result.updated}, patients={len(labelled)} "
            f"(new rows {created}, existing {existing})"
        )
