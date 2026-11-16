from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from data.loaders import mimic
from data.loaders.registry import record_load, register_patients


class Command(BaseCommand):
    help = "Download (checksum-verified) and load the MIMIC-IV FHIR demo into HAPI; register Patient rows."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument(
            "--full", action="store_true", help="Also load chart events, administrations, micro."
        )

    def handle(self, *args: Any, **opts: Any) -> None:
        report = mimic.load(full=opts["full"])
        labelled = [(pid, f"MIMIC {i:03d}") for i, pid in enumerate(sorted(report.patient_ids), start=1)]
        created, existing = register_patients(mimic.SOURCE, labelled)
        payload = {
            "files": len(report.files),
            "resources": sum(report.files.values()),
            "fhir_created": report.result.created,
            "fhir_updated": report.result.updated,
            "patients": len(labelled),
            "patient_rows_created": created,
            "full": opts["full"],
        }
        record_load("load_mimic", payload)
        self.stdout.write(
            f"mimic: {sum(report.files.values())} resources from {len(report.files)} files, "
            f"FHIR created={report.result.created} updated={report.result.updated}, "
            f"patients={len(labelled)} (new rows {created}, existing {existing})"
        )
