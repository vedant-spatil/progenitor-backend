from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from data.synthea.generate import generate


class Command(BaseCommand):
    help = "Generate the Synthea dev cohort (data/synthea/cohort.json) into data/external/synthea/."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--force", action="store_true", help="Regenerate even if output exists.")
        parser.add_argument("--only", nargs="*", help="Condition names to generate.")

    def handle(self, *args: Any, **opts: Any) -> None:
        for r in generate(force=opts["force"], only=opts["only"]):
            state = "skipped (present)" if r.skipped else "generated"
            self.stdout.write(f"{r.name}: {r.patients} patients {state}")
