from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from data.loaders.inventory import DEFAULT_REPORT, write_inventory


class Command(BaseCommand):
    help = "Count loaded data (HAPI, app tables, eval files) and write docs/eval-reports/data-inventory.md."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--out", default=str(DEFAULT_REPORT))

    def handle(self, *args: Any, **opts: Any) -> None:
        path = write_inventory(opts["out"])
        self.stdout.write(f"wrote {path}")
