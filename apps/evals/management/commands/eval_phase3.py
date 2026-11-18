import asyncio
import os
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from agents.llm import get_llm
from apps.runs.llm_cache import DjangoLlmCache
from evals.phase3 import check_thresholds, eval_judge, eval_parser, write_report


class Command(BaseCommand):
    help = "Gate 3 live eval: Parser F1 on Chia, Judge accuracy on hand-labelled pairs, citation validity."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--n-chia", type=int, default=100)
        parser.add_argument("--allow-draft-labels", action="store_true", help="Baseline run before label review.")
        parser.add_argument("--no-fail", action="store_true", help="Report only; do not exit non-zero.")

    def handle(self, *args: Any, **opts: Any) -> None:
        llm = get_llm(cache=DjangoLlmCache(task="eval_phase3"))
        fhir = os.environ.get("FHIR_BASE_URL", "http://localhost:8080/fhir")

        async def run() -> dict[str, Any]:
            parser = await eval_parser(llm, n=opts["n_chia"])
            judge = await eval_judge(llm, fhir, require_reviewed=not opts["allow_draft_labels"])
            return {"model": llm.model, "provider": getattr(llm, "provider", ""), "parser": parser, "judge": judge}

        report = asyncio.run(run())
        report["usage"] = vars(llm.usage)
        report["failures"] = check_thresholds(report)
        path = write_report(report)
        self.stdout.write(path.with_suffix(".md").read_text())
        if report["failures"] and not opts["no_fail"]:
            raise CommandError("; ".join(report["failures"]))
