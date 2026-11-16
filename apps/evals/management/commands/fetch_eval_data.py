from typing import Any

from django.core.management.base import BaseCommand

from data.loaders.eval_data import fetch_all


class Command(BaseCommand):
    help = "Download and verify TrialGPT cohorts and Chia (pinned in data/eval_manifest.json)."

    def handle(self, *args: Any, **opts: Any) -> None:
        report = fetch_all()
        for c in report.cohorts:
            self.stdout.write(
                f"{c.name}: {c.patients} patients (expected {c.expected_patients}), "
                f"{c.qrels_rows} qrels over {c.trials} trials"
            )
        if report.chia:
            self.stdout.write(
                f"chia: {report.chia.trials} trials (expected {report.chia.expected_trials}), "
                f"{report.chia.entities} entities, {report.chia.relations} relations"
            )
