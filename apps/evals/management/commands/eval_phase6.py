"""Phase 6 eval: TrialGPT cohorts, three systems, metrics, report.

    manage.py eval_phase6 --build-snapshot            # fetch pooled trials once (no LLM)
    manage.py eval_phase6 --per-cohort 10             # pilot: live LLM for cache misses only
    manage.py eval_phase6 --per-cohort 20             # baseline (reuses the pilot's cached replies)
    manage.py eval_phase6 --replay --require-thresholds   # gate: cache only, $0, fails on any miss

The LLM comes from LLM_PROVIDER / LLM_MODEL like every other run.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from django.core.management.base import BaseCommand, CommandError, CommandParser

from agents.guards import Limits
from agents.llm import get_llm
from data.loaders.common import DATA_DIR
from evals.phase6 import cohorts, metrics, report, runner, snapshot
from evals.phase6.cache import DjangoStore, EvalLLM, ReplayMiss


class Command(BaseCommand):
    help = "Phase 6 eval over TrialGPT cohorts (supervisor vs fixed pipeline vs BM25)."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--per-cohort", type=int, default=20)
        parser.add_argument("--pool", type=int, default=10, help="Candidate trials per patient.")
        parser.add_argument("--seed", type=int, default=7)
        parser.add_argument("--cohorts", default=",".join(cohorts.COHORTS))
        parser.add_argument("--systems", default=",".join(runner.SYSTEMS))
        parser.add_argument("--build-snapshot", action="store_true", help="Fetch missing pooled trials, then stop.")
        parser.add_argument("--replay", action="store_true", help="Cache only: no LLM calls; a miss is an error.")
        parser.add_argument("--require-thresholds", action="store_true", help="Exit 1 unless thresholds pass.")

    def handle(self, *args: Any, **opts: Any) -> None:
        chosen = cohorts.select(
            opts["per_cohort"], opts["pool"], opts["seed"], tuple(c for c in opts["cohorts"].split(",") if c)
        )
        wanted = sorted({n for p in chosen for n in p.pool})
        if opts["build_snapshot"]:
            trials = snapshot.build_sync(wanted)
            missing = sum(1 for n in wanted if "missing" in trials.get(n, {"missing": 1}))
            self.stdout.write(f"snapshot: {len(wanted)} pooled trials, {missing} missing -> {snapshot.SNAPSHOT_PATH}")
            return
        trials = snapshot.load()
        absent = [n for n in wanted if n not in trials]
        if absent:
            raise CommandError(f"{len(absent)} pooled trials are not in the snapshot; run --build-snapshot first")

        llm = EvalLLM(get_llm(cache=None), DjangoStore(), replay=opts["replay"])
        systems = tuple(s for s in opts["systems"].split(",") if s)
        limits = Limits.from_env()
        results = []
        for i, p in enumerate(chosen, start=1):
            try:
                results.append(asyncio.run(runner.run_patient(llm, p, trials, systems, limits)))
            except ReplayMiss as exc:
                raise CommandError(f"replay miss at {p.topic_id}: {exc}. Inputs changed since the baseline.") from exc
            self.stdout.write(f"[{i}/{len(chosen)}] {p.cohort} {p.topic_id} done")

        price = runner.price_for(llm.model)
        table = runner.score(results, price)
        thresholds = json.loads((DATA_DIR / "eval_thresholds.json").read_text()).get("phase6")
        fails = runner.check_thresholds(table, thresholds)
        spent = metrics.cost_usd(llm.spent.tokens_in, llm.spent.tokens_out, price)
        config = {
            "model": llm.model,
            "patients": len(chosen),
            "scored": sum(not any(sr.error for sr in r.systems.values()) for r in results),
            "per_cohort": opts["per_cohort"],
            "pool_size": opts["pool"],
            "seed": opts["seed"],
            "snapshot": snapshot.digest({n: trials[n] for n in wanted}),
            "replay": opts["replay"],
            "spent_calls": llm.spent.calls,
            "spent_tokens_in": llm.spent.tokens_in,
            "spent_tokens_out": llm.spent.tokens_out,
            "spent_usd": round(spent, 4) if spent is not None else None,
            "price": price,
        }
        md, _ = report.write(config, table, fails, runner.results_json(results))
        self.stdout.write(md.read_text())
        if opts["require_thresholds"] and fails:
            raise CommandError("Phase 6 thresholds not met")
