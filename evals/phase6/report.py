"""Phase 6 report: markdown + JSON in docs/eval-reports/."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.loaders.common import BACKEND_DIR

REPORT_DIR = BACKEND_DIR / "docs" / "eval-reports"
COLUMNS = [
    ("ndcg@10", "NDCG@10"),
    ("ndcg@5", "NDCG@5"),
    ("p@5", "P@5"),
    ("mrr", "MRR"),
    ("unknown_rate", "Unknown rate"),
    ("citation_validity", "Citation validity"),
    ("tokens_per_patient", "Tokens / patient"),
    ("cost_per_patient_usd", "Cost / patient (USD)"),
    ("seconds_per_patient", "Seconds / patient"),
    ("errors", "Errors"),
]


def _cell(v: Any) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.4f}".rstrip("0").rstrip(".") if v else "0"
    return str(v)


def write(
    config: dict[str, Any], table: dict[str, Any], fails: list[str], patients: list[dict[str, Any]]
) -> tuple[Path, Path]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    md = REPORT_DIR / f"phase6-{stamp}.md"
    js = md.with_suffix(".json")
    lines = [
        f"# Phase 6 eval ({stamp} UTC)",
        "",
        f"Model `{config['model']}` · {config['patients']} patients ({config['per_cohort']} per cohort, seed "
        f"{config['seed']}) · pool {config['pool_size']} trials per patient · snapshot `{config['snapshot']}` · "
        f"{'replay (cache only)' if config['replay'] else 'live'}",
        "",
        f"Tokens bought this run: {config['spent_tokens_in']:,} in / {config['spent_tokens_out']:,} out "
        f"({config['spent_calls']} calls); cost {_cell(config['spent_usd'])} USD. Per-system figures below count "
        "shared cached calls as if each system ran alone.",
        "",
        "All systems rank the same fixed pool (stratified sample of each patient's TrialGPT judged trials). "
        "Relevance: 2 eligible, 1 excluded, 0 not relevant; P@5 and MRR count eligible trials only.",
    ]
    for cohort, systems in table.items():
        lines += ["", f"## {cohort}", "", "| System | " + " | ".join(h for _, h in COLUMNS) + " |"]
        lines.append("|---|" + "---:|" * len(COLUMNS))
        for name, row in systems.items():
            lines.append(f"| {name} | " + " | ".join(_cell(row.get(k)) for k, _ in COLUMNS) + " |")
    lines += ["", "## Threshold check", ""]
    lines += [f"- FAIL: {f}" for f in fails] if fails else ["All thresholds met."]
    md.write_text("\n".join(lines) + "\n")
    js.write_text(json.dumps({"config": config, "table": table, "fails": fails, "patients": patients}, indent=1))
    return md, js
