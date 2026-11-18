"""Gate 3 live evals: Criteria Parser on Chia, Eligibility Judge on hand-labelled pairs, citation validity.

Runs with a real LLM (temperature 0). Replies are cached in Postgres, so reruns cost no tokens.
Results go to docs/eval-reports/phase3-<date>.json/.md; thresholds live in data/eval_thresholds.json
("set after baseline", see docs/decisions/).
"""

from __future__ import annotations

import asyncio
import json
import random
import re
import zipfile
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from agents.fhir_reader import HttpFhirReader
from agents.llm import StructuredLLM
from agents.schemas import Criterion, CriterionKind, EntityType, Verdict, VerdictStatus
from agents.workers.llm_workers import judge_trial, parse_criteria
from agents.workers.profiler import build_profile
from data.loaders.common import BACKEND_DIR, DATA_DIR
from data.loaders.eval_data import EVAL_DIR, load_manifest

REPORT_DIR = BACKEND_DIR / "docs" / "eval-reports"
HANDLABEL = DATA_DIR / "handlabel" / "handlabel_v1.json"
THRESHOLDS = DATA_DIR / "eval_thresholds.json"

# Chia entity types we expect the Parser to extract, mapped to our EntityType values.
CHIA_TYPES: dict[str, set[EntityType]] = {
    "Condition": {EntityType.CONDITION},
    "Drug": {EntityType.DRUG},
    "Procedure": {EntityType.PROCEDURE},
    "Measurement": {EntityType.LAB, EntityType.OBSERVATION},
    "Observation": {EntityType.OBSERVATION, EntityType.CONDITION},
    "Person": {EntityType.DEMOGRAPHIC},
}
TOKEN = re.compile(r"[a-z0-9]+")


# ---------------- Parser on Chia ----------------


@dataclass
class GoldCriterion:
    nct_id: str
    kind: str
    text: str
    entities: list[tuple[str, str]]  # (chia type, text)


def load_chia_criteria(n: int = 100, seed: int = 7) -> list[GoldCriterion]:
    """n criterion lines (with at least one evaluated entity), sampled reproducibly."""
    path = EVAL_DIR / "chia" / load_manifest()["chia"]["file"]
    items: list[GoldCriterion] = []
    with zipfile.ZipFile(path) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith(".txt") and Path(n).name.startswith("NCT"))
        for name in names:
            text = zf.read(name).decode("utf-8", errors="replace")
            ann = zf.read(name[:-4] + ".ann").decode("utf-8", errors="replace")
            spans: list[tuple[int, int, str, str]] = []
            for line in ann.splitlines():
                if not line.startswith("T"):
                    continue
                _, meta, surface = [*line.split("\t"), "", ""][:3]
                parts = meta.split(" ")
                etype = parts[0]
                if etype not in CHIA_TYPES:
                    continue
                try:
                    start, end = int(parts[1]), int(parts[-1].split(";")[-1])
                except ValueError:
                    continue
                spans.append((start, end, etype, surface))
            offset = 0
            stem = Path(name).name
            for line in text.split("\n"):
                start, end = offset, offset + len(line)
                offset = end + 1
                ents = [(t, s) for (a, b, t, s) in spans if a >= start and b <= end]
                if line.strip() and ents:
                    items.append(
                        GoldCriterion(
                            stem.split("_")[0], "exclusion" if "_exc" in stem else "inclusion", line.strip(), ents
                        )
                    )
    rng = random.Random(seed)  # noqa: S311  # nosec B311 - reproducible sampling
    return rng.sample(items, min(n, len(items)))


def _tokens(s: str) -> set[str]:
    return set(TOKEN.findall(s.lower()))


def _match(gold: tuple[str, str], pred_text: str, pred_type: EntityType) -> bool:
    gtype, gtext = gold
    if pred_type not in CHIA_TYPES[gtype]:
        return False
    g, p = _tokens(gtext), _tokens(pred_text)
    if not g or not p:
        return False
    return len(g & p) / len(g | p) >= 0.5 or g <= p or p <= g


@dataclass
class F1:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0


def score_entities(gold: list[tuple[str, str]], pred: list[tuple[str, EntityType]], score: F1) -> None:
    """Greedy one-to-one matching; predicted types outside the evaluated set are ignored."""
    evaluated = set().union(*CHIA_TYPES.values())
    preds = [p for p in pred if p[1] in evaluated]
    used: set[int] = set()
    for g in gold:
        hit = next((i for i, (t, ty) in enumerate(preds) if i not in used and _match(g, t, ty)), None)
        if hit is None:
            score.fn += 1
        else:
            used.add(hit)
            score.tp += 1
    score.fp += len(preds) - len(used)


async def eval_parser(llm: StructuredLLM, n: int = 100) -> dict[str, Any]:
    score = F1()
    errors = 0
    for item in load_chia_criteria(n):
        header = "Exclusion Criteria:" if item.kind == "exclusion" else "Inclusion Criteria:"
        try:
            crits = await parse_criteria(llm, f"{header}\n{item.text}")
        except Exception:  # noqa: BLE001 - count failures as misses
            errors += 1
            score.fn += len(item.entities)
            continue
        preds = [(e.text, e.type) for c in crits for e in c.entities]
        score_entities(item.entities, preds, score)
    return {
        "n": n,
        "precision": score.precision,
        "recall": score.recall,
        "f1": score.f1,
        "errors": errors,
        **asdict(score),
    }


# ---------------- Judge on hand-labelled pairs ----------------


@dataclass
class JudgeResult:
    n: int = 0
    correct: int = 0
    by_status: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    met_or_not_met: int = 0
    valid_citations: int = 0
    downgraded: int = 0
    evidence_hits: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.n if self.n else 0.0

    @property
    def citation_validity(self) -> float:
        return self.valid_citations / self.met_or_not_met if self.met_or_not_met else 1.0


def load_handlabel(require_reviewed: bool) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads(HANDLABEL.read_text())
    if require_reviewed and doc.get("status") != "reviewed":
        raise RuntimeError(f"{HANDLABEL} is '{doc.get('status')}'; a person must review it first (status: reviewed)")
    return doc


async def eval_judge(llm: StructuredLLM, fhir_base: str, require_reviewed: bool = True) -> dict[str, Any]:
    doc = load_handlabel(require_reviewed)
    as_of = date.fromisoformat(doc["as_of"])
    by_patient: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in doc["pairs"]:
        by_patient[pair["patient_id"]].append(pair)
    res = JudgeResult()
    reader = HttpFhirReader(fhir_base)
    try:
        for pid, pairs in sorted(by_patient.items()):
            profile = await build_profile(reader, pid, as_of=as_of)
            valid = profile.resource_ids()
            criteria = [
                Criterion(index=i, kind=CriterionKind(p["criterion"]["kind"]), source_text=p["criterion"]["text"])
                for i, p in enumerate(pairs)
            ]
            try:
                verdicts = await asyncio.to_thread(judge_trial, llm, profile, criteria, as_of=as_of)
            except Exception as exc:  # noqa: BLE001
                res.errors.append(f"{pid}: {exc.__class__.__name__}")
                verdicts = [Verdict(criterion_index=i, status=VerdictStatus.UNKNOWN) for i in range(len(pairs))]
            for pair, v in zip(pairs, verdicts, strict=True):
                expected = pair["expected_status"]
                res.n += 1
                res.correct += v.status == expected
                res.by_status[expected][v.status] += 1
                res.downgraded += v.downgraded
                if v.status in (VerdictStatus.MET, VerdictStatus.NOT_MET):
                    res.met_or_not_met += 1
                    res.valid_citations += bool(v.evidence_refs) and all(r in valid for r in v.evidence_refs)
                    res.evidence_hits += bool(set(v.evidence_refs) & set(pair["expected_evidence"]))
    finally:
        await reader.aclose()
    return {
        "n": res.n,
        "accuracy": res.accuracy,
        "citation_validity": res.citation_validity,
        "met_or_not_met": res.met_or_not_met,
        "downgraded_by_guard": res.downgraded,
        "evidence_matches_label": res.evidence_hits,
        "confusion": {k: dict(v) for k, v in res.by_status.items()},
        "labels_status": doc.get("status"),
        "errors": res.errors,
    }


# ---------------- Report ----------------


def check_thresholds(report: dict[str, Any]) -> list[str]:
    """Failures against data/eval_thresholds.json. Citation validity is always 100%."""
    failures = []
    if report["judge"]["citation_validity"] < 1.0:
        failures.append(f"citation validity {report['judge']['citation_validity']:.3f} < 1.0")
    if not THRESHOLDS.exists():
        failures.append("thresholds not set: record the baseline in data/eval_thresholds.json + an ADR")
        return failures
    t = json.loads(THRESHOLDS.read_text())["phase3"]
    if report["parser"]["f1"] < t["parser_entity_f1"]:
        failures.append(f"parser F1 {report['parser']['f1']:.3f} < {t['parser_entity_f1']}")
    if report["judge"]["accuracy"] < t["judge_accuracy"]:
        failures.append(f"judge accuracy {report['judge']['accuracy']:.3f} < {t['judge_accuracy']}")
    return failures


def write_report(report: dict[str, Any]) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    path = REPORT_DIR / f"phase3-{stamp}.json"
    path.write_text(json.dumps(report, indent=2, default=str) + "\n")
    p, j = report["parser"], report["judge"]
    md = [
        f"# Phase 3 eval ({stamp} UTC)",
        "",
        f"Model: `{report['model']}` (temperature 0). Labels: {j['labels_status']}.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Parser entity F1 (Chia, n={p['n']}) | {p['f1']:.3f} (P {p['precision']:.3f}, R {p['recall']:.3f}) |",
        f"| Judge accuracy (hand-labelled, n={j['n']}) | {j['accuracy']:.3f} |",
        f"| Citation validity (met/not_met = {j['met_or_not_met']}) | {j['citation_validity']:.3f} |",
        f"| Verdicts downgraded by citation guard | {j['downgraded_by_guard']} |",
        "",
        f"Judge confusion (expected -> predicted): `{json.dumps(j['confusion'])}`",
        "",
        f"Threshold check: {'; '.join(report['failures']) or 'all met'}",
    ]
    path.with_suffix(".md").write_text("\n".join(md) + "\n")
    return path
