"""Gate runner with visible progress: `make gate PHASE=<n>` -> `python scripts/gate.py <n>`.

Gate n runs its own steps after its parent gate's (same chain as the Makefile: 1 and 2 build on 0,
3 on 2, 4 on 3, 5 on 2, 6 on 5). For each step it prints
`[ 5/14 ] unit tests ............ PASS  31s`, keeps full output in a log file, and updates a
status JSON after every step (read by `make gate-status`). Stops at the first failure and
shows the tail of that step's output. `make gate-report` prints the last summary as markdown.

Files: data/external/gates/gate-<n>-<UTC timestamp>.log / .json, plus last.json.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess  # nosec B404
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
GATES_DIR = BACKEND / "data" / "external" / "gates"
FRONTEND = os.environ.get("FRONTEND_DIR", "../frontend")


@dataclass(frozen=True)
class Step:
    name: str
    cmd: str


DEPLOY_CHECK = (
    "DJANGO_DEBUG=0 DJANGO_SECRET_KEY=\"$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')\" "
    "DATABASE_URL=sqlite:///:memory: uv run python manage.py check --deploy --fail-level WARNING"
)

# Steps per gate; commands run in backend/. PARENT mirrors the Makefile gate dependencies.
GATES: dict[int, list[Step]] = {
    0: [
        Step("install backend deps", "uv sync --frozen"),
        Step("ruff lint + format", "uv run ruff check . && uv run ruff format --check ."),
        Step("mypy", "DJANGO_DEBUG=1 uv run mypy ."),
        Step("django check", "DJANGO_DEBUG=1 uv run python manage.py check"),
        Step("django deploy check", DEPLOY_CHECK),
        Step("bandit", "uv run bandit -q -c pyproject.toml -r ."),
        Step("smoke tests", "uv run pytest tests/unit -k smoke -q"),
        Step("frontend install", f"cd {FRONTEND} && npm ci --no-audit --no-fund"),
        Step("frontend lint + types", f"cd {FRONTEND} && npm run lint && npm run typecheck"),
        Step("frontend build", f"cd {FRONTEND} && npm run build"),
        Step("stack up (compose, healthy)", "make -s stack-up"),
        Step("integration tests", "make -s integration-tests"),
    ],
    1: [
        Step("migrate", "uv run python manage.py migrate --noinput"),
        Step("data inventory", "uv run python manage.py data_inventory"),
        Step("data tests (counts, idempotent loads)", "make -s data-tests"),
    ],
    2: [Step("all unit tests (cassettes only)", "uv run pytest tests/unit -q --record-mode=none")],
    3: [
        Step("profiler exact match", "make -s profiler-exact-test"),
        Step("phase 3 eval (parser, judge, citations)", "uv run python manage.py eval_phase3"),
    ],
    4: [Step("phase 4 supervisor scenarios", "uv run python manage.py eval_phase4")],
    5: [
        Step("migrations applied", "uv run python manage.py migrate --noinput"),
        Step(
            "phase 5 API, approval, audit (unit)",
            "uv run pytest tests/unit/test_runs_api.py tests/unit/test_audit_wrappers.py -q",
        ),
        Step("phase 5 runs on Postgres + Redis (integration)", "make -s phase5-integration-tests"),
    ],
    6: [
        Step("phase 6 metrics, cache, reproducibility (unit)", "uv run pytest tests/unit/test_phase6_eval.py -q"),
        Step(
            "phase 6 eval replay vs thresholds ($0)",
            "uv run python manage.py eval_phase6 --replay --require-thresholds",
        ),
    ],
}


# Gate 5 builds on gate 2, not 4: the live LLM evals (gates 3-4, ~1 h of Bedrock calls) gate the agent
# changes; Phase 5 adds no prompt or agent logic, and its tests use the scripted LLM.
# Gate 6 replays the frozen baseline from the cache (no LLM spend), on top of gate 5.
PARENT: dict[int, int | None] = {0: None, 1: 0, 2: 0, 3: 2, 4: 3, 5: 2, 6: 5}


@dataclass
class StepResult:
    index: int
    name: str
    status: str = "pending"  # pending | running | pass | fail | not_run
    seconds: float = 0.0


@dataclass
class GateRun:
    phase: int
    started: str
    branch: str
    commit: str
    log: str
    steps: list[StepResult] = field(default_factory=list)
    status: str = "running"  # running | pass | fail
    finished: str = ""

    def save(self, path: Path) -> None:
        data = json.dumps(asdict(self), indent=2)
        path.write_text(data)
        (GATES_DIR / "last.json").write_text(data)


def git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=BACKEND, capture_output=True, text=True, timeout=10)  # noqa: S603, S607  # nosec B603 B607
        return out.stdout.strip() or "-"
    except (OSError, subprocess.SubprocessError):
        return "-"


def bar(done: int, total: int, width: int = 24) -> str:
    filled = round(width * done / total) if total else width
    return "█" * filled + "░" * (width - filled)


def fmt(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s" if m else f"{s}s"


def steps_for(phase: int) -> list[Step]:
    if phase not in GATES:
        raise SystemExit(f"no gate for phase {phase}; known: {sorted(GATES)}")
    parent = PARENT.get(phase)
    return (steps_for(parent) if parent is not None else []) + GATES[phase]


def run(phase: int) -> int:
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    log_path = GATES_DIR / f"gate-{phase}-{stamp}.log"
    json_path = log_path.with_suffix(".json")
    steps = steps_for(phase)
    gate = GateRun(
        phase=phase,
        started=datetime.now(UTC).isoformat(timespec="seconds"),
        branch=git("rev-parse", "--abbrev-ref", "HEAD"),
        commit=git("rev-parse", "--short", "HEAD"),
        log=str(log_path),
        steps=[StepResult(i + 1, s.name) for i, s in enumerate(steps)],
    )
    gate.save(json_path)
    total = len(steps)
    width = shutil.get_terminal_size((100, 20)).columns
    print(f"GATE {phase}  ·  {total} steps  ·  branch {gate.branch} @ {gate.commit}  ·  log {log_path}")
    t_gate = time.monotonic()
    with log_path.open("w") as log:
        for i, step in enumerate(steps):
            res = gate.steps[i]
            res.status = "running"
            gate.save(json_path)
            label = f"[{i + 1:>2}/{total}] {step.name} "
            print(f"\r{bar(i, total)} {label}{'.' * max(2, 46 - len(label))} running", end="", flush=True)
            log.write(f"\n===== [{i + 1}/{total}] {step.name}\n$ {step.cmd}\n")
            log.flush()
            t0 = time.monotonic()
            proc = subprocess.run(step.cmd, shell=True, cwd=BACKEND, stdout=log, stderr=subprocess.STDOUT)  # noqa: S602  # nosec B602
            res.seconds = round(time.monotonic() - t0, 1)
            res.status = "pass" if proc.returncode == 0 else "fail"
            gate.save(json_path)
            mark = "PASS" if res.status == "pass" else "FAIL"
            line = f"{bar(i + 1, total)} {label}{'.' * max(2, 46 - len(label))} {mark}  {fmt(res.seconds)}"
            print("\r" + line.ljust(min(width, 110)))
            if res.status == "fail":
                for later in gate.steps[i + 1 :]:
                    later.status = "not_run"
                break
    gate.status = "pass" if all(s.status == "pass" for s in gate.steps) else "fail"
    gate.finished = datetime.now(UTC).isoformat(timespec="seconds")
    gate.save(json_path)
    print(summary(gate, time.monotonic() - t_gate))
    if gate.status == "fail":
        failed = next(s for s in gate.steps if s.status == "fail")
        print(f"\nFailed at step {failed.index}: {failed.name}. Last lines of its output:\n")
        print(tail_of_step(log_path, failed.index))
    return 0 if gate.status == "pass" else 1


def tail_of_step(log_path: Path, index: int, lines: int = 40) -> str:
    text = log_path.read_text(errors="replace")
    part = text.split(f"===== [{index}/")[-1]
    return "\n".join(part.splitlines()[-lines:])


def summary(gate: GateRun, elapsed: float | None = None) -> str:
    done = sum(s.status == "pass" for s in gate.steps)
    total = len(gate.steps)
    head = f"\nGATE {gate.phase}: {gate.status.upper()}  ·  {done}/{total} steps passed"
    if elapsed is not None:
        head += f"  ·  {fmt(elapsed)}"
    rows = [
        f"  {s.index:>2}. {s.name:<44} {s.status.upper():<8} {fmt(s.seconds) if s.seconds else ''}" for s in gate.steps
    ]
    return "\n".join([head, *rows])


def status() -> int:
    path = GATES_DIR / "last.json"
    if not path.exists():
        print("no gate run yet")
        return 1
    d = json.loads(path.read_text())
    steps = [StepResult(**s) for s in d.pop("steps")]
    gate = GateRun(**d, steps=steps)
    done = sum(s.status in ("pass", "fail") for s in steps)
    current = next((s for s in steps if s.status == "running"), None)
    progress = f"{bar(done, len(steps))}  {done}/{len(steps)} done"
    print(f"GATE {gate.phase} [{gate.status}]  {progress}  ·  started {gate.started}")
    print(f"branch {gate.branch} @ {gate.commit}  ·  log {gate.log}")
    if current:
        print(f"now: [{current.index}/{len(steps)}] {current.name}")
    print(summary(gate))
    return 0


def report() -> int:
    path = GATES_DIR / "last.json"
    if not path.exists():
        print("no gate run yet")
        return 1
    d = json.loads(path.read_text())
    lines = [
        f"### Gate {d['phase']}: {d['status'].upper()}",
        f"branch `{d['branch']}` @ `{d['commit']}` · started {d['started']} · finished {d['finished'] or '-'}",
        "",
        "| # | Step | Status | Time |",
        "|---:|---|---|---:|",
    ]
    lines += [
        f"| {s['index']} | {s['name']} | {s['status']} | {fmt(s['seconds']) if s['seconds'] else ''} |"
        for s in d["steps"]
    ]
    print("\n".join(lines))
    return 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[1] == "status":
        return status()
    if len(argv) >= 2 and argv[1] == "report":
        return report()
    if len(argv) >= 2 and argv[1].isdigit():
        return run(int(argv[1]))
    print("usage: gate.py <phase> | status | report")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
