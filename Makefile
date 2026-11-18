SHELL := /bin/bash
.DEFAULT_GOAL := help

# This Makefile lives in the backend repo. The frontend is a separate repo,
# checked out as a sibling folder (override with FRONTEND_DIR=...).
FRONTEND_DIR ?= ../frontend
# Stack: local (default) or ec2 (adds docker-compose.ec2.yml: CloudWatch logs, 3g HAPI heap).
STACK ?= $(shell grep -s '^STACK=' .env | tail -1 | cut -d= -f2)
STACK := $(or $(STACK),local)
# Tracing: langfuse (starts the `tracing` compose profile) | cloudwatch | off.
TRACING_MODE ?= $(shell grep -s '^TRACING_MODE=' .env | tail -1 | cut -d= -f2)
TRACING_MODE := $(or $(TRACING_MODE),langfuse)
COMPOSE_FILES := -f docker-compose.yml $(if $(filter ec2,$(STACK)),-f docker-compose.ec2.yml)
COMPOSE_PROFILE := $(if $(filter langfuse,$(TRACING_MODE)),--profile tracing)
COMPOSE := docker compose $(COMPOSE_FILES) $(COMPOSE_PROFILE)
export TRACING_MODE
FRONTEND := cd $(FRONTEND_DIR) &&
PHASE ?=
# Host port for HAPI FHIR; override in .env when 8080 is taken.
HAPI_HOST_PORT ?= $(shell grep -s '^HAPI_HOST_PORT=' .env | tail -1 | cut -d= -f2)
HAPI_HOST_PORT := $(or $(HAPI_HOST_PORT),8080)

# Integration tests run Django against the compose Postgres (needed for the audit trigger).
GATE_DATABASE_URL ?= postgresql://progenitor:progenitor@localhost:5432/progenitor
INTEGRATION := DATABASE_URL=$(GATE_DATABASE_URL) uv run pytest tests/integration

export FRONTEND_DIR


.PHONY: db-dump db-restore gate-status gate-report stack-up integration-tests data-tests profiler-exact-test
.PHONY: help env up down logs ps wait test test-backend test-frontend lint gate eval check-frontend-dir \
        up-tracing migrate data live eval-phase3 eval-phase4 gate-0 gate-0-backend gate-0-frontend gate-0-stack gate-1 gate-2 gate-3 gate-4

help: ## List targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || { cp .env.example .env && echo "created .env from .env.example"; }

check-frontend-dir:
	@test -f $(FRONTEND_DIR)/package.json || { echo "frontend repo not found at $(FRONTEND_DIR); clone it there or set FRONTEND_DIR"; exit 2; }

up: env check-frontend-dir ## Start the full local stack and wait until healthy
	$(COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory wait

up-tracing: env check-frontend-dir ## Start the stack plus Langfuse (tracing profile), whatever TRACING_MODE says
	docker compose $(COMPOSE_FILES) --profile tracing up -d --build --wait
	@$(MAKE) --no-print-directory wait

down: ## Stop the stack (keeps volumes)
	docker compose $(COMPOSE_FILES) --profile tracing down

logs: ## Tail stack logs
	$(COMPOSE) logs -f --tail=100

ps: ## Show service status
	$(COMPOSE) ps

wait: ## Wait for HAPI FHIR (no container healthcheck: distroless image)
	@echo -n "waiting for hapi-fhir "; \
	for i in $$(seq 1 90); do \
	  if curl -fsS -o /dev/null http://localhost:$(HAPI_HOST_PORT)/fhir/metadata; then echo " ok"; exit 0; fi; \
	  echo -n "."; sleep 2; \
	done; echo " timed out"; exit 1

lint: ## Lint and type-check backend (frontend lint runs in the frontend repo)
	uv run ruff check . && uv run ruff format --check . && DJANGO_DEBUG=1 uv run mypy .

test: test-backend test-frontend ## Backend unit + integration, frontend unit

test-backend:
	uv run pytest tests/unit
	$(INTEGRATION) -m "integration or data"

test-frontend: check-frontend-dir
	@if grep -q '"test"' $(FRONTEND_DIR)/package.json; then $(FRONTEND) npm test; else echo "frontend: no unit tests yet (Vitest arrives in Phase 7a)"; fi

migrate: ## Apply migrations to the dev database
	uv run python manage.py migrate --noinput

data: migrate ## Generate + load dev data (idempotent): Synthea, MIMIC demo, eval sets, inventory
	uv run python manage.py generate_synthea
	uv run python manage.py load_synthea
	uv run python manage.py load_mimic
	uv run python manage.py fetch_eval_data
	uv run python manage.py data_inventory

live: ## Live smoke tests against real external APIs (not part of any gate)
	MCP_CACHE_DISABLED=1 uv run pytest tests/live -m live -v

eval: eval-phase3 ## Run the eval harness (live LLM; not in CI)

eval-phase4: ## Phase 4 live scenarios: supervisor decisions, 3 runs each, majority counts
	uv run python manage.py eval_phase4

eval-phase3: ## Phase 3 live eval: Parser F1 (Chia), Judge accuracy (hand labels), citation validity
	uv run python manage.py eval_phase3

gate: ## Run the gate suite with progress: make gate PHASE=<n>
	@test -n "$(PHASE)" || { echo "usage: make gate PHASE=<n>"; exit 2; }
	@uv run python scripts/gate.py $(PHASE)

gate-status: ## Progress of the running (or last) gate; use from a second terminal
	@uv run python scripts/gate.py status

gate-report: ## Last gate summary as markdown (paste it when reporting a result)
	@uv run python scripts/gate.py report

# ---------------- Data transfer (EC2 staging: restore instead of regenerating) ----------------
DUMP_DIR := data/external/dumps
DUMP_DBS := hapi progenitor
PG_USER := $(or $(shell grep -s '^POSTGRES_USER=' .env | tail -1 | cut -d= -f2),progenitor)
PG := $(COMPOSE) exec -T postgres

db-dump: ## Dump the hapi + progenitor databases to data/external/dumps/ (custom format, compressed)
	@mkdir -p $(DUMP_DIR)
	@for db in $(DUMP_DBS); do \
	  echo "dumping $$db"; \
	  $(PG) pg_dump -U $(PG_USER) -Fc -Z 6 $$db > $(DUMP_DIR)/$$db.dump || exit 1; \
	done
	@ls -lh $(DUMP_DIR)

db-restore: ## Restore data/external/dumps/*.dump into the running stack (replaces hapi + progenitor data)
	@for db in $(DUMP_DBS); do test -s $(DUMP_DIR)/$$db.dump || { echo "missing $(DUMP_DIR)/$$db.dump"; exit 1; }; done
	$(COMPOSE) stop hapi-fhir backend worker
	@for db in $(DUMP_DBS); do \
	  echo "restoring $$db"; \
	  $(PG) pg_restore -U $(PG_USER) -d $$db --clean --if-exists --no-owner < $(DUMP_DIR)/$$db.dump || exit 1; \
	done
	$(COMPOSE) up -d --wait hapi-fhir backend worker
	@$(MAKE) --no-print-directory wait
	uv run python manage.py fetch_eval_data
	uv run python manage.py data_inventory

# Atomic targets used by scripts/gate.py (they carry the compose file/profile settings).
stack-up: env check-frontend-dir
	$(COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory wait

integration-tests:
	$(INTEGRATION) -m integration

data-tests:
	$(INTEGRATION) -m data

profiler-exact-test:
	$(INTEGRATION)/test_profiler_exact.py -m data

# ---------------- Gate 0: Foundations ----------------
gate-0: gate-0-backend gate-0-frontend gate-0-stack
	@echo "GATE 0 PASSED"

gate-0-backend:
	uv sync --frozen
	uv run ruff check . && uv run ruff format --check .
	DJANGO_DEBUG=1 uv run mypy .
	DJANGO_DEBUG=1 uv run python manage.py check
	@# Dev-safe deploy check: production flags on, throwaway secret, in-memory db.
	DJANGO_DEBUG=0 DJANGO_SECRET_KEY="$$(python3 -c 'import secrets; print(secrets.token_urlsafe(64))')" \
	  DATABASE_URL=sqlite:///:memory: uv run python manage.py check --deploy --fail-level WARNING
	uv run bandit -q -c pyproject.toml -r .
	uv run pytest tests/unit -k smoke

# Same checks the frontend repo's own CI runs; repeated here so the gate covers both repos.
gate-0-frontend: check-frontend-dir
	$(FRONTEND) npm ci --no-audit --no-fund
	$(FRONTEND) npm run lint && npm run typecheck && npm run build

gate-0-stack: env check-frontend-dir
	$(COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory wait
	$(INTEGRATION) -m integration

# ---------------- Gate 1: Data foundation ----------------
# Precondition: `make data` has run. The gate does not reload first: its idempotency
# test already re-runs both loaders in full and asserts 0 resources created.
gate-1: gate-0
	uv run python manage.py migrate --noinput
	uv run python manage.py data_inventory
	$(INTEGRATION) -m data
	@echo "GATE 1 PASSED"

# ---------------- Gate 2: MCP tool layer ----------------
# Gate 0 (incl. MCP stack integration tests, which need the MCP containers),
# then every unit test with cassettes replayed only: no live calls.
gate-2: gate-0
	uv run pytest tests/unit --record-mode=none
	@echo "GATE 2 PASSED"

# ---------------- Gate 3: worker agents ----------------
# Gate 2, then the Profiler exact-match test (needs `make data`), then the live LLM eval
# against the thresholds in data/eval_thresholds.json (set after baseline, ADR 0003).
# Needs GROQ_API_KEY and reviewed hand labels (data/handlabel/handlabel_v1.json status "reviewed").
gate-3: gate-2
	$(INTEGRATION)/test_profiler_exact.py -m data
	uv run python manage.py eval_phase3
	@echo "GATE 3 PASSED"

# ---------------- Gate 4: LLM supervisor ----------------
# Gate 3 (incl. graph mechanics + guard unit tests and the resumability integration test),
# then the live scenario suite: pass rate >= 0.80 and no PHI in decision args.
gate-4: gate-3
	uv run python manage.py eval_phase4
	@echo "GATE 4 PASSED"
