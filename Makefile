SHELL := /bin/bash
.DEFAULT_GOAL := help

# This Makefile lives in the backend repo. The frontend is a separate repo,
# checked out as a sibling folder (override with FRONTEND_DIR=...).
FRONTEND_DIR ?= ../frontend
COMPOSE := docker compose
FRONTEND := cd $(FRONTEND_DIR) &&
PHASE ?=
# Host port for HAPI FHIR; override in .env when 8080 is taken.
HAPI_HOST_PORT ?= $(shell grep -s '^HAPI_HOST_PORT=' .env | tail -1 | cut -d= -f2)
HAPI_HOST_PORT := $(or $(HAPI_HOST_PORT),8080)

# Integration tests run Django against the compose Postgres (needed for the audit trigger).
GATE_DATABASE_URL ?= postgresql://progenitor:progenitor@localhost:5432/progenitor
INTEGRATION := DATABASE_URL=$(GATE_DATABASE_URL) uv run pytest tests/integration

export FRONTEND_DIR

.PHONY: help env up down logs ps wait test test-backend test-frontend lint gate eval check-frontend-dir \
        migrate data gate-0 gate-0-backend gate-0-frontend gate-0-stack gate-1

help: ## List targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-16s %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing
	@test -f .env || { cp .env.example .env && echo "created .env from .env.example"; }

check-frontend-dir:
	@test -f $(FRONTEND_DIR)/package.json || { echo "frontend repo not found at $(FRONTEND_DIR); clone it there or set FRONTEND_DIR"; exit 2; }

up: env check-frontend-dir ## Start the full local stack and wait until healthy
	$(COMPOSE) up -d --build --wait
	@$(MAKE) --no-print-directory wait

down: ## Stop the stack (keeps volumes)
	$(COMPOSE) down

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

eval: ## Run the eval harness (live LLM; Phase 6)
	@echo "eval harness arrives in Phase 6"; exit 1

gate: ## Run the gate suite: make gate PHASE=<n>
	@test -n "$(PHASE)" || { echo "usage: make gate PHASE=<n>"; exit 2; }
	@$(MAKE) --no-print-directory gate-$(PHASE)

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
