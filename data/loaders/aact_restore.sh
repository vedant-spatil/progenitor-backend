#!/usr/bin/env bash
# Optional: restore an AACT snapshot (offline copy of ClinicalTrials.gov) into the
# compose Postgres as database "aact". Not part of any gate.
#
# 1. Download a "Flat files / pg_dump" snapshot zip from
#    https://aact.ctti-clinicaltrials.org/downloads (free account) into
#    data/external/aact/  (gitignored).
# 2. From backend/:  bash data/loaders/aact_restore.sh data/external/aact/<snapshot>.zip
#
# Restore takes about 5-15 minutes and needs several GB of disk.
set -euo pipefail

ZIP="${1:?usage: aact_restore.sh <aact-snapshot.zip>}"
PGUSER="${POSTGRES_USER:-progenitor}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

unzip -q "$ZIP" -d "$WORK"
DUMP="$(find "$WORK" -name '*.dmp' | head -n1)"
[ -n "$DUMP" ] || { echo "no .dmp file in $ZIP"; exit 1; }

compose() { docker compose exec -T postgres "$@"; }

# The dump expects role "ctti" to exist.
compose psql -U "$PGUSER" -d postgres -v ON_ERROR_STOP=1 -c \
  "DO \$\$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ctti') THEN CREATE ROLE ctti; END IF; END \$\$;"
compose psql -U "$PGUSER" -d postgres -c "CREATE DATABASE aact" 2>/dev/null || echo "database aact exists"

docker compose cp "$DUMP" postgres:/tmp/aact.dmp
compose pg_restore -U "$PGUSER" -d aact --no-owner -c --if-exists /tmp/aact.dmp || true
compose rm -f /tmp/aact.dmp
compose psql -U "$PGUSER" -d aact -c "SELECT count(*) AS studies FROM ctgov.studies"
