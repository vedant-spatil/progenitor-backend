-- Runs once, on first start of an empty postgres volume.
-- One server, separate databases: app (with pgvector), HAPI FHIR, Langfuse.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE DATABASE hapi;
CREATE DATABASE langfuse;
