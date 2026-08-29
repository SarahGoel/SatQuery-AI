#!/bin/bash
# Instantiates PostGIS (and UUID helpers) inside the local PostgreSQL container.
# Mounted at /docker-entrypoint-initdb.d so it runs once on first volume init.
set -euo pipefail

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS postgis_topology;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS pgcrypto;
    SELECT PostGIS_Full_Version();
EOSQL

echo "[satquery] PostGIS extensions enabled on database ${POSTGRES_DB}"
