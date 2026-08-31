#!/bin/bash
# Instantiates PostGIS (and UUID helpers) inside the local PostgreSQL cluster.
# Mounted at /docker-entrypoint-initdb.d so it runs once on first volume init.
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE EXTENSION IF NOT EXISTS postgis;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS pgcrypto;

    SELECT PostGIS_Full_Version();
    SELECT extname, extversion
      FROM pg_extension
     WHERE extname IN ('postgis', 'uuid-ossp', 'pgcrypto')
     ORDER BY extname;
EOSQL

echo "[satquery] PostGIS extensions created and verified on database ${POSTGRES_DB}"
