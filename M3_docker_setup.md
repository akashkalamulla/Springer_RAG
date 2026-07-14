# Task: M3 — Stand up Postgres + pgvector and load the index

**Scope: operations only.** Start the database, install the DB drivers, run the
existing `build_index.py`, and verify the load. Do not modify any Python logic,
the schema, chunking, embedding, or the scraper.

**Definition of done:** the `chunks` table holds **1331** rows and the console
prints `stored in Postgres: 30 articles, 1331 chunks`.

---

## Preconditions — check ALL of these first. If any fail, STOP and report back.

1. **Docker is running.** Run `docker --version` and `docker ps`.
   - If Docker is not running, you cannot fix this yourself — tell me to launch
     Docker Desktop, then stop.
2. **Working directory** is the project root `SPRINGER_RAG`, and these files exist:
   `docker-compose.yml`, `build_index.py`, `RAG_Scraper.py`.
3. **build_index.py has the M3 step.** It must contain both `def store_chunks`
   and `import psycopg`. If either is missing, STOP — the store code isn't applied.
4. **Source JSON exists.** Use this path (update only if a newer scraper run exists —
   in that case use the newest `BMC_Public_Health_*.json` under `C:\RAG`):
   ```
   C:\RAG\20260708_205404\BMC_Public_Health\BMC_Public_Health_20260708205408.json
   ```
5. **Port 5432 is free.** If a local Postgres is already bound to 5432, the
   container will fail to start. Check; if it's taken, report back — do not change
   ports silently.
6. **The venv is active** (the same one that ran M1/M2, where `sentence-transformers`
   is installed).

---

## Steps

1. **Start the database:**
   ```
   docker compose up -d
   ```
   First run pulls the `pgvector/pgvector:pg16` image — allow a minute.

2. **Wait until Postgres accepts connections** before doing anything else:
   ```
   docker compose exec postgres pg_isready -U raguser -d ragdb
   ```
   Repeat until it prints `accepting connections`. Running `build_index.py` before
   this point produces `connection refused` — do not skip the wait.

3. **Install the DB drivers into the venv:**
   ```
   pip install "psycopg[binary]" pgvector
   ```

4. **Run the pipeline** (chunk → embed → store, single command):
   ```
   python build_index.py "C:\RAG\20260708_205404\BMC_Public_Health\BMC_Public_Health_20260708205408.json"
   ```
   The embed step takes ~75–100s on CPU. Expected final line:
   ```
   stored in Postgres: 30 articles, 1331 chunks
   ```

---

## Verify (both must pass)

- Console shows `stored in Postgres: 30 articles, 1331 chunks`.
- Database confirms it independently:
  ```
  docker compose exec postgres psql -U raguser -d ragdb -c "SELECT count(*) FROM chunks;"
  ```
  Must print **1331**.

Report both results back.

---

## If it breaks

- **`connection refused` / `could not connect`** → DB wasn't ready. Return to
  step 2, wait for `pg_isready`, retry step 4.
- **`port is already allocated` (5432)** → a local Postgres owns the port. Report
  back; do not silently remap ports.
- **`relation "chunks" does not exist`** on the count → `build_index.py` never
  reached `store_chunks`. Look for an earlier traceback in the console and paste it.
- **Model re-downloads** → harmless; it's cached after the first run.
- **`vector` type errors** → the extension line (`CREATE EXTENSION IF NOT EXISTS
  vector`) didn't run; confirm the image is `pgvector/pgvector:pg16`, not plain postgres.

---

## Do NOT

- Do not edit `build_index.py`, the SQL schema, or the chunk/embed logic.
- Do not change the vector dimension (384) or `DB_DSN`.
- Do not add journals, change `SMOKE_LIMIT`, or touch `RAG_Scraper.py`.
- Do not "improve" chunk size or retrieval. That comes after the eval harness exists.

This task is done when the count is 1331. Nothing more.
