# Task: Repo hygiene + M5 (generate) — wire Gemini into the RAG pipeline

**Scope: operations + one wiring step.** Get the repo commit-ready
(`.gitignore`, `.env`, `requirements.txt`, `README.md`) and wire the generation
layer (`llm.py`) into `rag.py` so a query returns a **grounded answer**, not just
retrieved chunks. Do NOT modify chunking, embedding, the schema, the scraper, or
retrieval logic. Do NOT author or modify `golden_set.jsonl` — that is a human
task (see **The gate** at the bottom).

**Definition of done:**
- `git status` shows no `.env`, no `__pycache__/`, no `.venv/` tracked.
- `requirements.txt` exists and installs clean in a fresh venv.
- `python rag.py "..."` prints a generated answer that cites DOIs, sourced ONLY
  from the retrieved chunks.

---

## Preconditions — check ALL first. If any fail, STOP and report back.

1. **Postgres is up and loaded (M3 done).** Verify:
   ```
   docker compose exec postgres psql -U raguser -d ragdb -c "SELECT count(*) FROM chunks;"
   ```
   Must print **1331**. If not, run the M3 runbook first.
2. **Retrieval works.** `python rag.py "heat stress hospital workers"` prints
   ranked hits with cosine sims. If it errors, fix M4 before touching M5.
3. **`llm.py` exists and the key works.** Smoke test:
   ```
   python -c "from llm import generate; print(generate('Reply with only the word: pong'))"
   ```
   Must print `pong`. A 401/403 means a bad key in `.env`; a 429 means rate
   limit (unlikely at this scale).
4. **venv active**, with `google-genai` and `python-dotenv` installed.
5. **`.env` holds a REAL key**, not the `AIza-REPLACE...` placeholder.

---

## Steps

1. **Place `.gitignore` and `.env` in the repo root** (files provided
   separately). If `.env` was ever committed by accident, untrack it:
   ```
   git rm --cached .env
   ```
   Confirm `git status` no longer lists `.env`.

2. **Write `requirements.txt`.** Do NOT dump `pip freeze` — it captures every
   transitive package and reads as noise in a portfolio. Hand-list the direct
   dependencies, pinned to the versions currently installed:
   ```
   requests
   beautifulsoup4
   pymupdf4llm
   sentence-transformers
   psycopg[binary]
   pgvector
   google-genai
   python-dotenv
   ```
   Pin each with `==<version>` (read versions from `pip show <pkg>`).

3. **Wire M5 (generate) into `rag.py`.** Retrieval already returns rows with
   `text`, `section_title`, `doi`, `cited_refs`. Add a generation step that
   feeds those rows to the model and constrains it to answer only from them.
   Apply these additions to `rag.py` (do not rewrite the file):

   At the top, with the other imports:
   ```python
   from llm import generate
   ```

   Near the constants (`TOP_K`, etc.):
   ```python
   PROMPT_SYSTEM = (
       "You are a research assistant. Answer the question using ONLY the "
       "provided context passages. If the context does not contain the answer, "
       "say so plainly — do not use outside knowledge. Cite the DOI of every "
       "passage you rely on, in square brackets like [10.1186/...]."
   )
   ```

   New functions, above `main()`:
   ```python
   def build_prompt(query, hits):
       blocks = []
       for h in hits:
           blocks.append(f"[DOI: {h['doi']} | {h['section_title']}]\n{h['text']}")
       context = "\n\n".join(blocks)
       return f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"

   def answer(query, k=TOP_K):
       """Retrieve, then generate a grounded answer. Returns (text, hits)."""
       hits = retrieve(query, k)
       text = generate(build_prompt(query, hits), system=PROMPT_SYSTEM)
       return text, hits
   ```

   In `main()`, after the loop that prints the hits, add:
   ```python
   ans, _ = answer(query)
   print("\n=== ANSWER ===\n")
   print(ans)
   ```

4. **Write a `README.md` skeleton.** Sections, in order: one-line what/why;
   architecture (medallion Bronze→Silver→Gold); pipeline stages (M0 corpus →
   M1 chunk → M2 embed → M3 store → M4 retrieve → M5 generate → M6 eval); how to
   run (`RAG_Scraper.py` → `build_index.py <json>` → `rag.py "<question>"`); and
   an **Evaluation** section that names the eval harness as the differentiator.
   Leave the Evaluation section stubbed with a TODO — it depends on the gate
   below. Do NOT invent eval numbers.

---

## Verify (all must pass)

- `python rag.py "which preventive measures lowered heat stress among hospital staff"`
  prints an answer that cites at least one DOI **present in the retrieved hits**.
- The answer does NOT cite a DOI absent from the retrieved set (if it does, the
  prompt isn't constraining to context — tighten `PROMPT_SYSTEM`, do NOT widen
  retrieval).
- `git status` is clean of `.env`, `.venv/`, `__pycache__/`.
- `pip install -r requirements.txt` succeeds in a fresh venv.

---

## If it breaks

- **`generate()` raises 401/403** → wrong/expired key in `.env`.
- **`generate()` raises 429** → free-tier rate limit; back off and retry, don't
  loop.
- **Empty answer / `resp.text` is None** → confirm the model string in `llm.py`
  is `gemini-3.5-flash`.
- **Answer cites a DOI not in the hits** → prompt leak; strengthen the "ONLY the
  provided context" instruction. This is a prompt bug, never a retrieval bug.

---

## Do NOT

- Do NOT author, edit, or fabricate `golden_set.jsonl`. You cannot invent
  domain-valid evaluation questions; that is the human's job and the gate to M6.
- Do NOT build `eval.py` to run against the two placeholder rows — the numbers
  would be meaningless and worse than no numbers.
- Do NOT change chunk size, embedding model, vector dimension (384), `DB_DSN`,
  or retrieval. Tuning comes AFTER the eval harness exists, never before.
- Do NOT commit `.env` or the key.

---

## The gate (read this)

M6 — `eval.py`, measuring retrieval hit@k / recall AND answer faithfulness — is
**BLOCKED** on `golden_set.jsonl` containing 15–20 hand-authored questions across
diverse DOIs, each verified against `rag.py`. This runbook deliberately stops
before M6.

No tooling substitutes for a human writing those questions. Everything above is
plumbing; the eval is the actual deliverable for the AI-Engineer target, and it
cannot start until the golden set is real. When it is, the next runbook builds
`eval.py` on top of the `answer()` function added in Step 3.
