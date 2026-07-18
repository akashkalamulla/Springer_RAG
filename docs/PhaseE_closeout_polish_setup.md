# Task: v2 Phase E — Closeout polish (restructure, gitignore/.env, docstring, README)

**Scope: file moves + docs + one docstring, NO logic changes.** Tidy the repo
into folders, prove `.env` is git-ignored, fix the stale `llm.py` docstring, and
rewrite the README to reflect what Phases A–D actually established. Do NOT change
any pipeline behavior — this is the phase where you make a working system *look*
like a working system, without touching how it works.

**Why the caution flag is higher than it looks:** moving files in this repo is
NOT drag-and-drop. `app.py`/`rag.py`/`eval.py`/`build_index.py`/`llm.py`/
`settings.py` import each other by flat module name (`import rag`,
`from settings import ...`), `eval.py` opens `golden_set.jsonl` by bare filename,
and the Dockerfile does `COPY . .`. A naive reorg silently breaks imports and
file lookups. The restructure below is deliberately **conservative** — code and
config stay at root; only logs, data, and docs move.

**Definition of done:**
- Repo organized into `docs/`, `results/`, `eval/` — with all Python modules and
  runtime config still at root so imports and file lookups are unbroken.
- `git check-ignore .env` prints `.env` (the key is provably not committable).
- `llm.py` docstring says OpenAI, not Gemini.
- README reflects the real A–D state: measured-null hybrid, discriminating eval,
  ~100-article corpus, service layer — no hardcoded chunk count.
- `eval.py`, `/health`, and `/ask` all still work **after** the moves (verified
  before the commit).

---

## Preconditions — check ALL of these first. If any fail, STOP and report back.

1. **The two open Phase D checks are resolved** (the README can't honestly quote
   Phase D numbers until they are):
   - The rank-5 respiratory-hospitalization miss (cosine 0.49–0.53): is it
     topical crowd-out, or a chunking coverage gap (answer lives in a
     dropped table/number)? Pull that article's chunks and eyeball whether the
     answer text is actually present.
   - The two near-duplicate misses (mHealth-vaccination pair, food-environment
     pair): real misses, or NEW ambiguity where a human couldn't pick the right
     DOI either (underspecified, like Q8)? Read each hard question against BOTH
     the target and the distractor and decide.
   Record the verdicts — they go in the README's honest-read section.
2. **The Phase C containerized `/ask` was confirmed** — a real grounded answer
   from the API container (not just `/health`, which doesn't exercise
   generation). If never confirmed, do it now:
   ```
   curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d "{\"question\": \"what reduces heat stress in hospital workers?\"}"
   ```
   Must return an `answer` field with DOI citations, not a 502.
3. **`.gitignore` exists** and the repo is a git repo (`git status` works).
4. **Everything currently runs** — `python eval.py --retrieval-only` succeeds,
   `/health` is ok. Establish the green baseline BEFORE moving anything, so a
   post-move break is unambiguous.

---

## Step 1 — Restructure (conservative; verify after each group)

**Rule: Python modules and runtime config stay at root.** These import each
other flatly and read files by bare name — moving them means rewriting imports
for zero benefit. Do NOT move:
`app.py rag.py llm.py eval.py build_index.py migrate_add_tsv.py settings.py
RAG_Scraper.py config.ini urlDetails.txt docker-compose.yml Dockerfile
requirements.txt .env .gitignore .dockerignore golden_set.jsonl golden_hard.jsonl
golden_refuse.jsonl`

> Golden sets stay at root because `eval.py` opens them by bare filename
> (`open("golden_set.jsonl")`). Moving them means editing every open() path.
> Not worth it — leave them. (If you insist on an `eval/` data folder, you must
> also update the `GOLDEN`/`HARD`/`REFUSE` constants in `eval.py` to point at it,
> and re-run the eval to confirm. Only do this if you're going to test it.)

**Create two folders and move only these:**

```
mkdir docs results
```

Move the phase runbooks and design docs into `docs/`:
```
move PhaseA_eval_hardening_setup.md docs\
move PhaseB_hybrid_retrieval_setup.md docs\
move PhaseC_fastapi_serving_setup.md docs\
move PhaseD_corpus_growth_setup.md docs\
move M6_eval_setup.md docs\
move M3_docker_setup.md docs\        (if present)
```

Move the run logs and result docs into `results/`:
```
move run_vector_baseline.txt results\
move run_vector.txt results\
move run_hybrid.txt results\
move run_vector_grown.txt results\
move run_hybrid_grown.txt results\
move v2_baseline.md results\
move v2_phaseB_delta.md results\
move v2_phaseD_corpus.md results\
```

Delete the leftover scratch (one-off diagnostics, already served their purpose):
```
del rank_check.py verify_hard.py hard_review.txt
```
(Any of these already gone → skip.)

**Verify nothing broke** — moves were docs/logs only, but confirm the import
graph and file lookups are still intact:
```
python -c "from rag import retrieve; from settings import DB_DSN; print('imports ok')"
python eval.py --retrieval-only
curl -s http://localhost:8000/health
```
All three must pass. If `eval.py` now errors on a missing golden file, a golden
set got moved — move it back to root.

> **Dockerfile note:** `COPY . .` copies everything including the new folders, so
> the image still builds. `.dockerignore` should exclude `docs/`, `results/`,
> `venv/`, `.git/`, `__pycache__/`, `.claude/`, and scraped data so they don't
> bloat the build context — check it covers these (Step 2 handles the git side).

## Step 2 — Harden `.gitignore` and PROVE `.env` is ignored

This is the one step in Phase E that's genuinely dangerous to get wrong. `.env`
holds `OPENAI_API_KEY`.

Ensure `.gitignore` contains at least:
```
.env
venv/
__pycache__/
*.pyc
.claude/
# scraped data / PDFs / models — never commit
*.pdf
chunks.jsonl
```

Then PROVE it:
```
git check-ignore .env
```
- Prints `.env` → safe, it's ignored. Proceed.
- Prints **nothing** → `.env` is NOT ignored. Add it, and check whether it was
  ever already committed:
  ```
  git log --all --oneline -- .env
  ```
  If that shows ANY commits, the key is in history — **the key must be rotated**
  (generate a new `OPENAI_API_KEY`, update `.env`, revoke the old one). Deleting
  the file does not scrub git history. Report this to Akash immediately; do not
  proceed to commit.

Also confirm the folders you just made aren't accidentally ignored (they should
be committed):
```
git check-ignore docs results
```
Should print **nothing** for both (they're tracked).

## Step 3 — Fix the stale `llm.py` docstring

The module docstring still describes Gemini; the code is OpenAI. Cosmetic but
it's a lie in the file. Find the top docstring:
```python
"""
llm.py — M5 (generate). Thin wrapper around the Gemini API.

Loads GEMINI_API_KEY from .env and exposes generate(prompt, system=...), used by
rag.py to turn retrieved chunks into a grounded answer.
"""
```
Replace with:
```python
"""
llm.py — M5 (generate). Thin wrapper around the OpenAI API.

Loads OPENAI_API_KEY from .env and exposes generate(prompt, system=..., model=...),
used by rag.py to turn retrieved chunks into a grounded answer. The generator and
the eval's faithfulness judge pass different model names through the same call.
"""
```
No code changes — docstring only.

## Step 4 — Rewrite the README to the true A–D state

The current README describes v1. Update it to reflect what actually exists now.
Keep the honest-metrics ethos that's already the README's strength. Sections:

- **Status:** v2 — evaluated, served, corpus-grown. Core vertical slice built and
  measured end to end.
- **What it does / pipeline:** unchanged 4 stages, plus the FastAPI service
  (`/ask`, `/health`) as the interface.
- **Results — the real story, three phases of measurement:**
  - v1 baseline: saturated 1.0s on a 30-article corpus (state plainly *why* they
    saturate — distinctive questions, few distractors).
  - Eval hardening (Phase A): added paraphrased HARD set + out-of-corpus refusal
    set. Note the honest caveat — the hard/refuse sets were AI-drafted then
    human-verified answerable; refusal is tested on obvious off-topic, not
    adversarial near-misses.
  - Hybrid retrieval (Phase B/D): built BM25+vector RRF, measured it as a
    **null** — and explain the *mechanism* (on paraphrased hard queries BM25
    returns nothing to fuse, so RRF degenerates to the vector list). Frame as
    "hybrid's real test is a lexically-hard set, not a bigger corpus." This is
    the strongest single item in the README — a measured, explained negative
    result.
  - Corpus growth (Phase D): 30 → ~100 articles around the SAME 30 golden
    needles made the eval discriminating for the first time — HARD hit@1
    0.967 → 0.9, MRR 0.983 → 0.94, hit@5 held at 1.0. Include the per-row read
    from precondition #1 (near-duplicate vs ambiguity vs coverage gap).
- **Corpus size:** say "~100 articles" and point to `GET /health` for the live
  chunk count. Do NOT hardcode 4527 — it goes stale the next time the corpus
  changes.
- **Architecture (what actually runs):** update the diagram/stack to include the
  service layer, settings module, and hybrid-behind-a-flag. Keep the
  aspirational AWS/Neo4j/GraphRAG items clearly marked as roadmap, not built
  (the ~15% figure).
- **Repository layout:** reflect the new `docs/` and `results/` folders.
- **Roadmap / deferred:** lexically-hard eval set (the real hybrid test),
  cross-encoder reranking, UI, auth/streaming, and the v3 items (Neo4j, GraphRAG,
  AWS, MCP, self-healing extractor).

---

## Verify (all must pass)

- `python -c "from rag import retrieve; from settings import DB_DSN; print('ok')"`
  → `ok` (imports unbroken by the moves).
- `python eval.py --retrieval-only` → runs, prints EASY + HARD blocks.
- `curl http://localhost:8000/health` → ok with live chunk count.
- `git check-ignore .env` → prints `.env`.
- `git status` → `docs/`, `results/`, updated README, updated `.gitignore`,
  `llm.py` show as changes; `.env`/`venv/`/`__pycache__/` do NOT appear.
- README renders and describes the real current state (no hardcoded chunk count,
  hybrid null explained).

---

## Commit (only after every verify passes)

Do it as **two commits**, so a path break is caught before it's history:
```
git add docs results *.md .gitignore
git commit -m "chore: organize docs+results, harden gitignore, fix llm docstring"

git add README.md
git commit -m "docs: rewrite README for v2 (eval hardening, hybrid null, corpus growth, service)"
```
Push. If anything in Verify failed, do NOT commit — fix first.

---

## If it breaks

- **`ModuleNotFoundError` / `ImportError` after moving** → a Python module got
  moved out of root. Move it back; only docs/logs move.
- **`FileNotFoundError: golden_set.jsonl`** → a golden set got moved; return it
  to root (or update `eval.py`'s path constants AND re-test).
- **Docker build fails after restructure** → `.dockerignore` or a `COPY` path
  references a moved file. Since only docs/logs moved and they're `COPY . .`-ed
  wholesale, this is unlikely; check `.dockerignore` didn't start excluding
  something the app needs.
- **`git check-ignore .env` prints nothing** → see Step 2; do not commit until
  it prints `.env`, and rotate the key if it was ever committed.

---

## Do NOT

- Do NOT move Python modules, config, or golden sets out of root. The import
  graph and bare-filename `open()` calls depend on the flat layout.
- Do NOT change any pipeline logic. Phase E is moves + docs + one docstring.
- Do NOT commit `.env`, `venv/`, `__pycache__/`, `.claude/`, PDFs, or scraped
  data. Verify with `git status` before committing.
- Do NOT hardcode the chunk count in the README. "~100 articles" + `/health`.
- Do NOT overclaim in the README — the hybrid result is a null, the refusal
  metric is tested on obvious off-topic only, the hard/refuse sets were
  AI-drafted. Naming the limits IS the signal; hiding them is the anti-signal.

---

## The gate (read this)

Phase E is done when the repo is clean, `.env` is **provably** un-committable
(`git check-ignore .env` prints `.env`), everything still runs after the moves,
and the README tells the true A–D story — including the measured-null hybrid and
the caveats — without a single hardcoded number that will go stale.

The one irreversible mistake available in this phase is committing the API key.
Everything else is a file move you can undo. So the real gate is Step 2: prove
the key is ignored before you type `git commit`. If it was ever committed, the
key is compromised and gets rotated — deleting the file is not enough.
