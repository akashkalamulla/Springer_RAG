# Springer_RAG

A retrieval-augmented generation pipeline over biomedical literature — built end to
end, **served over an API**, and **evaluated with real metrics** at every phase. It
scrapes open-access articles from a Springer Nature journal, indexes them in
Postgres/pgvector, answers questions grounded strictly in the retrieved text (with
DOI citations), and scores itself on retrieval accuracy, answer faithfulness, and
refusal behavior.

**Status:** v2 — evaluated, served, corpus-grown. The core vertical slice is built
and measured end to end, including a FastAPI service in front of it.

---

## What it does

Ask a question; get an answer grounded **only** in the source articles, cited by DOI —
and an explicit refusal when the answer isn't in the corpus rather than a fabricated one.

```
$ curl -s -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
    -d "{\"question\": \"what reduces heat stress in hospital workers?\"}"

"...several individual measures are associated with lower perceived occupational heat
stress...including the use of cooling towels, reducing physical activity, and
discussing heat-related challenges with colleagues...[10.1186/s12889-026-28452-4]"
```

The pipeline is four stages, plus a service layer in front of them:

1. **Scrape** — pull the current issue of *BMC Public Health* into structured JSON:
   metadata, abstract, section-structured body, references (with resolved DOIs/PMIDs), and
   PDFs.
2. **Index** — chunk the article text (~200-token, embedding-window-aware), embed with a
   sentence-transformer, and store the vectors in Postgres + pgvector behind an HNSW cosine
   index.
3. **Retrieve + generate** — vector top-k (optionally fused with BM25 via RRF) over the
   chunks, then an LLM answers using only those passages, citing the DOI of every passage
   it relies on.
4. **Evaluate** — score retrieval (hit@k, MRR), answer faithfulness, and refusal behavior
   against hand-authored golden sets, using a *separate* judge model to avoid self-grading.
5. **Serve** — a FastAPI service (`/ask`, `/health`) exposes the same `rag.answer()`
   pipeline the eval measures, so the API inherits the measured behavior rather than
   reimplementing it.

---

## Results — the real story, three phases of measurement

### v1 baseline: saturated at 1.0

Measured against a 30-question golden set — one hand-authored, verified question per
article, all in-corpus, on a 30-article corpus:

| Metric | Score |
|---|---|
| Retrieval hit@5 | **1.0** |
| Retrieval MRR | **1.0** |
| Answer faithfulness (mean) | **1.0** |

Each question names its source study distinctively (e.g. *"the German tertiary care
hospital study,"* *"1,047 adults in Istanbul"*), and against only 30 candidate articles
there were too few plausible distractors for retrieval to ever miss. The metrics
establish that *when the correct context is retrieved, the generator grounds its
answers and refuses rather than fabricating* — they say nothing about robustness to
harder queries or a smaller/bigger corpus, because the set couldn't move off 1.0 either
way.

### Eval hardening (Phase A): a set that can actually fail

Added two more golden sets: a **HARD** set (the same 30 target articles, paraphrased
into vaguer, less name-dropping questions) and a **REFUSE** set (10 out-of-corpus
questions about real studies the corpus doesn't contain, which the system should
decline to answer).

**Honest caveat:** the HARD and REFUSE sets were AI-drafted, then human-verified as
answerable/unanswerable against the corpus — they were not independently written by a
second human. And REFUSE currently tests *obvious* off-topic questions (different
diseases, countries, studies entirely outside *BMC Public Health*), not adversarial
near-misses that sound like they could be in-corpus. Refusal held at **1.0 (10/10)**
across every run in this repo's history — a real but narrow result.

### Hybrid retrieval (Phase B / D): a measured null

Built BM25 (Postgres full-text) + vector RRF fusion (`retrieve_hybrid` in
[rag.py](rag.py)) and measured it against the same golden sets. Result: **identical**
retrieval metrics to vector-only, on both the 30-article and grown ~100-article corpus,
on both EASY and HARD sets.

The mechanism, verified directly rather than assumed: this repo's BM25 query
(`plainto_tsquery`, which **AND**s every content word together) returns **zero rows**
for all 30/30 EASY and all 30/30 HARD golden questions — because a full natural-language
question carries more content words than any single ~200-token chunk contains verbatim.
Short keyword queries (`"heat stress hospital workers"`, `"influenza"`) return rows fine;
full sentences don't. So on every query this repo's eval sets actually ask, BM25
contributes nothing to fuse, and RRF degenerates to the pure vector ranking — hybrid
isn't underperforming, it's structurally idle for this query style.

**What this means:** hybrid's real test isn't a bigger corpus (that's a retrieval
*difficulty* lever, not a lexical-matching one) — it's a golden set of genuinely
lexically-anchored queries (rare identifiers, acronyms, exact numbers) where BM25 could
plausibly out-rank a paraphrase-blurred vector embedding. That set doesn't exist yet;
see Roadmap.

### Corpus growth (Phase D): the eval finally discriminates

Grew the corpus 30 → **100 articles** (same 30 golden-question needles, more
distractors) via `build_index.py`. For the first time, HARD retrieval moved off its
near-ceiling:

| Metric | 30 articles | ~100 articles |
|---|---|---|
| HARD hit@1 | 0.967 | **0.9** |
| HARD hit@3 | 1.0 | **0.967** |
| HARD hit@5 | 1.0 | 1.0 (held) |
| HARD MRR | 0.983 | **0.94** |
| EASY / faithfulness / refusal | 1.0 / 1.0 / 1.0 | unchanged |

hit@5 held at 1.0 — nothing fell out of the corpus entirely — but three questions
dropped in rank. Pulled each one's retrieved chunks and read the question against both
the target and the distractor to tell a real miss from new, defensible ambiguity:

- **Rank-5 respiratory/hospitalization question** (target cosine 0.487, crowded by three
  other COVID-themed articles at cosine 0.50–0.53): the target article's abstract states
  the answer directly (a dose–response relationship between neighborhood deprivation
  index and hospitalization odds) — the chunk is present and well-formed. This is
  **topical crowd-out** from the larger corpus, not a chunking coverage gap.
- **mHealth/vaccination pair**: target is a pediatric-influenza digital-recall study;
  the distractor is an unrelated Ethiopia mHealth childhood-immunization trial that now
  outranks it. The question says *"seasonal vaccine,"* which maps unambiguously to the
  influenza-specific target — a careful reader would not have picked the distractor.
  **Real miss**, not ambiguity.
- **Food-environment pair**: target is a Nigeria urban-slum food-decision study; the
  distractor is a Chile pandemic-lockdown diet study (itself the correct answer to a
  *different* HARD question). The question says *"low-income urban neighborhoods,"* the
  distractor's defining feature is *"pandemic lockdown"* — again distinguishable by a
  human. **Real miss**: two topically similar articles the embedding model conflates on
  shared vocabulary ("family," "food," "children") that the query doesn't disambiguate at
  the chunk-similarity level.

All three are retrieval crowding effects from a bigger, more realistic distractor pool —
exactly what a 30-article corpus was too small to ever surface.

---

## Corpus size

**~100 articles.** Query `GET /health` for the live, authoritative chunk count — it's
not hardcoded here because it changes every time the corpus grows.

---

## Architecture (what actually runs)

```
  Springer Nature          RAG_Scraper.py            build_index.py                 rag.py + llm.py
  (BMC Public Health) ──▶  requests + BS4     ──▶   chunk (~200 tok) ─▶ embed  ──▶  retrieve top-k (vector,
                           structured JSON          (MiniLM 384-d)                   optionally RRF w/ BM25)
                           + PDFs                    store in pgvector               ─▶ generate (grounded)
                                                     HNSW cosine index                ─▶ cite DOIs / refuse
                                                             │                                │
                                                             │                                ▼
                                                             │                         app.py (FastAPI)
                                                             │                         /ask · /health
                                                             ▼
                                                          eval.py  ── hit@k · MRR · faithfulness · refusal
                                                                       (separate judge model)
```

`settings.py` centralizes every config value (`DB_DSN`, `EMBED_MODEL`, `GEN_MODEL`,
`JUDGE_MODEL`, `TOP_K`, `RRF_K`) sourced from `.env`, so nothing is hardcoded across
`build_index.py` / `rag.py` / `app.py`.

**Stack**

| Layer | Choice |
|---|---|
| Scraper | Python, `requests` + BeautifulSoup, `pymupdf4llm` |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim, 256-token window) |
| Vector store | Postgres 16 + `pgvector` (HNSW, cosine), via Docker |
| Retrieval | vector top-k, or BM25+vector RRF fusion behind a `--hybrid` / `hybrid` flag |
| Generation | OpenAI `gpt-4o-mini` |
| Faithfulness / refusal judge | OpenAI `gpt-4o` (deliberately different from the generator) |
| Service | FastAPI + uvicorn (`/ask`, `/health`), containerized (Docker Compose) |
| Corpus | ~100 articles · see `/health` for the live chunk count |

**Aspirational (design reference only, not built)** — roughly the remaining ~15% of a
fuller target architecture: a citation graph over `cited_refs` in Neo4j/GraphRAG, a
cloud-native (AWS) orchestration path, and an MCP server exposing `search_articles` /
`get_citations`. These describe a destination, not the current build — see Roadmap.

---

## Design decisions worth noting

- **Window-aware chunking.** `all-MiniLM-L6-v2` silently truncates input past 256 tokens, so
  chunks target ~200 — a 500-token chunk would be half-ignored in its vector.
- **Subsection recursion.** In this corpus most body text lives in `subsections`, not
  top-level paragraphs. A naive "read `section.paragraphs`" walk drops most of the article;
  the chunker recurses into subsections and folds the heading into the section title.
- **Abstract indexing, driven by eval.** Headline facts (sample sizes, country/setting) often
  appear only in the abstract. An early faithfulness run flagged such facts as ungrounded
  because the body-only index had dropped them; indexing abstracts as extra chunks fixed it.
- **Separate judge model.** Faithfulness and refusal are graded by `gpt-4o` while answers are
  generated by `gpt-4o-mini` — a model grading its own output inflates the score.
- **Faithful-by-default prompt.** The generator is instructed to answer *only* from the
  provided passages and to say so plainly when the answer isn't there.
- **Hybrid is measured, not assumed.** RRF fusion exists behind a flag and is proven a null
  on the current query style (see Results) rather than left unmeasured or quietly dropped.
- **pgvector correctness details.** The `vector` extension is created before
  `register_vector()`, query vectors are cast `::vector` for operator resolution, and the
  HNSW op-class (`vector_cosine_ops`) matches the query operator (`<=>`) so the index is
  actually used.

---

## Quickstart

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate                      # Windows
pip install -r requirements.txt

# 2. secrets — create a .env file (git-ignored; verify with `git check-ignore .env`)
echo OPENAI_API_KEY=sk-your-key-here > .env

# 3. scrape the current issue
python RAG_Scraper.py

# 4. database
docker compose up -d
docker compose exec postgres pg_isready -U raguser -d ragdb   # wait for "accepting connections"

# 5. build the index (point at the JSON the scraper wrote)
python build_index.py "C:\RAG\<run>\BMC_Public_Health\BMC_Public_Health_<ts>.json"

# 6. query directly, or run the API
python rag.py "what reduces heat stress in hospital workers?"
docker compose up -d --build api      # or: uvicorn app:app --reload
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
     -d "{\"question\": \"what reduces heat stress in hospital workers?\"}"

# 7. evaluate
python eval.py                             # retrieval + faithfulness + refusal
python eval.py --retrieval-only            # retrieval only (no LLM; use during an API outage)
python eval.py --hybrid                    # BM25+vector RRF instead of vector-only
```

---

## Repository layout

| Path | Role |
|---|---|
| `RAG_Scraper.py` | Springer Nature scraper → structured JSON + PDFs |
| `build_index.py` | chunk → embed → store into Postgres/pgvector |
| `rag.py` | retrieve (vector / BM25 / RRF hybrid) + generate a grounded, cited answer |
| `llm.py` | thin LLM wrapper (`generate(prompt, system=..., model=...)`) — OpenAI |
| `app.py` | FastAPI service — `/ask`, `/health` — thin skin over `rag.py` |
| `settings.py` | central config, sourced from `.env` |
| `eval.py` | golden-set evaluation: hit@k, MRR, faithfulness, refusal |
| `golden_set.jsonl` / `golden_hard.jsonl` / `golden_refuse.jsonl` | EASY / HARD (paraphrased) / REFUSE (out-of-corpus) question sets |
| `docker-compose.yml`, `Dockerfile` | Postgres 16 + pgvector, and the API container |
| `config.ini`, `urlDetails.txt` | scraper output path and journal worklist |
| `docs/` | phase runbooks and design docs |
| `results/` | raw eval run logs, kept for before/after comparison |

Python modules, runtime config, and golden sets stay at the repository root — they
import each other by flat module name and open the golden sets by bare filename, so
moving them would mean rewriting those paths for no behavioral benefit.

---

## Roadmap / deferred

- **Lexically-hard eval set** — the real test hybrid retrieval hasn't had yet: queries
  with rare identifiers, acronyms, or exact figures where BM25 could plausibly beat a
  paraphrase-blurred vector embedding. Also worth revisiting: swapping `plainto_tsquery`
  for something that doesn't AND every content word in a full sentence.
- **Cross-encoder reranking** over the retrieved pool.
- **UI** — a minimal front end over `/ask`.
- **Auth / streaming** on the API.
- **v3 items (design reference only):** Neo4j/GraphRAG citation graph, AWS-native
  orchestration, an MCP server (`search_articles`, `get_citations`), a self-healing
  extractor for scraper breakage.

---

## License / data

Corpus is open-access content from *BMC Public Health* (Springer Nature). This repo stores
derived text and vectors for research/portfolio purposes; source articles remain under their
original licenses.
