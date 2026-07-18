# Springer_RAG

A retrieval-augmented generation pipeline over biomedical literature — built end to
end and **evaluated with real metrics**. It scrapes open-access articles from a Springer
Nature journal, indexes them in Postgres/pgvector, answers questions grounded strictly in
the retrieved text (with DOI citations), and scores itself on both retrieval accuracy and
answer faithfulness.

**Status:** v1 complete — every stage built, measured, and reproducible.

---

## What it does

Ask a question; get an answer grounded **only** in the source articles, cited by DOI —
and an explicit refusal when the answer isn't in the corpus rather than a fabricated one.

```
$ python rag.py "In which country is the Physio-HeMAB heat-exposure pregnancy trial conducted?"

The Physio-HeMAB cluster randomized trial is being conducted in Ghana, hosted by the
Health and Demographic Surveillance System in Navrongo [10.1186/s12889-026-28030-8].
```

The pipeline is four stages:

1. **Scrape** — pull the current issue of *BMC Public Health* into structured JSON:
   metadata, abstract, section-structured body, references (with resolved DOIs/PMIDs), and
   PDFs.
2. **Index** — chunk the article text (~200-token, embedding-window-aware), embed with a
   sentence-transformer, and store the vectors in Postgres + pgvector behind an HNSW cosine
   index.
3. **Retrieve + generate** — semantic top-5 over the chunks, then an LLM answers using only
   those passages, citing the DOI of every passage it relies on.
4. **Evaluate** — score retrieval (hit@k, MRR) and answer faithfulness against a
   hand-authored golden set, using a *separate* judge model to avoid self-grading.

---

## Results

Measured against a 30-question golden set — one hand-authored, verified question per
article, all in-corpus:

| Metric | Score | What it measures |
|---|---|---|
| Retrieval hit@5 | **1.0** | the correct article appears in the top-5 retrieved chunks |
| Retrieval MRR | **1.0** | ...and it's ranked #1 |
| Answer faithfulness (mean) | **1.0** | every claim in the generated answer is grounded in retrieved context |

**How to read these numbers honestly** (this matters more than the numbers):

Each golden-set question names its source study distinctively (e.g. *"the German tertiary
care hospital study,"* *"1,047 adults in Istanbul"*), which makes retrieval near-trivial for
this set — so the metrics are **saturated at 1.0**. What they establish: *when the correct
context is retrieved, the generator grounds its answers and does not fabricate, and when a
fact is absent it refuses rather than inventing.* What they do **not** yet establish:
robustness to vague or paraphrased queries, or a measured false-refusal rate. Hardening the
eval with adversarial questions — including out-of-corpus ones the system *should* refuse —
is the first item on the v2 roadmap. The harness exists precisely so that every future
change is measured, not guessed.

One such measured change already happened: an earlier faithfulness run flagged specific
facts (a study's country, a data registry) as ungrounded. Diagnosis showed those facts lived
in article **abstracts**, which the body-only index had dropped. Adding abstract chunks made
them retrievable and the flags cleared — a change made *because the eval caught it*, not on a
hunch.

---

## Architecture (what actually runs)

```
  Springer Nature          RAG_Scraper.py            build_index.py                 rag.py + llm.py
  (BMC Public Health) ──▶  requests + BS4     ──▶   chunk (~200 tok) ─▶ embed  ──▶  retrieve top-5 (cosine)
                           structured JSON          (MiniLM 384-d)                   ─▶ generate (grounded)
                           + PDFs                    store in pgvector               ─▶ cite DOIs / refuse
                                                     HNSW cosine index
                                                             │
                                                             ▼
                                                          eval.py  ── hit@k · MRR · faithfulness
                                                                       (separate judge model)
```

**Stack**

| Layer | Choice |
|---|---|
| Scraper | Python, `requests` + BeautifulSoup, `pymupdf4llm` |
| Embeddings | `sentence-transformers` `all-MiniLM-L6-v2` (384-dim, 256-token window) |
| Vector store | Postgres 16 + `pgvector` (HNSW, cosine), via Docker |
| Generation | OpenAI `gpt-4o-mini` |
| Faithfulness judge | OpenAI `gpt-4o` (deliberately different from the generator) |
| Corpus | 30 articles · ≈1,360 embedded chunks (bodies + abstracts) |

> Replace `≈1,360` with the exact count from your latest `build_index.py` run
> (`stored in Postgres: 30 articles, N chunks`).

---

## Design decisions worth noting

- **Window-aware chunking.** `all-MiniLM-L6-v2` silently truncates input past 256 tokens, so
  chunks target ~200 — a 500-token chunk would be half-ignored in its vector. The M1 build
  step prints how many chunks exceed the window; it should be zero.
- **Subsection recursion.** In this corpus ~90% of body text lives in `subsections`, not
  top-level paragraphs. A naive "read `section.paragraphs`" walk drops most of the article;
  the chunker recurses into subsections and folds the heading into the section title.
- **Abstract indexing, driven by eval.** Headline facts (sample sizes, country/setting) often
  appear only in the abstract or a table. The faithfulness eval surfaced these as ungrounded;
  indexing abstracts as extra chunks made them retrievable. Change → measure → keep.
- **Separate judge model.** Faithfulness is graded by `gpt-4o` while answers are generated by
  `gpt-4o-mini`. A model grading its own output inflates the score.
- **Faithful-by-default prompt.** The generator is instructed to answer *only* from the
  provided passages and to say so plainly when the answer isn't there — the mechanism behind
  the faithfulness result and the refusals.
- **pgvector correctness details.** The `vector` extension is created and committed *before*
  `register_vector()` (or a cold DB fails), query vectors are cast `::vector` for operator
  resolution, and the HNSW op-class (`vector_cosine_ops`) matches the query operator (`<=>`)
  so the index is actually used.

---

## Quickstart

```bash
# 1. environment
python -m venv venv
venv\Scripts\activate                      # Windows
pip install requests beautifulsoup4 pymupdf4llm sentence-transformers \
            "psycopg[binary]" pgvector openai python-dotenv

# 2. secrets — create a .env file (git-ignored)
echo OPENAI_API_KEY=sk-your-key-here > .env

# 3. scrape the current issue (capped at 30 articles)
set SMOKE_LIMIT=30                         # Windows cmd
python RAG_Scraper.py

# 4. database
docker compose up -d
docker compose exec postgres pg_isready -U raguser -d ragdb   # wait for "accepting connections"

# 5. build the index (point at the JSON the scraper wrote)
python build_index.py "C:\RAG\<run>\BMC_Public_Health\BMC_Public_Health_<ts>.json"

# 6. query
python rag.py "what reduces heat stress in hospital workers?"

# 7. evaluate
python eval.py                             # retrieval + faithfulness
python eval.py --retrieval-only            # retrieval only (no LLM; use during an API outage)
```

---

## Repository layout

| File | Role |
|---|---|
| `RAG_Scraper.py` | Springer Nature scraper → structured JSON + PDFs |
| `build_index.py` | chunk → embed → store into Postgres/pgvector (M1–M3) |
| `rag.py` | retrieve + generate a grounded, cited answer (M4–M5) |
| `llm.py` | thin LLM wrapper (`generate(prompt, system=...)`) — OpenAI |
| `eval.py` | golden-set evaluation: hit@k, MRR, faithfulness (M6) |
| `golden_set.jsonl` | 30 hand-authored question → answer-DOI pairs |
| `docker-compose.yml` | Postgres 16 + pgvector |
| `config.ini`, `urlDetails.txt` | scraper output path and journal worklist |

---

## Roadmap

The repository implements the core, evaluated vertical slice. The following are **not built**
and are stated as roadmap, not capability:

- **Harder eval set** — adversarial/paraphrased questions plus out-of-corpus questions the
  system should refuse, so the metrics can move off 1.0 and detect regressions.
- **Corpus breadth** — more issues/journals, so retrieval is non-trivial.
- **Retrieval upgrades** — hybrid (BM25 + vector) search and a reranking pass, each validated
  against the harness.
- **Serving** — a small API + minimal UI so the pipeline is demoable.
- **Aspirational (design reference only)** — a citation graph over the `cited_refs` edges, a
  cloud-native orchestration path, and an MCP server exposing `search_articles` /
  `get_citations`. A fuller target-architecture diagram exists as a design reference; it
  describes the destination, not the current build.

---

## License / data

Corpus is open-access content from *BMC Public Health* (Springer Nature). This repo stores
derived text and vectors for research/portfolio purposes; source articles remain under their
original licenses.
