# Springer RAG

A retrieval-augmented generation pipeline over Springer open-access journal
articles: scrape → chunk → embed → store → retrieve → generate grounded,
DOI-cited answers.

## Architecture

Medallion-style pipeline:

- **Bronze** — raw scraped article JSON (`RAG_Scraper.py`), one file per
  journal, straight off the page with no cleaning.
- **Silver** — chunked, embedded records (`build_index.py`): article body
  split into ~200-token windows (sized under the embedding model's 256-token
  truncation limit), each chunk vectorized with `all-MiniLM-L6-v2`.
- **Gold** — `chunks` / `articles` tables in Postgres + pgvector, queryable by
  cosine similarity, joined back to DOI and citation metadata (`rag.py`).

## Pipeline stages

| Stage | Script | Does |
|-------|--------|------|
| M0 corpus | `RAG_Scraper.py` | Scrapes Springer journal pages into JSON |
| M1 chunk | `build_index.py` | Splits article bodies into token-bounded chunks |
| M2 embed | `build_index.py` | Embeds each chunk with `all-MiniLM-L6-v2` |
| M3 store | `build_index.py` | Writes chunks + vectors into Postgres/pgvector |
| M4 retrieve | `rag.py` | Cosine top-k search over `chunks`, joined to `articles` |
| M5 generate | `rag.py` + `llm.py` | Feeds retrieved chunks to Gemini, constrained to answer only from context, with DOI citations |
| M6 eval | *(not yet built)* | Retrieval hit@k / recall + answer faithfulness against a golden set |

## How to run

```bash
python RAG_Scraper.py                 # scrape journal(s) into JSON
python build_index.py <journal.json>  # chunk, embed, load into Postgres
python rag.py "<question>"            # retrieve + generate a grounded answer
```

Requires a running Postgres + pgvector instance (`docker-compose.yml`) and a
`.env` with a valid `GEMINI_API_KEY`.

## Evaluation

TODO — blocked on `golden_set.jsonl`: 15–20 hand-authored questions across
diverse DOIs, each verified against `rag.py`. The eval harness (`eval.py`),
measuring retrieval hit@k / recall and answer faithfulness, is the actual
differentiator for this project but cannot be built against placeholder
questions — see `M5_generate_setup.md` for the gate.
