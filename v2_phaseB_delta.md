# v2 Phase B Delta — vector-only vs hybrid (RRF) retrieval

Measured, not claimed: both runs executed against the identical golden sets,
identical corpus (1422 chunks, same 30 DOIs), identical index — only the
retriever differs. Raw output: `run_vector.txt` / `run_hybrid.txt`. Params:
`RRF_K=60`, `pool=20`, both left at their honest defaults (not tuned).

| Metric              | Vector-only | Hybrid (RRF) | Δ    |
|---------------------|-------------|--------------|------|
| EASY  hit@1         | 1.0         | 1.0          | 0    |
| EASY  mrr           | 1.0         | 1.0          | 0    |
| HARD  hit@1         | 0.967       | 0.967        | 0    |
| HARD  hit@3         | 1.0         | 1.0          | 0    |
| HARD  mrr           | 0.983       | 0.983        | 0    |
| FAITHFULNESS (easy) | 1.0         | 1.0          | 0    |
| REFUSAL             | 1.0         | 1.0          | 0    |

## Per-row check (HARD)

Aggregate HARD hit@1 is identical in both runs (0.967 = 29/30), which could
mean either "hybrid didn't touch the same row" or "hybrid fixed one row and
broke another, netting to the same number." Checked directly: in both the
vector-only and hybrid runs, the **same single question** — HARD Q8, *"How do
families in low-income urban neighborhoods make decisions about food?"* —
lands at rank 2, not rank 1. Every other HARD row is rank 1 in both runs. This
is the exact row flagged in the Phase A baseline caveat as under-specified
(it fits two food-decision studies in the corpus), so this isn't a new
retriever weakness — it's the same known ambiguity, unmoved.

## Honest read

- **EASY had no room to move** (already 1.0 in the baseline) — flat is
  correct here, not a failure of hybrid.
- **HARD is flat too, row-for-row**, not just in aggregate. RRF fusion over a
  wide pool (20) did not change which document ranks first for any of the 30
  paraphrased questions, including the one ambiguous case. On this 30-article
  corpus, MiniLM's semantic top-1 already agrees with BM25's lexical top
  matches closely enough that fusing the two rankings doesn't reorder
  anything.
- **Faithfulness and refusal are unchanged**, which is expected — they're a
  downstream consequence of which chunks got retrieved, and the retrieved set
  is effectively the same.

## Conclusion

**Hybrid (BM25 + RRF) does not beat vector-only on this corpus, at default
parameters.** This is a real, reportable null result, not a bug: `pool` and
`RRF_K` were left at their standard defaults per the task instructions (no
tuning to manufacture a positive delta). `retrieve` stays the default
retriever in `answer()`; `retrieve_hybrid()` and `retrieve_bm25()` are kept as
tested, working alternatives, wired behind `eval.py --hybrid`, but hybrid is
not adopted as the default retrieval path.

The likely reason: at 30 articles / 1422 chunks, the corpus is small enough
and the golden questions specific enough that semantic top-5 already contains
the right answer almost every time (this is the same reason the task doc
gives for deferring cross-encoder reranking to v2.5) — there's little room
for a second signal to change the ranking. Hybrid retrieval may earn its keep
once Phase D grows the corpus and lexical/semantic rankings start to diverge
more; on the current corpus it's a validated no-op, which is itself useful to
know before investing further in retrieval tuning.
