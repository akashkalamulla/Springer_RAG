# Task: M6 — Build `eval.py` (the evaluation harness)

**Scope: one new file + two small edits to `llm.py`.** Build `eval.py`, which
scores the RAG pipeline against `golden_set.jsonl`. Do NOT modify chunking,
embedding, the schema, retrieval, or the scraper. Do NOT author, edit, or
fabricate `golden_set.jsonl` — that is the human's job and the precondition for
this entire task (see **The gate**).

**Definition of done:**
- `eval.py` exists and reuses the real pipeline (`retrieve`, `answer`) — it does
  not reimplement retrieval or generation.
- `python eval.py --retrieval-only` prints **hit@k / MRR** with no LLM call.
- `python eval.py` additionally prints a **mean faithfulness** score.
- Run against a placeholder golden set, `eval.py` **refuses** and prints why — it
  does not emit numbers.

---

## Preconditions — check ALL first. If any fail, STOP and report back.

1. **The golden set is real.** `golden_set.jsonl` must contain **≥ 15
   hand-authored questions across diverse DOIs.** Check:
   ```
   python -c "import json; r=[json.loads(l) for l in open('golden_set.jsonl',encoding='utf-8')]; print(len(r),'questions,',len(set(x['answer_doi'] for x in r)),'distinct DOIs')"
   ```
   If this prints fewer than 15 questions (it currently prints `2 questions, 1
   distinct DOIs`), **STOP.** You cannot proceed and you must not write the
   questions yourself — report back that M6 is blocked on the human golden set.
2. **`rag.py` works end to end.** `retrieve()` returns ranked hits and
   `answer()` returns `(text, hits)`. Confirm M4/M5 (the M5 runbook) are done.
3. **Postgres is up** with 1331 chunks.
4. **venv active**, `google-genai` + `python-dotenv` installed.

> **Model-outage note.** The **retrieval** half of this harness makes **zero LLM
> calls** — it runs on local MiniLM embeddings + pgvector. So even while the
> generation model is throttled, once the golden set exists you can run
> `python eval.py --retrieval-only` and get half your scorecard as real numbers.
> Only the **faithfulness** half needs the model back.

---

## Steps

### 1. Extend `llm.py` (two small changes)

**a. Let the caller pick a model** (so the faithfulness judge can differ from the
generator). Change the signature and the model argument:
```python
def generate(prompt, system=None, temperature=0.2, model=None, retries=4):
    ...
        model=model or _MODEL,   # was: model=_MODEL
    ...
```

**b. Add retry/backoff on the free-tier rate limit.** Wrap the API call so a
single 429 during an eval loop doesn't kill the whole run:
```python
import time  # add to imports

def generate(prompt, system=None, temperature=0.2, model=None, retries=4):
    config = types.GenerateContentConfig(system_instruction=system, temperature=temperature)
    for attempt in range(retries):
        try:
            resp = _client.models.generate_content(
                model=model or _MODEL, contents=prompt, config=config)
            return (resp.text or "").strip()
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e).upper()
            if is_rate_limit and attempt < retries - 1:
                time.sleep(2 ** attempt)   # 1s, 2s, 4s
                continue
            raise
```

### 2. Create `eval.py`

```python
"""
eval.py — M6. Scores the RAG pipeline against golden_set.jsonl.

Two independent halves:
  RETRIEVAL    hit@k / MRR. LOCAL ONLY — MiniLM + pgvector, no LLM call.
               Runs even when the generation model is down.
  FAITHFULNESS is the generated answer grounded in the retrieved context? Uses
               the generation model to answer, then a DIFFERENT model to judge —
               a model grading its own output inflates the score.

Usage:
  python eval.py --retrieval-only     # no LLM; use during a model outage
  python eval.py                      # full: retrieval + faithfulness
"""

import argparse
import json

from rag import retrieve, answer          # reuse the real pipeline, don't fork it
from llm import generate

GOLDEN = "golden_set.jsonl"
TOP_K = 5
MIN_QUESTIONS = 15                        # refuse to score a placeholder set
JUDGE_MODEL = "gemini-2.5-pro"            # MUST differ from the Flash generator


def load_golden(path=GOLDEN):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    n_doi = len(set(r["answer_doi"] for r in rows))
    if len(rows) < MIN_QUESTIONS:
        raise SystemExit(
            f"golden_set has {len(rows)} question(s) across {n_doi} DOI(s). "
            f"Need >= {MIN_QUESTIONS} across diverse DOIs. Refusing to report "
            f"metrics on a placeholder set — the numbers would be meaningless."
        )
    return rows


def eval_retrieval(rows, k=TOP_K):
    """Local only. Did answer_doi come back in top-k, and at what rank?"""
    hits, ranks, misses = 0, [], []
    for r in rows:
        dois = [h["doi"] for h in retrieve(r["question"], k)]
        if r["answer_doi"] in dois:
            hits += 1
            ranks.append(dois.index(r["answer_doi"]) + 1)
        else:
            misses.append(r["question"])
    n = len(rows)
    return {
        "n": n, "k": k,
        "hit@k": round(hits / n, 3),
        "mrr": round(sum(1 / rk for rk in ranks) / n, 3) if ranks else 0.0,
        "misses": misses,
    }


def judge(question, answer_text, hits):
    context = "\n\n".join(f"[{h['doi']}] {h['text']}" for h in hits)
    prompt = (
        "Grade whether the ANSWER is fully supported by the CONTEXT.\n"
        'Return ONLY JSON: {"score": <0.0-1.0>, "unsupported": [<claims not in context>]}.\n'
        "1.0 = every claim grounded; subtract for each unsupported claim.\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {question}\n\nANSWER: {answer_text}\n\nJSON:"
    )
    raw = generate(prompt, model=JUDGE_MODEL, temperature=0.0)
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": None, "unsupported": ["judge did not return valid JSON"]}


def eval_faithfulness(rows):
    detail = []
    for r in rows:
        answer_text, hits = answer(r["question"])   # the REAL pipeline output
        detail.append({"question": r["question"], **judge(r["question"], answer_text, hits)})
    scored = [d["score"] for d in detail if d.get("score") is not None]
    return {
        "mean": round(sum(scored) / len(scored), 3) if scored else None,
        "graded": len(scored),
        "detail": detail,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip faithfulness (no LLM); use during a model outage")
    args = ap.parse_args()

    rows = load_golden()
    print(f"golden set: {len(rows)} questions, "
          f"{len(set(r['answer_doi'] for r in rows))} distinct DOIs\n")

    ret = eval_retrieval(rows)
    print(f"RETRIEVAL  hit@{ret['k']}={ret['hit@k']}  mrr={ret['mrr']}  (n={ret['n']})")
    for q in ret["misses"]:
        print(f"  MISS: {q}")

    if args.retrieval_only:
        print("\n(retrieval-only: faithfulness skipped)")
        return

    ff = eval_faithfulness(rows)
    print(f"\nFAITHFULNESS  mean={ff['mean']}  (graded {ff['graded']}/{len(rows)})")
    for d in ff["detail"]:
        if d.get("unsupported"):
            print(f"  {d['question'][:60]}  score={d.get('score')}  unsupported={d['unsupported']}")


if __name__ == "__main__":
    main()
```

---

## Verify (all must pass)

- Against the **current** `golden_set.jsonl` (2 rows), `python eval.py` exits with
  the refusal message and **no metrics**. This proves the gate works.
- Once the golden set is real: `python eval.py --retrieval-only` prints hit@k and
  MRR with no network call to the model.
- Full `python eval.py` prints a mean faithfulness score and lists any answers
  with unsupported claims.

---

## If it breaks

- **Refusal fires on a real set** → your golden set has < 15 rows; that's correct
  behavior, not a bug. Write more questions.
- **`JUDGE_MODEL` 404 / not available on your tier** → either enable that model,
  or fall back to the Flash generator model AND **document the self-preference
  limitation in the README** (do not hide it — naming it is the signal).
- **429 mid-run** → the backoff from Step 1b should absorb it; if it still dies,
  lower the request rate or add a `time.sleep(4)` between questions.
- **Judge returns non-JSON** → handled (score `None`, flagged); if it happens
  often, tighten the judge prompt.

---

## Do NOT

- Do NOT author, edit, or fabricate `golden_set.jsonl`. If it's short, STOP and
  report blocked — do not invent questions to make the harness runnable.
- Do NOT use the **same** model as generator and faithfulness judge without
  documenting it — self-grading inflates the score.
- Do NOT change chunk size, embedding model, vector dimension (384), or retrieval
  to "improve the numbers." Tuning happens *after* you have a baseline, and every
  change is re-measured against this harness.
- Do NOT report or commit any eval numbers produced from placeholder data.

---

## The gate (read this)

This runbook builds the harness, but the harness is inert until
`golden_set.jsonl` holds **≥ 15 hand-authored questions across diverse DOIs**,
each verified against `rag.py`. Right now it holds 2 placeholders on 1 DOI, so
`eval.py` will refuse to run — by design.

**Building `eval.py` does not advance M6. Writing the golden set does.** An agent
can build every piece of scaffolding around the evaluation; it cannot write the
questions, and the questions are the milestone. When the set is real, run
`--retrieval-only` first (works during the current model outage), then the full
harness once the model is back.
