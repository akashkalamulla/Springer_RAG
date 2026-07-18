import argparse
import json

from rag import retrieve, answer          # reuse the real pipeline, don't fork it
from llm import generate

GOLDEN = "golden_set.jsonl"
TOP_K = 5
MIN_QUESTIONS = 15                        # refuse to score a placeholder set
JUDGE_MODEL = "gpt-4o"             # MUST differ from the Flash generator


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
            "You are grading FAITHFULNESS only: is every factual claim the ANSWER "
            "asserts supported by the CONTEXT? Do not grade completeness or correctness.\n"
            "Apply these rules strictly:\n"
            "1. If the ANSWER states that some information is not in the context, that is "
            "FULLY faithful. Do not list it as unsupported and do not lower the score for it.\n"
            "2. Facts that appear in the QUESTION are given. If the ANSWER restates them, "
            "that is NOT an unsupported claim.\n"
            "3. Only list a claim as unsupported if the ANSWER asserts it as fact AND it is "
            "absent from the CONTEXT.\n"
            'Return ONLY JSON: {"score": <0.0-1.0>, "unsupported": [<ungrounded claims>]}.\n'
            "score 1.0 = every asserted claim is grounded, or the answer correctly says the "
            "information is absent. Subtract only for genuinely ungrounded assertions.\n\n"
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
    api_errors = 0
    for r in rows:
        try:
            answer_text, hits = answer(r["question"])   # the REAL pipeline output
            detail.append({"question": r["question"], **judge(r["question"], answer_text, hits)})
        except Exception as e:
            api_errors += 1
            detail.append({
                "question": r["question"],
                "score": None,
                "status": "api_error",
                "error": str(e),
            })
    scored = [d["score"] for d in detail if d.get("score") is not None]
    return {
        "mean": round(sum(scored) / len(scored), 3) if scored else None,
        "graded": len(scored),
        "api_errors": api_errors,
        "total": len(rows),
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
    print(f"\nFAITHFULNESS  mean={ff['mean']}  (graded {ff['graded']}/{len(rows)}, {ff['api_errors']} API errors)")
    for d in ff["detail"]:
        if d.get("status") == "api_error":
            print(f"  API ERROR: {d['question'][:50]}  ->  {d['error']}")
        elif d.get("unsupported"):
            print(f"  {d['question'][:60]}  score={d.get('score')}  unsupported={d['unsupported']}")


if __name__ == "__main__":
    main()
