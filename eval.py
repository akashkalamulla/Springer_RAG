import argparse
import json
import os

from rag import retrieve, answer          # reuse the real pipeline, don't fork it
from llm import generate

GOLDEN = "golden_set.jsonl"
TOP_K = 5
MIN_QUESTIONS = 15
MIN_REFUSE = 8                            # out-of-corpus refusal set floor
HARD = "golden_hard.jsonl"
REFUSE = "golden_refuse.jsonl"                   
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


def eval_retrieval(rows, max_k=5):
    """Local only. Retrieve once at max_k; compute hit@1/@3/@5 and MRR."""
    hit1 = hit3 = hit5 = 0
    ranks, misses = [], []
    for r in rows:
        dois = [h["doi"] for h in retrieve(r["question"], max_k)]
        tgt = r["answer_doi"]
        if tgt in dois:
            rank = dois.index(tgt) + 1
            ranks.append(rank)
            hit1 += rank <= 1
            hit3 += rank <= 3
            hit5 += rank <= 5
        else:
            misses.append(r["question"])
    n = len(rows)
    return {
        "n": n,
        "hit@1": round(hit1 / n, 3),
        "hit@3": round(hit3 / n, 3),
        "hit@5": round(hit5 / n, 3),
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

def classify_refusal(answer_text):
    """True if the answer refused / said the info is absent; False if it asserted
    a claim. Uses the judge model, not brittle string matching."""
    prompt = (
        'Classify the ANSWER. Return ONLY JSON {"refused": true|false}.\n'
        "refused=true if it states the information is not in the context, declines, "
        "or says it cannot find the answer. refused=false if it asserts any "
        "substantive factual answer.\n\n"
        f"ANSWER: {answer_text}\n\nJSON:"
    )
    raw = generate(prompt, model=JUDGE_MODEL, temperature=0.0)
    raw = raw.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        return bool(json.loads(raw).get("refused"))
    except json.JSONDecodeError:
        return None


def eval_refusal(rows):
    """Out-of-corpus questions the system SHOULD refuse. Score = correct / total."""
    correct, api_errors, detail = 0, 0, []
    for r in rows:
        try:
            answer_text, _ = answer(r["question"])
            refused = classify_refusal(answer_text)
            ok = refused is True
            correct += int(ok)
            detail.append({"question": r["question"], "ok": ok})
        except Exception as e:
            api_errors += 1
            detail.append({"question": r["question"], "ok": False, "error": str(e)})
    n = len(rows)
    return {"n": n, "refusal_rate": round(correct / n, 3) if n else None,
            "api_errors": api_errors, "detail": detail}


def load_set(path, min_n, kind):
    if not os.path.exists(path):
        return None
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if len(rows) < min_n:
        raise SystemExit(f"{path}: {len(rows)} rows, need >= {min_n} {kind}. Refusing.")
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true",
                    help="skip faithfulness + refusal (no LLM)")
    args = ap.parse_args()

    easy = load_golden()
    print(f"EASY  {len(easy)}q / {len(set(r['answer_doi'] for r in easy))} DOIs")
    e = eval_retrieval(easy)
    print(f"  hit@1={e['hit@1']}  hit@3={e['hit@3']}  hit@5={e['hit@5']}  mrr={e['mrr']}")

    hard = load_set(HARD, MIN_QUESTIONS, "paraphrased questions")
    if hard:
        print(f"\nHARD  {len(hard)}q / {len(set(r['answer_doi'] for r in hard))} DOIs")
        h = eval_retrieval(hard)
        print(f"  hit@1={h['hit@1']}  hit@3={h['hit@3']}  hit@5={h['hit@5']}  mrr={h['mrr']}")
        for q in h["misses"]:
            print(f"    MISS: {q[:70]}")
    else:
        print("\nHARD  golden_hard.jsonl not found — Phase A task 1 not done.")

    if args.retrieval_only:
        print("\n(retrieval-only: faithfulness + refusal skipped)")
        return

    ff = eval_faithfulness(easy)
    print(f"\nFAITHFULNESS(easy)  mean={ff['mean']}  graded={ff['graded']}/{len(easy)}  errs={ff['api_errors']}")

    refuse = load_set(REFUSE, MIN_REFUSE, "out-of-corpus questions")
    if refuse:
        rf = eval_refusal(refuse)
        print(f"\nREFUSAL  correct={rf['refusal_rate']}  n={rf['n']}  errs={rf['api_errors']}")
        for d in rf["detail"]:
            if not d["ok"]:
                print(f"    FAILED TO REFUSE: {d['question'][:70]}")
    else:
        print("\nREFUSAL  golden_refuse.jsonl not found — Phase A task 2 not done.")


if __name__ == "__main__":
    main()
