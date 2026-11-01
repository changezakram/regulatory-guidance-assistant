"""Step 3: Evaluate the assistant against your own question set.

Fill in eval_questions.csv first (see README). Then:
    python 03_eval.py                    # hybrid retrieval
    python 03_eval.py --mode vector      # compare retrieval modes
Writes eval_results.csv with one row per question (saved as it goes).

key_facts column (optional): the facts a correct answer must contain.
Separate facts with ";" and alternative wordings of one fact with "|".
Example:  conceptual soundness; ongoing monitoring; outcomes analysis|back-testing
"""
import argparse
import csv
import json
import re

from common import CHAT_MODEL, NEIGHBORS, REFUSAL, TOP_K, chat
from rag import answer

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "grounded": {"type": "string", "enum": ["yes", "partial", "no"]},
        "unsupported_claims": {"type": "string"},
    },
    "required": ["grounded", "unsupported_claims"],
}

JUDGE_PROMPT = """You check whether an answer is supported by its sources.
grounded = yes if every factual claim in the answer is supported by the sources,
partial if some claims are supported and some are not, no if the main claims are not.
List any unsupported claims briefly, or write "none"."""

# Ways the model says "I don't know" without using the exact refusal sentence.
REFUSAL_PATTERNS = [REFUSAL.lower().rstrip("."), "couldn't find", "could not find",
                    "not found in the provided sources", "sources do not contain"]

FIELDS = ["question", "answerable", "refused", "retrieval_hit", "facts_found", "facts_missing",
          "grounded", "unsupported_claims", "answer", "sources"]


def norm(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def simplify(s):
    """Lowercase and collapse punctuation and spacing, for fact matching."""
    s = s.lower().replace("’", "'").replace("‐", "-").replace("‑", "-")
    return re.sub(r"[\s\-]+", " ", s)


def is_refusal(text):
    t = text.lower()
    return any(p in t for p in REFUSAL_PATTERNS)


def retrieval_hit(sources, expected_doc, expected_page):
    """Did any retrieved passage come from the expected document (and exact page, if given)?"""
    for s in sources:
        if norm(expected_doc) and norm(expected_doc) in norm(s["doc"]):
            if not expected_page or int(expected_page) == s["page"]:
                return True
    return False


def check_facts(answer_text, key_facts):
    """Return (found, missing) lists of facts. Each fact passes if any alternative appears."""
    text = simplify(answer_text)
    found, missing = [], []
    for fact in [f.strip() for f in key_facts.split(";") if f.strip()]:
        alts = [simplify(a.strip()) for a in fact.split("|") if a.strip()]
        (found if any(a in text for a in alts) else missing).append(fact.split("|")[0].strip())
    return found, missing


def judge(result):
    context = "\n\n".join(f"[{i}] {s['text']}" for i, s in enumerate(result["sources"], 1))
    out = chat([
        {"role": "system", "content": JUDGE_PROMPT},
        {"role": "user", "content": f"Sources:\n{context}\n\nAnswer:\n{result['answer']}"},
    ], fmt=JUDGE_SCHEMA)
    return json.loads(out)


def load_questions(path):
    """Read the question file, skipping malformed lines with a warning instead of crashing."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    good = []
    for n, r in enumerate(rows, start=2):  # line 1 is the header
        if not (r.get("question") or "").strip():
            continue
        if r.get("answerable") is None or None in r:
            print(f"  Skipping line {n}: wrong number of columns -> {r.get('question', '')[:50]}")
            continue
        good.append(r)
    return good


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="eval_questions.csv")
    ap.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    ap.add_argument("--top-k", type=int, default=TOP_K)
    ap.add_argument("--no-judge", action="store_true", help="skip the LLM groundedness judge (faster)")
    args = ap.parse_args()

    questions = load_questions(args.file)
    results = []
    out = open("eval_results.csv", "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out, fieldnames=FIELDS)
    writer.writeheader()

    for i, q in enumerate(questions, 1):
        answerable = q["answerable"].strip().lower() in ("yes", "y", "true", "1")
        r = answer(q["question"], top_k=args.top_k, mode=args.mode)
        refused = is_refusal(r["answer"])
        row = {k: "" for k in FIELDS}
        row.update({"question": q["question"], "answerable": answerable, "refused": refused,
                    "answer": r["answer"],
                    "sources": "; ".join(f"{s['doc']} p.{s['page']}" for s in r["sources"])})
        if answerable:
            row["retrieval_hit"] = retrieval_hit(r["sources"], q.get("expected_doc", ""),
                                                 (q.get("expected_page") or "").strip())
            facts = (q.get("key_facts") or "").strip()
            if facts:
                found, missing = check_facts(r["answer"], facts)
                row["facts_found"] = f"{len(found)}/{len(found) + len(missing)}"
                row["facts_missing"] = "; ".join(missing)
            if not refused and not args.no_judge:
                j = judge(r)
                row["grounded"], row["unsupported_claims"] = j["grounded"], j["unsupported_claims"]
        results.append(row)
        writer.writerow(row)
        out.flush()  # keep finished rows even if the run stops early
        print(f"  {i}/{len(questions)}  {q['question'][:60]}")
    out.close()

    ans = [r for r in results if r["answerable"]]
    unans = [r for r in results if not r["answerable"]]
    graded = [r for r in ans if r["grounded"]]
    keyed = [r for r in ans if r["facts_found"]]
    complete = [r for r in keyed if not r["facts_missing"]]
    wrong = [r for r in keyed if r["facts_missing"] and not r["refused"]]

    def pct(n, d):
        return f"{100 * n / d:.0f}% ({n}/{d})" if d else "n/a"

    print(f"\nResults ({CHAT_MODEL}, {args.mode}, top {args.top_k}, neighbors {NEIGHBORS}):")
    print(f"  Retrieval hit rate:        {pct(sum(r['retrieval_hit'] is True for r in ans), len(ans))}")
    print(f"  Answered (not refused):    {pct(sum(not r['refused'] for r in ans), len(ans))}")
    if graded:
        print(f"  Fully grounded answers:    {pct(sum(r['grounded'] == 'yes' for r in graded), len(graded))}")
    print(f"  Correct refusals:          {pct(sum(r['refused'] for r in unans), len(unans))}")
    if keyed:
        print(f"  All key facts present:     {pct(len(complete), len(keyed))}")
        print(f"  Answered but incomplete:   {pct(len(wrong), len(keyed))}")
        for r in wrong:
            print(f"      - {r['question'][:55]}  missing: {r['facts_missing']}")
    print("Details in eval_results.csv")


if __name__ == "__main__":
    main()
