"""Step 2: Ask a question from the command line.

Examples:
    python 02_ask.py "What does effective challenge mean?"
    python 02_ask.py "What is model risk?" --mode vector --show-text
"""
import argparse

from rag import answer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["hybrid", "vector", "keyword"], default="hybrid")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--show-text", action="store_true", help="print the retrieved passages")
    args = ap.parse_args()

    kwargs = {"mode": args.mode}
    if args.top_k:
        kwargs["top_k"] = args.top_k
    result = answer(args.question, **kwargs)

    print("\n" + result["answer"] + "\n")
    print("Sources:")
    for i, s in enumerate(result["sources"], start=1):
        print(f"  [{i}] {s['doc']}, p. {s['page']}  (score {s['score']})")
        if args.show_text:
            print("      " + s["text"] + "\n")


if __name__ == "__main__":
    main()
