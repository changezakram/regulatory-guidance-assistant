"""Core RAG logic: hybrid retrieval (vector + keyword), document scoping, cited answers."""
import math
import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from common import EMBED_MODEL, NEIGHBORS, REFUSAL, TOP_K, chat, connect, embed

SYSTEM_PROMPT = f"""You answer questions about banking regulation and supervisory guidance.
Use ONLY the numbered sources provided. Do not use outside knowledge.
Cite sources inline after each claim, like [1] or [2][3].
If the sources do not contain the answer, reply exactly: {REFUSAL}
Be concise: a short paragraph or a few bullet points."""

# When a question names a document, search only that document.
# Keys must appear in the PDF file name; values are phrases to look for in the question.
DOC_ALIASES = {
    "SR 26-2": ["sr 26-2", "sr26-2"],
    "SR 11-7": ["sr 11-7", "sr11-7"],
    "NIST AI RMF": ["ai rmf", "ai risk management framework"],
    "NIST GenAI Profile": ["generative ai profile", "genai profile", "ai 600-1"],
    "Third-Party Risk Guidance": ["third-party risk guidance", "guidance on third-party relationships"],
}


# Glossary: words people use -> words the documents use. Added to the question
# before searching (the model still sees the original question). Extend as you find gaps.
QUERY_GLOSSARY = {
    "worsened": ["exacerbated"],
    "worsen": ["exacerbate"],
    "made worse": ["exacerbated"],
    "hallucination": ["confabulation"],
    "hallucinations": ["confabulation"],
    "hallucinate": ["confabulate"],
    "genai": ["generative AI"],
    "gen ai": ["generative AI"],
    "vendor": ["third-party", "third party"],
    "vendors": ["third parties"],
    "lifecycle": ["life cycle"],
    "life cycle": ["lifecycle"],
    "offboarding": ["termination"],
    "exit": ["termination"],
}


def expand_query(question):
    """Append document vocabulary for any glossary terms found in the question."""
    q = question.lower()
    extra = []
    for term, synonyms in QUERY_GLOSSARY.items():
        if re.search(rf"\b{re.escape(term)}\b", q):
            extra += [s for s in synonyms if s.lower() not in q and s not in extra]
    return f"{question} ({', '.join(extra)})" if extra else question


def _tokens(text):
    return re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text.lower())


@lru_cache(maxsize=1)
def _load():
    """Load chunks once and build the keyword index."""
    con = connect()
    info = dict(con.execute("SELECT key, value FROM index_info").fetchall())
    if info["embed_model"] != EMBED_MODEL:
        raise SystemExit(f"Index was built with {info['embed_model']} but RAG_EMBED_MODEL is "
                         f"{EMBED_MODEL}. Re-run 01_ingest.py or change the setting.")
    chunks = con.execute("SELECT chunk_id, doc, page, text FROM chunks ORDER BY chunk_id").fetchall()
    bm25 = BM25Okapi([_tokens(c[3]) for c in chunks])
    return con, int(info["dim"]), chunks, bm25


def named_documents(question):
    """Return the indexed documents the question mentions by name, in the order mentioned."""
    _, _, chunks, _ = _load()
    all_docs = sorted({c[1] for c in chunks})
    q = question.lower()
    found = []
    for key, phrases in DOC_ALIASES.items():
        positions = [q.find(p) for p in phrases if p in q]
        if not positions:
            continue
        for doc in all_docs:
            if key.lower() in doc.lower() and doc not in [d for _, d in found]:
                found.append((min(positions), doc))
    return [doc for _, doc in sorted(found)]


def vector_search(question, k=20, docs=None):
    """Rank chunks by meaning: cosine similarity between embeddings.
    If docs is given, only search those documents."""
    con, dim, _, _ = _load()
    qvec = embed([question], kind="query")[0]
    where = "WHERE list_contains(?, doc)" if docs else ""
    params = [qvec, list(docs)] if docs else [qvec]
    rows = con.execute(f"""
        SELECT chunk_id, array_cosine_similarity(embedding, ?::FLOAT[{dim}]) AS score
        FROM chunks {where} ORDER BY score DESC LIMIT {int(k)}""", params).fetchall()
    return [r[0] for r in rows]


def keyword_search(question, k=20, docs=None):
    """Rank chunks by exact terms (BM25): good for names like 'SR 11-7' or 'Reg E'.
    If docs is given, only search those documents."""
    _, _, chunks, bm25 = _load()
    scores = bm25.get_scores(_tokens(question))
    keep = [i for i in range(len(chunks)) if not docs or chunks[i][1] in docs]
    ranked = sorted(keep, key=lambda i: scores[i], reverse=True)
    return [chunks[i][0] for i in ranked[:k] if scores[i] > 0]


def _fused_ranking(question, mode, docs=None):
    """Merge vector and keyword rankings with reciprocal rank fusion:
    a chunk scores 1/(60 + rank) in each list it appears in, and the scores are added."""
    rankings = []
    if mode != "keyword":
        rankings.append(vector_search(question, docs=docs))
    if mode != "vector":
        rankings.append(keyword_search(question, docs=docs))
    fused = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking):
            fused[cid] = fused.get(cid, 0) + 1 / (60 + rank)
    return fused


def _select(ranking, n, by_id, neighbors):
    """Take the top n hits from a ranking. With neighbors > 0, each hit's passage also
    includes the chunks just before and after it in the same document, so lists and
    tables that run across chunks come through whole. Hits already included as
    someone else's neighbor are skipped, so the n passages don't repeat each other."""
    picked, covered = [], set()
    for cid in ranking:
        if len(picked) == n:
            break
        if cid in covered:
            continue
        doc = by_id[cid][1]
        window = [c for c in range(cid - neighbors, cid + neighbors + 1)
                  if c in by_id and by_id[c][1] == doc]
        covered.update(window)
        picked.append((cid, window))
    return picked


def retrieve(question, top_k=TOP_K, mode="hybrid", scope=True, neighbors=NEIGHBORS):
    """Return the top passages. mode: 'hybrid', 'vector', or 'keyword'.

    scope=True: if the question names one document, search only it; if it names
    several (a comparison), search each separately and take an equal share from each.
    neighbors: how many adjacent chunks to add on each side of every hit.
    """
    _, _, chunks, _ = _load()
    by_id = {c[0]: c for c in chunks}
    named = named_documents(question) if scope else []
    query = expand_query(question)  # search with document vocabulary added

    if len(named) <= 1:
        fused = _fused_ranking(query, mode, docs=named or None)
        picked = _select(sorted(fused, key=fused.get, reverse=True), top_k, by_id, neighbors)
    else:
        per_doc = math.ceil(top_k / len(named))
        picked, fused = [], {}
        for doc in named:
            f = _fused_ranking(query, mode, docs=[doc])
            fused.update(f)
            picked += _select(sorted(f, key=f.get, reverse=True), per_doc, by_id, neighbors)

    return [{"chunk_id": cid, "doc": by_id[cid][1], "page": by_id[cid][2],
             "text": "\n".join(by_id[c][3] for c in window),
             "score": round(fused[cid], 4)} for cid, window in picked]


def answer(question, top_k=TOP_K, mode="hybrid", scope=True, neighbors=NEIGHBORS):
    """Retrieve passages, then ask the LLM to answer from them with citations."""
    sources = retrieve(question, top_k, mode, scope, neighbors)
    context = "\n\n".join(f"[{i}] ({s['doc']}, p. {s['page']})\n{s['text']}"
                          for i, s in enumerate(sources, start=1))
    reply = chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Sources:\n{context}\n\nQuestion: {question}"},
    ])
    return {"question": question, "answer": reply.strip(), "sources": sources,
            "scope": named_documents(question) if scope else []}
