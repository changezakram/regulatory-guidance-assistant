"""Step 1: Parse PDFs in docs/, split them into chunks, embed, and store in DuckDB.

Rebuilds the index from scratch each run, so you can change chunk settings and
re-run to compare. Example:
    python 01_ingest.py
    RAG_CHUNK_WORDS=120 RAG_OVERLAP_WORDS=20 python 01_ingest.py
"""
import re
import unicodedata

import pymupdf

from common import CHUNK_WORDS, DOCS_DIR, EMBED_MODEL, OVERLAP_WORDS, connect, embed


# Pages to index for documents that carry extra material (e.g. a Federal Register
# preamble and comment responses before the guidance itself). (first, last); None = to the end.
# Keys must appear in the PDF file name. Page numbers stay as in the original PDF.
PAGE_RANGES = {
    "Third-Party Risk Guidance": (25, None),
}


def read_pdf(path):
    """Yield (page_number, text) for each page, with whitespace tidied."""
    first, last = next((r for key, r in PAGE_RANGES.items() if key in path.stem), (1, None))
    with pymupdf.open(path) as pdf:
        for i, page in enumerate(pdf, start=1):
            if i < first or (last and i > last):
                continue
            text = unicodedata.normalize("NFKC", page.get_text())  # fixes ligatures like ﬁ -> fi
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                yield i, text


def chunk_words(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    """Split text into overlapping windows of words.

    Chunks stay within a page so every chunk has one page number to cite.
    """
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(size - overlap, 1)
    return [" ".join(words[i:i + size]) for i in range(0, len(words) - overlap, step)]


def main():
    pdfs = sorted(DOCS_DIR.glob("*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs found in {DOCS_DIR}/ . Add some and re-run.")

    rows = []  # (doc, page, text)
    for path in pdfs:
        n_before = len(rows)
        for page, text in read_pdf(path):
            for piece in chunk_words(text):
                if len(piece.split()) >= 25:   # skip near-empty fragments (headers, page numbers)
                    rows.append((path.stem, page, f"Document: {path.stem}. {piece}"))
        print(f"  {path.name}: {len(rows) - n_before} chunks")

    print(f"Embedding {len(rows)} chunks with {EMBED_MODEL}...")
    vectors = embed([r[2] for r in rows], kind="document")
    dim = len(vectors[0])

    con = connect()
    con.execute("DROP TABLE IF EXISTS chunks")
    con.execute(f"""
        CREATE TABLE chunks (
            chunk_id  INTEGER PRIMARY KEY,
            doc       VARCHAR,
            page      INTEGER,
            text      VARCHAR,
            embedding FLOAT[{dim}]
        )""")
    con.executemany("INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                    [(i, d, p, t, v) for i, ((d, p, t), v) in enumerate(zip(rows, vectors))])

    # Record how the index was built, so search uses the same embedding model.
    con.execute("CREATE OR REPLACE TABLE index_info (key VARCHAR, value VARCHAR)")
    con.executemany("INSERT INTO index_info VALUES (?, ?)", [
        ("embed_model", EMBED_MODEL), ("dim", str(dim)),
        ("chunk_words", str(CHUNK_WORDS)), ("overlap_words", str(OVERLAP_WORDS)),
    ])
    print(f"Done. {len(rows)} chunks from {len(pdfs)} documents indexed (vector size {dim}).")


if __name__ == "__main__":
    main()
