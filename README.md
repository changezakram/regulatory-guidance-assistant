# Regulatory Guidance Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about bank model risk, AI risk and third-party risk guidance, citing the document and page for every claim. Built as a personal proof of concept on public documents only, and evaluated against an expert-written answer key.

**Live demo:** _add your Streamlit link here_

## What it does

Ask a question in plain English. The assistant searches five public documents, retrieves the most relevant passages, and has a language model answer only from those passages, with citations. If the documents don't cover the question, it says so instead of guessing.

| Document | Issuer |
|---|---|
| SR 26-2, Supervisory Guidance on Model Risk Management (2026) | Federal Reserve, FDIC, OCC |
| SR 11-7, Supervisory Guidance on Model Risk Management (2011, superseded) | Federal Reserve, OCC |
| AI Risk Management Framework 1.0 (AI 100-1) | NIST |
| Generative AI Profile (AI 600-1) | NIST |
| Interagency Guidance on Third-Party Relationships: Risk Management (2023) | Federal Reserve, FDIC, OCC |

## Results

The test set has 26 questions: 21 the documents can answer and 5 they cannot. Each run changes one thing.

**Phase 1: graded by hand** (21 answerable questions)

| Run | Change | Fully correct | Correct refusals |
|---|---|---|---|
| 1 | Baseline: hybrid search, Llama 3.1 8B | 52% | 100% |
| 2 | Stamp each passage with its document name | 67% | 100% |
| 3 | Search only the documents a question names | 76% | 100% |

**Phase 2: automated answer key** (19 questions with a list of facts a complete answer must contain)

| Run | Change | Key facts present | Correct refusals |
|---|---|---|---|
| 4 | Same system as run 3, answer key introduced | 74% | 100% |
| 5 | Include neighboring passages around each search hit | 79% | 100% |
| 6 | Add a search glossary (everyday wording to regulatory wording)* | 89% | 100% |
| 7 | Remove Federal Register background and comment pages from the index | 95% | 100% |
| 8 | Same pipeline, embeddings moved in-process for hosting | 89% | 100% |
| 9 | Same pipeline, Claude Haiku 5.5 writes the answers | 89% | 100% |

\*Run 6 also includes one answer-key correction; the glossary itself fixed one question. With 19 keyed questions, one question moves the score about 5 points, so runs 7 to 9 are within one question of each other.

## What I learned

1. **Data and search drove every gain; swapping the model did not.** Every improvement came from changes to the documents and the search. Replacing an 8B local model with Claude left completeness unchanged.
2. **The model mattered for honesty, not completeness.** The smaller model filled gaps with plausible content, at one point inventing two stages of the third-party life cycle. Claude stated when the passages did not cover something, flagged a superseded source on its own, and cited every claim.
3. **An AI judge grading answers is not validation evidence.** In run 4, an LLM groundedness judge rated every answer as fully grounded, while the answer key found 5 of 19 incomplete or unanswered, including one with invented content.
4. **Document preparation is most of the work.** The largest single gain came from removing 24 pages of Federal Register background and public-comment responses that crowded out the guidance itself. Other problems were a key list that existed only in a figure, user vocabulary that differs from regulatory vocabulary ("worsened" vs "exacerbated"), and PDF character encoding that broke keyword search.
5. **SR 26-2 places generative and agentic AI outside its scope**, leaving banks to govern systems like this one under their own risk management practices.

## How it works

```
PDFs -> text by page -> ~220-word chunks -> embeddings + keyword index (DuckDB)
                                                      |
question -> glossary expansion -> document scoping -> hybrid search (vector + BM25)
                                                      |
             top passages + neighbors -> LLM (answer only from sources, cite) -> answer
```

- **Hybrid search:** meaning-based (nomic-embed-text) and keyword (BM25) rankings, merged with reciprocal rank fusion.
- **Document scoping:** a question that names a document searches only that document; a comparison searches each named document separately.
- **Neighbors:** each hit includes the adjacent chunk on either side, so lists that span chunks arrive whole.
- **Glossary:** a small, reviewable mapping from everyday terms to regulatory terms, applied before search.
- **Evaluation:** retrieval hit rate, refusal accuracy, and an answer key checked by plain text match rather than by a model.

## Run it locally

Requires Python 3.12. Ollama is needed only to run the local Llama model.

```
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python 01_ingest.py                         # build the index from docs/
python 02_ask.py "Does SR 26-2 apply to generative AI?"
streamlit run app.py                        # web page at localhost:8501
python 03_eval.py --no-judge                # run the evaluation
```

By default answers come from Llama 3.1 8B through Ollama (`ollama pull llama3.1:8b`). To use Claude instead, set an Anthropic API key and `RAG_CHAT_BACKEND=anthropic`.

## Limitations

- Five documents and 26 questions: a controlled test, not a production benchmark.
- Text extraction ignores figures and tables rendered as images.
- The answer key checks that required facts are present, not that unsupported ones are absent.
- Most test questions name their source document; questions that don't are likely harder.
- Not legal or regulatory advice.

## Repository contents

| File | Purpose |
|---|---|
| `01_ingest.py` | Parse PDFs, chunk, embed, store in DuckDB |
| `rag.py` | Search, scoping, glossary, neighbors, answer generation |
| `common.py` | Settings, embedding and model calls |
| `02_ask.py` | Ask from the command line |
| `03_eval.py` | Evaluation with answer key |
| `app.py` | Streamlit web page with usage guardrails |
| `eval_questions.csv` | Questions, expected sources, answer key |
| `results/` | Results from each evaluation run |
| `docs/` | The five source PDFs (U.S. government publications) |
| `rag.duckdb` | The prebuilt index used by the hosted app |
