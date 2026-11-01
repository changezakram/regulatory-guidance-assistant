"""Shared settings and helpers: configuration, Ollama calls, database."""
import json
import os
from pathlib import Path

import duckdb
import requests

# ---- Settings (override any of these with environment variables) ----------
DB_PATH = Path(os.getenv("RAG_DB_PATH", "rag.duckdb"))
DOCS_DIR = Path(os.getenv("RAG_DOCS_DIR", "docs"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
# Answers: "ollama" (Llama on your Mac) or "anthropic" (Claude via the API;
# needs ANTHROPIC_API_KEY set in the environment or in Streamlit secrets).
CHAT_BACKEND = os.getenv("RAG_CHAT_BACKEND", "ollama")
CHAT_MODEL = os.getenv("RAG_CHAT_MODEL", "claude-haiku-5-5"
                       if CHAT_BACKEND == "anthropic" else "llama3.1:8b")
# Embeddings: "fastembed" runs inside Python (works locally and in the cloud);
# "ollama" uses the Ollama app on your Mac.
EMBED_BACKEND = os.getenv("RAG_EMBED_BACKEND", "fastembed")
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "nomic-ai/nomic-embed-text-v1.5"
                        if EMBED_BACKEND == "fastembed" else "nomic-embed-text")
CHUNK_WORDS = int(os.getenv("RAG_CHUNK_WORDS", "220"))     # words per chunk
OVERLAP_WORDS = int(os.getenv("RAG_OVERLAP_WORDS", "40"))  # words shared between neighbours
TOP_K = int(os.getenv("RAG_TOP_K", "5"))                   # passages sent to the LLM
NEIGHBORS = int(os.getenv("RAG_NEIGHBORS", "1"))           # chunks added on each side of a hit
CONTEXT_TOKENS = int(os.getenv("RAG_CONTEXT_TOKENS", "8192"))  # how much text the model can read

REFUSAL = "I couldn't find this in the documents."


def connect():
    return duckdb.connect(str(DB_PATH))


# ---- Embeddings ---------------------------------------------------------------
def _prefix(kind: str) -> str:
    """nomic-embed-text expects task prefixes; other models don't need them."""
    if "nomic" not in EMBED_MODEL:
        return ""
    return "search_document: " if kind == "document" else "search_query: "


_fastembed_model = None


def _embed_fastembed(texts: list[str]) -> list[list[float]]:
    global _fastembed_model
    if _fastembed_model is None:
        from fastembed import TextEmbedding  # downloads the model on first use
        _fastembed_model = TextEmbedding(EMBED_MODEL)
    return [v.tolist() for v in _fastembed_model.embed(texts)]


def embed(texts: list[str], kind: str = "document", batch: int = 32) -> list[list[float]]:
    """Turn texts into vectors. kind is 'document' (when indexing) or 'query'."""
    if EMBED_BACKEND == "fastembed":
        return _embed_fastembed([_prefix(kind) + t for t in texts])
    out = []
    for i in range(0, len(texts), batch):
        chunk = [_prefix(kind) + t for t in texts[i:i + batch]]
        r = requests.post(f"{OLLAMA_URL}/api/embed",
                          json={"model": EMBED_MODEL, "input": chunk}, timeout=300)
        r.raise_for_status()
        out.extend(r.json()["embeddings"])
    return out


# ---- Chat model ---------------------------------------------------------------
def chat(messages: list[dict], fmt=None) -> str:
    """Call the chat model. fmt can be a JSON schema to force structured output."""
    if CHAT_BACKEND == "anthropic":
        return _chat_anthropic(messages, fmt)
    body = {"model": CHAT_MODEL, "messages": messages, "stream": False,
            "options": {"temperature": 0, "num_ctx": CONTEXT_TOKENS}}
    if fmt is not None:
        body["format"] = fmt
    r = requests.post(f"{OLLAMA_URL}/api/chat", json=body, timeout=600)
    r.raise_for_status()
    return r.json()["message"]["content"]


_anthropic_client = None


def _chat_anthropic(messages: list[dict], fmt=None) -> str:
    """Call Claude. The API takes the system prompt separately from the conversation."""
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic  # reads ANTHROPIC_API_KEY from the environment
        _anthropic_client = anthropic.Anthropic()
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    convo = [m for m in messages if m["role"] != "system"]
    if fmt is not None:
        system += ("\n\nRespond with only a JSON object matching this schema, "
                   f"and nothing else:\n{json.dumps(fmt)}")
    resp = _anthropic_client.messages.create(
        model=CHAT_MODEL, max_tokens=4000, system=system, messages=convo)
    text = "".join(b.text for b in resp.content if b.type == "text")
    if resp.stop_reason == "max_tokens" and fmt is None:
        text += "\n\n[Answer cut short: it reached the length limit.]"
    if fmt is not None:  # keep only the JSON object, in case of stray text around it
        start, end = text.find("{"), text.rfind("}")
        text = text[start:end + 1] if start >= 0 else text
    return text
