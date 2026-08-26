"""Embedding backends.

Two implementations behind one small interface:

* GeminiEmbedder    - real semantic embeddings via the free `text-embedding-004`
                      model. Used when a GEMINI_API_KEY is available.
* TfidfEmbedder     - a dependency-light, fully deterministic lexical embedder
                      (hashed TF-IDF vectors). It needs no API key and no
                      network, which lets the retrieval layer and most of the
                      evaluation suite run offline and reproducibly.

Both return L2-normalized float32 vectors, so a dot product is cosine similarity.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

import numpy as np

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (mat / norms).astype(np.float32)


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class TfidfEmbedder:
    """Deterministic hashed TF-IDF embedder. No key, no network.

    IDF is fit on the document set. Query vectors reuse the fitted IDF so query
    and document vectors share a space. Hashing keeps the vocabulary bounded
    without a stored vocab map.
    """

    name = "tfidf"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self._idf = np.ones(dim, dtype=np.float32)
        self._fitted = False

    def _hash(self, token: str) -> int:
        # Stable across runs (Python's hash() is salted; use a fixed digest).
        h = 2166136261
        for ch in token:
            h = (h ^ ord(ch)) * 16777619 & 0xFFFFFFFF
        return h % self.dim

    def _raw_counts(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok, count in Counter(_tokenize(text)).items():
            vec[self._hash(tok)] += count
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        counts = np.vstack([self._raw_counts(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)
        df = (counts > 0).sum(axis=0)
        n_docs = max(len(texts), 1)
        self._idf = np.log((1 + n_docs) / (1 + df)).astype(np.float32) + 1.0
        self._fitted = True
        tf = np.log1p(counts)
        return _normalize_rows(tf * self._idf)

    def embed_query(self, text: str) -> np.ndarray:
        tf = np.log1p(self._raw_counts(text))
        vec = (tf * self._idf).reshape(1, -1)
        return _normalize_rows(vec)[0]


class GeminiEmbedder:
    """Semantic embeddings via google-genai `text-embedding-004`."""

    name = "gemini"

    def __init__(self, api_key: str, model: str = "text-embedding-004") -> None:
        from google import genai

        self._genai = genai
        self._client = genai.Client(api_key=api_key)
        self.model = model

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        vectors: list[list[float]] = []
        # The API accepts batches; keep them modest for the free tier.
        for start in range(0, len(texts), 100):
            batch = texts[start:start + 100]
            resp = self._client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            vectors.extend(e.values for e in resp.embeddings)
        return _normalize_rows(np.array(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)
        return self._embed(texts, task_type="RETRIEVAL_DOCUMENT")

    def embed_query(self, text: str) -> np.ndarray:
        return self._embed([text], task_type="RETRIEVAL_QUERY")[0]


def build_embedder(backend: str, *, api_key: str | None = None, model: str = "text-embedding-004") -> Embedder:
    if backend == "gemini":
        if not api_key:
            raise ValueError("Gemini embedding backend requires an API key.")
        return GeminiEmbedder(api_key=api_key, model=model)
    if backend == "tfidf":
        return TfidfEmbedder()
    raise ValueError(f"Unknown embedding backend: {backend!r}")
