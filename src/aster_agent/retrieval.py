"""Vector index + precedence-aware retrieval.

Retrieval has two jobs beyond plain similarity:

1. Document precedence. The corpus deliberately mixes an active policy with a
   superseded one, plus a non-authoritative draft "migration scratchpad". We
   multiply raw similarity by an *authority weight* so active/official content
   outranks superseded or draft content. Low-authority chunks stay retrievable
   (the agent may need to say "the legacy doc is superseded") but never win by
   default.

2. Conflict surfacing. When two *active official* documents both score highly on
   the same query (e.g. product care vs. the product card on dishwasher safety),
   both are returned so the agent can surface the conflict instead of silently
   picking one.

Embeddings are cached on disk keyed by (backend, corpus fingerprint) so repeated
runs — especially with the Gemini backend — don't re-embed the corpus.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import REPO_ROOT
from .embeddings import Embedder
from .ingestion import Chunk, load_corpus

CACHE_DIR = REPO_ROOT / ".kb_cache"

# Authority weights applied to raw cosine similarity.
_WEIGHT_ACTIVE_OFFICIAL = 1.0
_WEIGHT_SUPERSEDED = 0.40
_WEIGHT_NON_AUTHORITATIVE = 0.25  # draft, or policy_authority == none
_WEIGHT_OTHER = 0.70


def authority_weight(chunk: Chunk) -> float:
    if chunk.status == "draft" or chunk.policy_authority == "none":
        return _WEIGHT_NON_AUTHORITATIVE
    if chunk.is_superseded:
        return _WEIGHT_SUPERSEDED
    if chunk.is_active_official:
        return _WEIGHT_ACTIVE_OFFICIAL
    return _WEIGHT_OTHER


@dataclass
class RetrievedChunk:
    chunk: Chunk
    similarity: float  # raw cosine similarity
    score: float       # similarity * authority_weight (used for ranking)

    def to_debug(self) -> dict:
        return {
            "source": self.chunk.citation,
            "doc_id": self.chunk.doc_id,
            "status": self.chunk.status,
            "policy_authority": self.chunk.policy_authority,
            "similarity": round(self.similarity, 4),
            "authority_weight": round(authority_weight(self.chunk), 3),
            "score": round(self.score, 4),
        }


def _corpus_fingerprint(chunks: list[Chunk]) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c.doc_filename.encode())
        h.update(c.heading.encode())
        h.update(c.text.encode())
    return h.hexdigest()[:16]


class KnowledgeBase:
    def __init__(self, chunks: list[Chunk], matrix: np.ndarray, embedder: Embedder) -> None:
        self.chunks = chunks
        self.matrix = matrix  # (n_chunks, dim), L2-normalized rows
        self.embedder = embedder

    @classmethod
    def build(cls, embedder: Embedder, *, use_cache: bool = True, kb_dir: Path | None = None) -> "KnowledgeBase":
        chunks = load_corpus(kb_dir)
        fingerprint = _corpus_fingerprint(chunks)
        cache_path = CACHE_DIR / f"{embedder.name}-{fingerprint}.npy"

        if use_cache and embedder.name == "gemini" and cache_path.exists():
            matrix = np.load(cache_path)
            # TF-IDF must re-fit IDF at load; Gemini vectors are self-contained.
            if matrix.shape[0] == len(chunks):
                return cls(chunks, matrix, embedder)

        matrix = embedder.embed_documents([c.embedding_text for c in chunks])

        if use_cache and embedder.name == "gemini":
            CACHE_DIR.mkdir(exist_ok=True)
            np.save(cache_path, matrix)

        return cls(chunks, matrix, embedder)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        if not self.chunks:
            return []
        qvec = self.embedder.embed_query(query)
        sims = self.matrix @ qvec  # cosine similarity, both sides normalized
        results: list[RetrievedChunk] = []
        for idx, sim in enumerate(sims):
            chunk = self.chunks[idx]
            sim_f = float(sim)
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity=sim_f,
                    score=sim_f * authority_weight(chunk),
                )
            )
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]
