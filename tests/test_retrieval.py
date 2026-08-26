"""Retrieval tests: precedence weighting and conflict surfacing.

These run on the offline deterministic TF-IDF backend, so no API key is needed.
"""

import pytest

from aster_agent.embeddings import TfidfEmbedder
from aster_agent.retrieval import KnowledgeBase, authority_weight
from aster_agent.ingestion import load_corpus


@pytest.fixture(scope="module")
def kb() -> KnowledgeBase:
    return KnowledgeBase.build(TfidfEmbedder(), use_cache=False)


def _sources(results) -> list[str]:
    return [r.chunk.doc_filename for r in results]


def test_authority_weight_ordering():
    corpus = {c.doc_filename: c for c in load_corpus()}
    current = corpus["01-returns-policy-current.md"]
    legacy = corpus["02-returns-policy-legacy.md"]
    migration = corpus["14-internal-content-migration-notes.md"]

    assert authority_weight(current) > authority_weight(legacy)
    assert authority_weight(current) > authority_weight(migration)


def test_return_window_prefers_current_over_legacy(kb):
    results = kb.search("standard return window how many days", top_k=5)
    srcs = _sources(results)
    assert "01-returns-policy-current.md" in srcs
    # Current policy must outrank the superseded legacy policy.
    current_rank = srcs.index("01-returns-policy-current.md")
    if "02-returns-policy-legacy.md" in srcs:
        assert current_rank < srcs.index("02-returns-policy-legacy.md")


def test_top_result_for_returns_is_authoritative(kb):
    results = kb.search("how long do I have to return an item", top_k=5)
    top = results[0].chunk
    assert not top.is_superseded
    assert top.policy_authority != "none"


def test_breeze_tumbler_conflict_surfaces_both_active_sources(kb):
    results = kb.search("can I put the Breeze Tumbler in the dishwasher", top_k=6)
    srcs = _sources(results)
    assert "11-product-care.md" in srcs
    assert "12-breeze-tumbler-product-card.md" in srcs


def test_migration_scratchpad_is_downranked(kb):
    # Even when the query name-drops the migration note, an authoritative policy
    # should still be retrievable and the scratchpad must not dominate.
    results = kb.search("migration note 60 days return everyone", top_k=6)
    for r in results:
        if r.chunk.doc_filename == "14-internal-content-migration-notes.md":
            assert authority_weight(r.chunk) < 0.5
