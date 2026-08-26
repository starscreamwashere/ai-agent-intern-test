"""Ingestion tests: front matter, chunking, and metadata preservation."""

from aster_agent.ingestion import (
    load_corpus,
    load_document,
    parse_front_matter,
    split_into_sections,
)
from aster_agent.config import KNOWLEDGE_BASE_DIR


def test_parse_front_matter_extracts_yaml_and_body():
    raw = "---\ndocument_id: X\nstatus: active\n---\n# Title\n\nBody text.\n"
    meta, body = parse_front_matter(raw)
    assert meta["document_id"] == "X"
    assert meta["status"] == "active"
    assert "Body text." in body
    assert not body.startswith("---")


def test_parse_front_matter_missing_returns_empty_meta():
    raw = "# No front matter\n\nBody.\n"
    meta, body = parse_front_matter(raw)
    assert meta == {}
    assert body == raw


def test_split_into_sections_folds_h1_title():
    body = "# Doc Title\n\n## First\n\nAlpha.\n\n## Second\n\nBeta.\n"
    sections = split_into_sections(body)
    headings = [h for h, _ in sections]
    assert "Doc Title" not in headings  # H1 is treated as container
    assert headings == ["First", "Second"]
    assert sections[0][1] == "Alpha."


def test_load_corpus_loads_all_documents():
    chunks = load_corpus()
    filenames = {c.doc_filename for c in chunks}
    # All 14 supplied KB files should appear.
    assert len(filenames) == 14
    assert "01-returns-policy-current.md" in filenames


def test_metadata_flags_current_vs_legacy_vs_draft():
    by_file = {}
    for c in load_corpus():
        by_file.setdefault(c.doc_filename, c)

    current = by_file["01-returns-policy-current.md"]
    legacy = by_file["02-returns-policy-legacy.md"]
    migration = by_file["14-internal-content-migration-notes.md"]

    assert current.is_active_official is True
    assert current.is_superseded is False

    assert legacy.is_superseded is True
    assert legacy.is_active_official is False

    assert migration.status == "draft"
    assert migration.policy_authority == "none"


def test_citation_includes_filename_and_heading():
    chunk = load_document(KNOWLEDGE_BASE_DIR / "01-returns-policy-current.md")[0]
    assert chunk.doc_filename in chunk.citation
    assert chunk.heading in chunk.citation
