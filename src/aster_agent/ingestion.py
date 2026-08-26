"""Load knowledge-base Markdown files into retrievable chunks.

Each document carries YAML front matter whose metadata (status, policy_authority,
supersedes/superseded_by) drives *document precedence* later. We preserve that
metadata on every chunk so the retriever can prefer authoritative, active
documents over superseded, draft, or non-policy ones.

Chunking is heading-based: each Markdown section (a heading and its body) becomes
one chunk. Sections are small and self-contained in this corpus, so this keeps
each chunk focused on a single topic while retaining a citable heading.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .config import KNOWLEDGE_BASE_DIR

FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


@dataclass
class Chunk:
    """A retrievable passage plus the metadata needed for citation + precedence."""

    doc_filename: str
    doc_id: str
    doc_title: str
    heading: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def status(self) -> str:
        return str(self.metadata.get("status", "unknown")).lower()

    @property
    def policy_authority(self) -> str:
        return str(self.metadata.get("policy_authority", "unknown")).lower()

    @property
    def is_superseded(self) -> bool:
        return self.status == "superseded" or bool(self.metadata.get("superseded_by"))

    @property
    def is_active_official(self) -> bool:
        return self.status == "active" and self.policy_authority == "official"

    @property
    def citation(self) -> str:
        """A human-readable source reference: filename + heading."""
        if self.heading:
            return f"{self.doc_filename} — {self.heading}"
        return self.doc_filename

    @property
    def embedding_text(self) -> str:
        """Text handed to the embedder: title + heading give topical context."""
        return f"{self.doc_title}\n{self.heading}\n\n{self.text}".strip()


def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    """Split a Markdown file into (front_matter_dict, body)."""
    match = FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    body = raw[match.end():]
    return meta, body


def split_into_sections(body: str) -> list[tuple[str, str]]:
    """Split a Markdown body into (heading, section_text) pairs.

    The top-level document title (# Heading) is treated as a container: its own
    body (usually empty) is folded into the first subsection so we don't emit an
    empty chunk. Content before any heading becomes a section with an empty
    heading.
    """
    lines = body.splitlines()
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []
    seen_content_heading = False

    def flush() -> None:
        text = "\n".join(current_lines).strip()
        if text:
            sections.append((current_heading, text))

    for line in lines:
        m = HEADING_RE.match(line)
        if m:
            level = len(m.group(1))
            title = m.group(2).strip()
            # Treat the first H1 as the doc title container, not a section.
            if level == 1 and not seen_content_heading and not current_lines:
                current_heading = ""
                continue
            flush()
            current_heading = title
            current_lines = []
            seen_content_heading = True
        else:
            current_lines.append(line)
    flush()
    return sections


def load_document(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_front_matter(raw)
    doc_id = str(meta.get("document_id", path.stem))
    doc_title = str(meta.get("title", path.stem))

    chunks: list[Chunk] = []
    for heading, text in split_into_sections(body):
        chunks.append(
            Chunk(
                doc_filename=path.name,
                doc_id=doc_id,
                doc_title=doc_title,
                heading=heading,
                text=text,
                metadata=dict(meta),
            )
        )
    # A document with no sectioned content still deserves one chunk.
    if not chunks and body.strip():
        chunks.append(
            Chunk(
                doc_filename=path.name,
                doc_id=doc_id,
                doc_title=doc_title,
                heading="",
                text=body.strip(),
                metadata=dict(meta),
            )
        )
    return chunks


def load_corpus(kb_dir: Path | None = None) -> list[Chunk]:
    """Load and chunk every Markdown file in the knowledge base, sorted by name."""
    kb_dir = kb_dir or KNOWLEDGE_BASE_DIR
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        chunks.extend(load_document(path))
    return chunks
