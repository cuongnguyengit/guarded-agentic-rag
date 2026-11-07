"""Knowledge base: markdown files -> section-level chunks, screened at ingestion time."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .guards.injection import scan_text


@dataclass(frozen=True)
class Chunk:
    id: str          # "<doc>#<n>", used in citations as [c:<id>]
    doc: str
    title: str       # "Doc title > Section"
    text: str


@dataclass
class KnowledgeBase:
    chunks: list[Chunk]
    quarantined: list[tuple[Chunk, list[str]]] = field(default_factory=list)

    def by_id(self) -> dict[str, Chunk]:
        return {c.id: c for c in self.chunks}


def chunk_markdown(doc: str, text: str) -> list[Chunk]:
    title = (re.search(r"^#\s+(.+)$", text, re.M) or [None, doc])[1].strip()
    out = []
    for n, sec in enumerate(re.split(r"^##\s+", text, flags=re.M)[1:], 1):
        head, _, body = sec.partition("\n")
        body = body.strip()
        if body:
            out.append(Chunk(id=f"{doc}#{n}", doc=doc, title=f"{title} > {head.strip()}", text=body))
    return out


def load_kb(kb_dir: Path) -> KnowledgeBase:
    """Load every *.md file. Sections that look like instructions aimed at the model (indirect prompt
    injection / KB poisoning, OWASP LLM01 + LLM08) are quarantined and never reach retrieval."""
    kb = KnowledgeBase(chunks=[])
    for f in sorted(Path(kb_dir).glob("*.md")):
        for c in chunk_markdown(f.stem, f.read_text(encoding="utf-8")):
            hits = scan_text(c.text)
            if hits:
                kb.quarantined.append((c, hits))
            else:
                kb.chunks.append(c)
    return kb
