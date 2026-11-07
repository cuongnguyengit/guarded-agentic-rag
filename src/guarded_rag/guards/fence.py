"""Fencing of untrusted content.

Everything that did not come from the developer (user text, retrieved chunks, chat history) is wrapped
in tags before it is placed in a prompt, and any tag-like text inside it is neutralised so the content
cannot close the fence and "escape" into instructions.
"""

from __future__ import annotations

import re

_TAG = re.compile(r"</?\s*(untrusted|chunk)\b[^>]*>", re.I)


def neutralise(text: str) -> str:
    return _TAG.sub(lambda m: m.group(0).replace("<", "‹").replace(">", "›"), text)


def fence(text: str, source: str) -> str:
    return f'<untrusted source="{source}">\n{neutralise(text)}\n</untrusted>'


def fence_chunk(chunk_id: str, title: str, text: str) -> str:
    return f'<chunk id="{chunk_id}" title="{neutralise(title)}">\n{neutralise(text)}\n</chunk>'
