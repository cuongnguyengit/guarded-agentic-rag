"""PII masking at the model boundary (OWASP LLM02 Sensitive Information Disclosure).

Personal data in the user's question is replaced by stable placeholders before any LLM call and
restored only in the final answer shown to the same user. The model never sees the raw values.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Order matters: more specific patterns first.
_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b")),
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("PHONE", re.compile(r"(?<!\w)\+?\d[\d ().-]{7,}\d\b")),
]


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits][::-1]
    total = sum(n if i % 2 == 0 else (n * 2 - 9 if n * 2 > 9 else n * 2) for i, n in enumerate(nums))
    return total % 10 == 0


@dataclass
class PiiMap:
    values: dict[str, str] = field(default_factory=dict)   # placeholder -> original

    def restore(self, text: str) -> str:
        for ph, orig in self.values.items():
            text = text.replace(ph, orig)
        return text


def mask(text: str) -> tuple[str, PiiMap]:
    pm = PiiMap()
    seen: dict[str, str] = {}
    counters: dict[str, int] = {}

    for kind, rx in _PATTERNS:
        def repl(m: re.Match, kind=kind) -> str:
            raw = m.group(0)
            if kind == "CARD" and not _luhn_ok(re.sub(r"\D", "", raw)):
                return raw
            if raw not in seen:
                counters[kind] = counters.get(kind, 0) + 1
                seen[raw] = f"<{kind}_{counters[kind]}>"
                pm.values[seen[raw]] = raw
            return seen[raw]

        text = rx.sub(repl, text)
    return text, pm
