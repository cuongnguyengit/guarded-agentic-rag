"""Deterministic answer validators (OWASP LLM09 Misinformation, LLM05 Improper Output Handling).

The writer's draft is accepted only if every factual line is cited, every citation points at evidence
that was actually retrieved, and every link / e-mail address in the answer appears in that evidence —
so a poisoned or hallucinated "send your password to X" cannot reach the user.
"""

from __future__ import annotations

import re

CITE = re.compile(r"\[c:([\w.-]+#\d+)\]")
CONTACT = re.compile(r"https?://\S+|\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
REFUSAL = re.compile(r"\b(could not find|not covered|don't know|do not know|cannot answer)\b", re.I)
MAX_CHARS = 1500


def citations(answer: str) -> list[str]:
    return list(dict.fromkeys(CITE.findall(answer)))


def validate(answer: str, evidence: dict[str, str]) -> list[str]:
    """`evidence` maps chunk id -> chunk text. Returns a list of human-readable issues (empty = valid)."""
    issues: list[str] = []
    text = answer.strip()
    if not text:
        return ["empty answer"]
    if len(text) > MAX_CHARS:
        issues.append(f"answer longer than {MAX_CHARS} characters")
    if re.search(r"</?\s*(untrusted|chunk)\b", text, re.I):
        issues.append("answer contains prompt fence tags")

    cited = citations(text)
    unknown = [c for c in cited if c not in evidence]
    if unknown:
        issues.append(f"citations not in retrieved evidence: {', '.join(unknown)}")

    if REFUSAL.search(text) and not cited:
        return issues   # an honest "not covered" answer needs no citations

    for line in text.splitlines():
        body = CITE.sub("", line).strip(" -*•\t")
        if len(re.findall(r"[A-Za-z]", body)) >= 15 and not body.endswith(":") and not CITE.search(line):
            issues.append(f"uncited statement: {body[:80]!r}")

    corpus = " ".join(evidence.get(c, "") for c in cited) if cited else " ".join(evidence.values())
    for contact in CONTACT.findall(text):
        if contact.rstrip(".,)") not in corpus:
            issues.append(f"link or address not found in cited evidence: {contact}")
    return issues
