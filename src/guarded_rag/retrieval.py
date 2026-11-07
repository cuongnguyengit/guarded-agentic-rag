"""BM25 retrieval (no external index needed) + a query-coverage signal used for scope control."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from .kb import Chunk

STOP = set(
    "a an the of to in on for and or is are be by with at as it this that from your you i my me we our what "
    "how do does can should if when which who whom whose where why will would must may not no yes about into "
    "than then there their they them its was were been being have has had any all some".split()
)


def tokens(text: str) -> list[str]:
    toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP and len(t) > 1]
    return [t[:-1] if len(t) > 4 and t.endswith("s") else t for t in toks]   # crude plural folding


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    coverage: float   # IDF-weighted share of the query's terms present in the chunk (0..1)


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.docs = [tokens(c.title + " " + c.text) for c in chunks]
        self.tf = [Counter(d) for d in self.docs]
        self.avgdl = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def search(self, query: str, k: int = 4) -> list[Hit]:
        q = tokens(re.sub(r"<[A-Z]+_\d+>", " ", query))   # ignore PII placeholders
        qset = set(q)
        unseen = max(self.idf.values(), default=1.0)   # a term the KB never uses weighs the most
        q_weight = sum(self.idf.get(t, unseen) for t in qset)
        hits = []
        for c, tf, d in zip(self.chunks, self.tf, self.docs):
            s = sum(
                self.idf.get(t, 0.0) * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
                for t in q
                if t in tf
            )
            if s > 0:
                cov = sum(self.idf.get(t, unseen) for t in qset if t in tf) / q_weight if q_weight else 0.0
                hits.append(Hit(c, s, cov))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:k]
