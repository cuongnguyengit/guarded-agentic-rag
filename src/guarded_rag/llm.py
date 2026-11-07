"""LLM adapters.

`OpenAICompatLLM` talks to any OpenAI-compatible server (Ollama, vLLM, LiteLLM, OpenAI).
`ScriptedLLM` is a deterministic stand-in used by tests and the offline demo: it reads the
`TASK:` tag that every prompt in this project starts with and returns a plausible, well-formed
answer, so the whole graph (planning, fan-out, validation, repair) runs without a model.
"""

from __future__ import annotations

import json
import re
from typing import Protocol

THINK = re.compile(r"<think>.*?</think>", re.S)


class LLM(Protocol):
    calls: int

    def complete(self, system: str, user: str) -> str: ...


def extract_json(text: str) -> dict:
    """Parse the first JSON object in a model reply (tolerates code fences and thinking blocks)."""
    text = THINK.sub("", text)
    start = text.find("{")
    if start < 0:
        raise ValueError(f"no JSON object in reply: {text[:120]!r}")
    depth = 0
    for i, ch in enumerate(text[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            return json.loads(text[start : i + 1])
    raise ValueError("unterminated JSON object")


class OpenAICompatLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "unused", timeout: float = 120.0):
        from openai import OpenAI

        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        # "/no_think" disables the reasoning trace on Qwen3-style models; other models ignore it.
        r = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user + "\n/no_think"}],
        )
        return THINK.sub("", r.choices[0].message.content or "").strip()


class ScriptedLLM:
    """Deterministic fake. Behaviour per task:

    plan    -> split the question on ' and ' / '?' into sub-questions
    notes   -> first sentence of each fenced chunk, cited
    write   -> one bullet per note (optionally sabotaged to exercise the repair loop)
    judge_injection -> never flags (heuristics decide in offline mode)
    judge   -> perfect scores
    """

    def __init__(self, sabotage_first_write: bool = False):
        self.calls = 0
        self.sabotage_first_write = sabotage_first_write
        self._writes = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        task = re.search(r"TASK:\s*(\w+)", system)
        task = task.group(1) if task else ""
        if task == "plan":
            q = re.search(r"QUESTION:\s*(.+)", user, re.S).group(1)
            q = re.sub(r"</?untrusted[^>]*>", "", q).strip()
            parts = [p.strip(" ?.") for p in re.split(r"\?|\band\b", q) if len(p.strip(" ?.")) > 3]
            return json.dumps({"sub_questions": parts[:3] or [q]})
        if task == "notes":
            notes = []
            for cid, body in re.findall(r'<chunk id="([^"]+)"[^>]*>\s*(.*?)\s*</chunk>', user, re.S):
                first = re.split(r"(?<=[.!?])\s", body.strip().split("\n")[-1])[0]
                notes.append({"text": first, "cite": cid})
            return json.dumps({"notes": notes[:2]})
        if task == "write":
            self._writes += 1
            notes = re.findall(r"- (.+?) \[c:([^\]]+)\]", user)
            if self.sabotage_first_write and self._writes == 1:
                return "Here is what I found.\n" + "\n".join(f"- {t}" for t, _ in notes)  # no citations
            return "\n".join(f"- {t} [c:{c}]" for t, c in notes) or "I could not find this in the handbook."
        if task == "judge_injection":
            return json.dumps({"injection": False, "reason": "scripted"})
        if task == "judge":
            return json.dumps({"groundedness": 5, "helpfulness": 5, "reason": "scripted"})
        return "{}"
