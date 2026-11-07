"""Runtime settings, read from environment variables (any OpenAI-compatible endpoint works)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    llm_base_url: str = field(default_factory=lambda: os.environ.get("GRAG_LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_model: str = field(default_factory=lambda: os.environ.get("GRAG_LLM_MODEL", "qwen3:8b"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("GRAG_LLM_API_KEY", "unused"))
    kb_dir: Path = field(default_factory=lambda: Path(os.environ.get("GRAG_KB_DIR", ROOT / "data" / "kb")))
    top_k: int = 4                 # chunks per sub-question
    max_sub_questions: int = 3     # planner fan-out cap
    max_repairs: int = 2           # writer self-repair attempts before extractive fallback
    llm_call_budget: int = 12      # hard cap on LLM calls per question
    deadline_s: float = 90.0       # wall-clock budget per question
    scope_min_coverage: float = 0.25  # cheap lexical gate; the writer makes the final in/out-of-scope call
