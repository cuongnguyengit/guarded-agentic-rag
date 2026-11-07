"""Command line: `grag ask`, `grag eval`, `grag kb`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import Settings
from .eval import run_eval
from .graph import answer
from .kb import load_kb
from .llm import OpenAICompatLLM, ScriptedLLM


def _llm(args, s: Settings):
    return ScriptedLLM() if args.offline else OpenAICompatLLM(s.llm_base_url, s.llm_model, s.llm_api_key)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="grag", description="Guarded agentic RAG over a security handbook")
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("ask", help="answer one question")
    a.add_argument("question")
    a.add_argument("--trace", action="store_true", help="print the step-by-step trace")
    e = sub.add_parser("eval", help="run the evaluation set")
    e.add_argument("--judge", action="store_true", help="also score answers with an LLM judge")
    e.add_argument("--out", type=Path, help="write full results JSON here")
    sub.add_parser("kb", help="show indexed and quarantined chunks")
    for sp in (a, e):
        sp.add_argument("--offline", action="store_true", help="use the deterministic ScriptedLLM (no model needed)")
    args = p.parse_args(argv)

    s = Settings()
    kb = load_kb(s.kb_dir)
    if args.cmd == "kb":
        print(f"{len(kb.chunks)} chunks indexed, {len(kb.quarantined)} quarantined")
        for c, hits in kb.quarantined:
            print(f"  QUARANTINED {c.id} ({c.title}): {', '.join(hits)}")
        return 0

    llm = _llm(args, s)
    if args.cmd == "ask":
        r = answer(args.question, llm, kb, s, use_llm_judge=not args.offline)
        print(r.answer)
        print(f"\n[{r.status} · {r.llm_calls} LLM calls · {r.seconds}s · citations: {', '.join(r.citations) or '-'}]")
        if args.trace:
            for t in r.trace:
                print(json.dumps(t, ensure_ascii=False))
        return 0

    res = run_eval(llm, kb, s, use_llm_judge=not args.offline, judge_llm=llm if args.judge else None)
    print(json.dumps(res["summary"], indent=2))
    if args.out:
        args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
