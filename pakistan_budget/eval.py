"""Run the checked-in budget questions and print the citation for each one."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from pakistan_budget.answer import answer_question
from pakistan_budget.chunker import chunk_pages
from pakistan_budget.__main__ import load_dotenv
from pakistan_budget.explain import warn_if_incomplete
from pakistan_budget.loader import load_documents
from pakistan_budget.retriever import TfidfRetriever

DEFAULT_QUESTIONS = Path(__file__).resolve().parents[1] / "evals" / "questions.json"


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Ask the eval questions and show which document and page were cited."
    )
    parser.add_argument("--data", default="data", help="Directory that holds the four budget PDFs.")
    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS),
        help="JSON list of questions (default: evals/questions.json).",
    )
    args = parser.parse_args(argv)
    questions_path = Path(args.questions)
    try:
        questions = json.loads(questions_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Question file not found: {questions_path}", file=sys.stderr)
        return 1
    try:
        warn_if_incomplete(args.data)
        pages = load_documents(args.data)
        retriever = TfidfRetriever(chunk_pages(pages))
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    misses = 0
    for item in questions:
        result = answer_question(item["question"], retriever)
        expected = item.get("expect_sources") or []
        sources = {citation.source for citation in result.citations}
        ok = (not expected) or any(source in sources for source in expected)
        if not ok:
            misses += 1
        citation = result.citations[0].label() if result.citations else "(none)"
        mark = "ok" if ok else "miss"
        print(f"[{mark}] {item['id']}")
        print(f"Question: {item['question']}")
        print(f"Citation: {citation}")
        if expected:
            print(f"Expected document: {', '.join(expected)}")
        preview = result.text.replace("\n", " ")
        if len(preview) > 360:
            preview = preview[:360].rstrip() + "..."
        print(f"Passage: {preview}")
        print()
    print(f"{len(questions) - misses}/{len(questions)} citations matched the expected document.")
    return 0 if misses == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
