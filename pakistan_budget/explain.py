"""Load the local PDFs, chunk them, and answer one question."""

from __future__ import annotations

from pathlib import Path
import sys

from pakistan_budget.answer import Answer, answer_question
from pakistan_budget.chunker import chunk_pages
from pakistan_budget.loader import BUDGET_FILES, load_documents, missing_budget_files, missing_message
from pakistan_budget.retriever import TfidfRetriever


def explain(question: str, data_dir: str | Path = "data") -> Answer:
    root = Path(data_dir)
    missing = missing_budget_files(root)
    if len(missing) == len(BUDGET_FILES):
        raise FileNotFoundError(missing_message(root, missing))
    pages = load_documents(root)
    chunks = chunk_pages(pages)
    if not chunks:
        raise FileNotFoundError(f"The PDFs in {root} did not contain any text to search.")
    return answer_question(question, TfidfRetriever(chunks))


def warn_if_incomplete(data_dir: str | Path) -> None:
    missing = missing_budget_files(data_dir)
    if not missing or len(missing) == len(BUDGET_FILES):
        return
    names = ", ".join(missing)
    print(f"Searching without: {names}", file=sys.stderr)
