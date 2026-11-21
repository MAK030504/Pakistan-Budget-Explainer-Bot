"""Retrieval tests use the small text fixture, not the federal budget PDFs."""

import json
from pathlib import Path

from pakistan_budget.answer import answer_question
from pakistan_budget.chunker import Chunk
from pakistan_budget.retriever import TfidfRetriever

FIXTURE = Path(__file__).parent / "fixtures" / "passages.json"


def _retriever() -> TfidfRetriever:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    chunks = [Chunk(**item) for item in raw]
    return TfidfRetriever(chunks)


def test_solar_question_cites_the_finance_bill_page():
    result = answer_question(
        "Is the sales tax exemption on solar panels withdrawn?",
        _retriever(),
    )
    assert result.citations[0].source == "finance_bill_25_26.pdf"
    assert result.citations[0].page == 9
    assert result.citations[0].section == "SALES TAX"
    assert "solar panels" in result.text.lower()
    assert result.used_llm is False


def test_wheat_question_cites_the_budget_brief():
    result = answer_question("What wheat subsidy goes to Gilgit-Baltistan?", _retriever())
    assert result.citations[0].source == "budget_brief_25_26.pdf"
    assert result.citations[0].page == 27
    assert "gilgit-baltistan" in result.text.lower()


def test_defence_question_does_not_return_the_wheat_page():
    result = answer_question("What is Defence Affairs and Services?", _retriever())
    assert result.citations[0].source == "annual_budget_25_26.pdf"
    assert result.citations[0].page == 18


def test_answer_text_is_copied_from_the_retrieved_passage():
    retriever = _retriever()
    question = "Is the sales tax exemption on solar panels withdrawn?"
    result = answer_question(question, retriever)
    passage = next(chunk.text for chunk in retriever.chunks if chunk.page == 9)
    for line in result.text.splitlines():
        assert line in passage


def test_unknown_question_has_no_citation():
    result = answer_question("quantum xylophone submarine", _retriever())
    assert result.citations == ()
    assert "No passage" in result.text


def test_empty_index_returns_nothing():
    assert TfidfRetriever([]).search("defence") == []
