"""Loader tests build a tiny PDF instead of opening the federal budget books."""

from pathlib import Path

import pymupdf

from pakistan_budget.chunker import chunk_pages
from pakistan_budget.loader import (
    BUDGET_FILES,
    load_documents,
    load_pdf,
    missing_message,
    section_from_text,
)
from pakistan_budget.retriever import TfidfRetriever
from pakistan_budget.answer import answer_question


def _write_fixture_pdf(path: Path) -> None:
    document = pymupdf.open()
    first = document.new_page()
    first.insert_text((72, 72), "TABLE - 12", fontsize=14)
    first.insert_text((72, 110), "Wheat subsidy for Gilgit-Baltistan is listed under food subsidies.", fontsize=11)
    second = document.new_page()
    second.insert_text((72, 72), "TABLE - 10", fontsize=14)
    second.insert_text((72, 110), "Defence Affairs and Services budget estimate is 2550 million rupees in this fixture.", fontsize=11)
    document.save(path)
    document.close()


def test_section_heading_is_read_from_plain_text():
    text = "TABLE - 10\nCURRENT EXPENDITURE\nPension 1055000"
    assert section_from_text(text) == "TABLE - 10"


def test_tiny_pdf_keeps_page_numbers_and_answers_from_them(tmp_path):
    pdf_path = tmp_path / "budget_brief_25_26.pdf"
    _write_fixture_pdf(pdf_path)

    pages = load_pdf(pdf_path)
    assert [page.page for page in pages] == [1, 2]
    assert pages[0].section == "TABLE - 12"
    assert "Gilgit-Baltistan" in pages[0].text
    assert pages[1].section == "TABLE - 10"

    result = answer_question(
        "What is the wheat subsidy for Gilgit-Baltistan?",
        TfidfRetriever(chunk_pages(pages)),
    )
    assert result.citations[0].source == "budget_brief_25_26.pdf"
    assert result.citations[0].page == 1
    assert "Gilgit-Baltistan" in result.text

    defence = answer_question(
        "Defence Affairs and Services",
        TfidfRetriever(chunk_pages(pages)),
    )
    assert defence.citations[0].page == 2


def test_missing_pdfs_message_names_the_four_files(tmp_path):
    message = missing_message(tmp_path)
    for name in BUDGET_FILES:
        assert name in message
    try:
        load_documents(tmp_path)
    except FileNotFoundError as exc:
        assert "annual_budget_25_26.pdf" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
