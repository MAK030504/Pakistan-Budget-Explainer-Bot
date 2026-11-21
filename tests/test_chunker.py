"""Chunking tests use plain text, not the federal budget PDFs."""

from pakistan_budget.chunker import chunk_pages, chunk_text
from pakistan_budget.loader import Page


def test_short_fragment_is_dropped():
    assert chunk_text("too short") == []
    assert chunk_text("   ") == []


def test_keeps_a_passage_over_fifty_characters():
    text = "Wheat subsidy for Gilgit-Baltistan is recorded in the subsidies table."
    assert chunk_text(text) == [text]


def test_long_line_splits_on_word_boundaries():
    words = [f"word{i}" for i in range(250)]
    chunks = chunk_text(" ".join(words), max_words=200)
    assert len(chunks) == 2
    assert chunks[0].split() == words[:200]
    assert chunks[1].split() == words[200:]


def test_packs_whole_lines_until_the_word_limit():
    line = " ".join(["alpha"] * 20)
    chunks = chunk_text("\n".join([line, line, line]), max_words=50)
    assert len(chunks) == 2
    assert chunks[0].splitlines() == [line, line]
    assert chunks[1].splitlines() == [line]


def test_chunks_do_not_cross_pages():
    long_page = " ".join(["gamma"] * 250)
    pages = [
        Page(source="budget_brief_25_26.pdf", page=4, text=long_page, section="TABLE - 3"),
        Page(
            source="budget_brief_25_26.pdf",
            page=5,
            text="Wheat subsidy to Gilgit-Baltistan is twenty thousand rupees this year.",
            section=None,
        ),
    ]
    chunks = chunk_pages(pages, max_words=200)
    assert [chunk.page for chunk in chunks] == [4, 4, 5]
    assert all(chunk.source == "budget_brief_25_26.pdf" for chunk in chunks)
    assert chunks[0].section == "TABLE - 3"
    assert chunks[-1].section is None
    assert "Wheat subsidy" in chunks[-1].text
