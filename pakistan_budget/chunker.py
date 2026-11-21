"""Split each budget page into passages short enough to cite."""

from __future__ import annotations

from dataclasses import dataclass

from pakistan_budget.loader import Page

# The original indexer kept about 200 words and dropped scraps of 50 characters or fewer.
DEFAULT_MAX_WORDS = 200
DEFAULT_MIN_CHARS = 50


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int
    section: str | None = None


def chunk_text(
    text: str,
    max_words: int = DEFAULT_MAX_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[str]:
    """Pack lines into chunks of at most `max_words` words.

    A single line longer than `max_words` is split on word boundaries, which is
    how the original script broke up a long page. Shorter lines stay intact so
    a table row is not cut in half. A chunk of `min_chars` characters or fewer
    is dropped.
    """
    if max_words < 1:
        raise ValueError("max_words must be at least 1")
    pieces = _lines_within_limit(text, max_words)
    chunks: list[str] = []
    current: list[str] = []
    current_words = 0
    for piece in pieces:
        word_count = len(piece.split())
        if current and current_words + word_count > max_words:
            _store(chunks, current, min_chars)
            current = []
            current_words = 0
        current.append(piece)
        current_words += word_count
    if current:
        _store(chunks, current, min_chars)
    return chunks


def chunk_pages(
    pages: list[Page],
    max_words: int = DEFAULT_MAX_WORDS,
    min_chars: int = DEFAULT_MIN_CHARS,
) -> list[Chunk]:
    """Chunk every page on its own so a citation keeps the right page number."""
    chunks: list[Chunk] = []
    for page in pages:
        for text in chunk_text(page.text, max_words=max_words, min_chars=min_chars):
            chunks.append(
                Chunk(
                    text=text,
                    source=page.source,
                    page=page.page,
                    section=page.section,
                )
            )
    return chunks


def _lines_within_limit(text: str, max_words: int) -> list[str]:
    pieces: list[str] = []
    for raw_line in text.splitlines():
        words = raw_line.split()
        if not words:
            continue
        if len(words) <= max_words:
            pieces.append(" ".join(words))
            continue
        for start in range(0, len(words), max_words):
            pieces.append(" ".join(words[start : start + max_words]))
    return pieces


def _store(chunks: list[str], lines: list[str], min_chars: int) -> None:
    piece = "\n".join(lines).strip()
    if len(piece) > min_chars:
        chunks.append(piece)
