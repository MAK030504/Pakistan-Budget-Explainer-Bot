"""Rank budget passages with TF-IDF. No model download and no API key."""

from __future__ import annotations

from dataclasses import dataclass
import re

from sklearn.feature_extraction import text as sklearn_text

from pakistan_budget.chunker import Chunk

# Words that show up in almost every budget table, or in the question rather than
# the books. Leaving "estimate" in the index lets that one rare word outrank the
# heading the person actually asked about.
_EXTRA_STOPWORDS = {
    "budget",
    "budgets",
    "estimate",
    "estimates",
    "estimated",
    "revised",
    "classification",
    "million",
    "billion",
    "rupee",
    "rupees",
    "pkr",
    "table",
    "contd",
    "continued",
    "allocation",
    "allocated",
    "budgeted",
    "target",
    "listed",
    "total",
    "amount",
}
_YEAR = re.compile(r"\b20\d{2}\b")
_NON_WORD = re.compile(r"[^a-z0-9\s]+")


def prepare_text(text: str) -> str:
    """Lowercase and drop years and punctuation before TF-IDF sees the text."""
    cleaned = text.lower().replace("&", " and ")
    cleaned = _YEAR.sub(" ", cleaned)
    return _NON_WORD.sub(" ", cleaned)


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class TfidfRetriever:
    """Binary TF-IDF over the chunks.

    Each shared term contributes its IDF once. Cosine length-normalization
    preferred short contents lines over the table row that also holds the
    figure, so this retriever does not divide by passage length.
    """

    def __init__(self, chunks: list[Chunk]):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.chunks = list(chunks)
        stopwords = list(sklearn_text.ENGLISH_STOP_WORDS.union(_EXTRA_STOPWORDS))
        self.vectorizer = TfidfVectorizer(
            lowercase=False,
            preprocessor=prepare_text,
            stop_words=stopwords,
            ngram_range=(1, 2),
            binary=True,
            norm=None,
            use_idf=True,
            smooth_idf=True,
        )
        if self.chunks:
            self._matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)
            self._idf = {
                term: float(weight)
                for term, weight in zip(self.vectorizer.get_feature_names_out(), self.vectorizer.idf_)
            }
        else:
            self._matrix = None
            self._idf = {}

    def search(self, query: str, k: int = 3) -> list[Hit]:
        if k < 1 or not self.chunks or self._matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        if query_vector.nnz == 0:
            return []
        scores = (self._matrix @ query_vector.T).toarray().ravel()
        order = sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))
        hits: list[Hit] = []
        for index in order[:k]:
            score = float(scores[index])
            if score <= 0:
                break
            hits.append(Hit(chunk=self.chunks[index], score=score))
        return hits

    def best_line(self, text: str, query: str) -> str:
        """Pick the line in `text` that shares the most IDF with the question."""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return text.strip()
        analyzer = self.vectorizer.build_analyzer()
        wanted = set(analyzer(query))
        if not wanted:
            return "\n".join(lines[:6])

        line_terms = [set(analyzer(line)) for line in lines]
        unigrams = sorted(
            (term for term in wanted if " " not in term),
            key=lambda term: self._idf.get(term, 0.0),
            reverse=True,
        )
        present = set().union(*line_terms) if line_terms else set()
        anchor = next((term for term in unigrams if term in present), None)

        def line_score(found: set[str]) -> float:
            if anchor and anchor not in found:
                return -1.0
            return sum(self._idf.get(term, 0.0) for term in wanted & found)

        best = max(range(len(lines)), key=lambda index: (line_score(line_terms[index]), -index))
        if line_score(line_terms[best]) <= 0:
            return "\n".join(lines[:6])
        chosen = lines[best]
        # A short label often sits above its figures. Keep the next line so the
        # number stays with the heading, without pulling in the previous row.
        if len(chosen) < 40 and best + 1 < len(lines):
            chosen = f"{chosen}\n{lines[best + 1]}"
        return chosen
