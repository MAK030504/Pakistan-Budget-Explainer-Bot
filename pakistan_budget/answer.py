"""Turn the best matching passage into an answer with a citation.

The default path quotes the retrieved text. It does not add facts. When
BUDGET_LLM_API_KEY is set, a language model may rewrite that same passage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pakistan_budget.retriever import TfidfRetriever

_DEFAULT_BASE_URL = "https://api.together.xyz/v1"
_DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


@dataclass(frozen=True)
class Citation:
    source: str
    page: int
    section: str | None = None

    def label(self) -> str:
        parts = [self.source]
        if self.section:
            parts.append(self.section)
        parts.append(f"page {self.page}")
        return ", ".join(parts)


@dataclass(frozen=True)
class Answer:
    text: str
    citations: tuple[Citation, ...]
    used_llm: bool = False


def answer_question(
    question: str,
    retriever: TfidfRetriever,
    *,
    k: int = 1,
) -> Answer:
    hits = retriever.search(question, k=k)
    if not hits:
        return Answer(
            text="No passage in the loaded budget documents matched this question.",
            citations=(),
        )
    hit = hits[0]
    citation = Citation(source=hit.chunk.source, page=hit.chunk.page, section=hit.chunk.section)
    quoted = retriever.best_line(hit.chunk.text, question)
    api_key = os.environ.get("BUDGET_LLM_API_KEY", "").strip()
    if not api_key:
        return Answer(text=quoted, citations=(citation,))
    rewritten = rewrite_passages(question, quoted, citation, api_key)
    if not rewritten:
        return Answer(text=quoted, citations=(citation,))
    return Answer(text=rewritten, citations=(citation,), used_llm=True)


def rewrite_passages(
    question: str,
    passage: str,
    citation: Citation,
    api_key: str,
    opener=urlopen,
) -> str | None:
    """Ask an OpenAI-compatible chat endpoint to rephrase the cited passage.

    Returns None when the call fails so the caller can keep the quoted passage.
    """
    base_url = os.environ.get("BUDGET_LLM_BASE_URL", _DEFAULT_BASE_URL).strip() or _DEFAULT_BASE_URL
    model = os.environ.get("BUDGET_LLM_MODEL", _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    passage = passage.strip()
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 400,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You rewrite one cited passage from Pakistan's federal budget documents. "
                    "Use only facts written in the passage. Do not add figures, names, rates, "
                    "or years that are not in the passage. If the passage does not answer the "
                    "question, say that the cited text does not answer it. Keep the reply short."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question: {question.strip()}\n\n"
                    f"Cited passage ({citation.label()}):\n{passage}"
                ),
            },
        ],
    }
    request = Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with opener(request, timeout=30) as response:
            body = json.loads(response.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
    except (HTTPError, URLError, KeyError, IndexError, TypeError, json.JSONDecodeError, TimeoutError, OSError):
        return None
    cleaned = str(text).strip()
    return cleaned or None

