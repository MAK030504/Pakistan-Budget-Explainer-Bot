"""The answer stays inside the retrieved passage unless a model key is set."""

import json

from pakistan_budget.answer import Citation, answer_question, rewrite_passages
from pakistan_budget.chunker import Chunk
from pakistan_budget.retriever import TfidfRetriever


def _retriever() -> TfidfRetriever:
    return TfidfRetriever(
        [
            Chunk(
                text="Sales tax exemption available on import or supply of solar panels withdrawn.",
                source="finance_bill_25_26.pdf",
                page=9,
                section="SALES TAX",
            )
        ]
    )


def test_without_a_key_the_answer_is_the_passage(monkeypatch):
    monkeypatch.delenv("BUDGET_LLM_API_KEY", raising=False)
    result = answer_question("solar panels sales tax exemption", _retriever())
    assert result.used_llm is False
    assert result.text == "Sales tax exemption available on import or supply of solar panels withdrawn."
    assert result.citations == (Citation("finance_bill_25_26.pdf", 9, "SALES TAX"),)


def test_failed_rewrite_falls_back_to_the_passage(monkeypatch):
    monkeypatch.setenv("BUDGET_LLM_API_KEY", "test-key")

    def broken(*args, **kwargs):
        return None

    monkeypatch.setattr("pakistan_budget.answer.rewrite_passages", broken)
    result = answer_question("solar panels", _retriever())
    assert result.used_llm is False
    assert "solar panels" in result.text


def test_rewrite_replaces_the_passage_when_the_call_succeeds(monkeypatch):
    monkeypatch.setenv("BUDGET_LLM_API_KEY", "test-key")

    def fake(question, passage, citation, api_key, opener=None):
        assert api_key == "test-key"
        assert "solar panels" in passage
        return "The cited passage says the solar-panel exemption is withdrawn."

    monkeypatch.setattr("pakistan_budget.answer.rewrite_passages", fake)
    result = answer_question("solar panels", _retriever())
    assert result.used_llm is True
    assert result.text.startswith("The cited passage")
    assert result.citations[0].page == 9


class _Response:
    def __init__(self, payload):
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_rewrite_request_forbids_new_facts_and_quotes_the_passage():
    captured = {}

    def opener(request, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        assert request.get_header("Authorization") == "Bearer secret"
        return _Response({"choices": [{"message": {"content": "  Rewritten.  "}}]})

    citation = Citation("finance_bill_25_26.pdf", 9, "SALES TAX")
    text = rewrite_passages("solar panels?", "solar panels withdrawn", citation, "secret", opener=opener)
    assert text == "Rewritten."
    assert captured["url"].endswith("/chat/completions")
    system = captured["body"]["messages"][0]["content"]
    user = captured["body"]["messages"][1]["content"]
    assert "Do not add" in system
    assert "solar panels withdrawn" in user
    assert "finance_bill_25_26.pdf" in user
    assert "page 9" in user


def test_rewrite_http_failure_returns_none():
    from urllib.error import HTTPError

    def opener(request, timeout):
        raise HTTPError(request.full_url, 401, "unauthorized", hdrs=None, fp=None)

    text = rewrite_passages("q", "passage", Citation("a.pdf", 1, None), "secret", opener=opener)
    assert text is None
