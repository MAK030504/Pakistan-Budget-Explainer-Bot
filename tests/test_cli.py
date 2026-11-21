"""Command tests use a temporary directory, not the real budget PDFs."""

from pakistan_budget.__main__ import main
from pakistan_budget.eval import main as eval_main
from tests.test_loader import _write_fixture_pdf


def test_cli_prints_the_citation(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv("BUDGET_LLM_API_KEY", raising=False)
    _write_fixture_pdf(tmp_path / "budget_brief_25_26.pdf")
    code = main(["wheat subsidy Gilgit-Baltistan", "--data", str(tmp_path)])
    output = capsys.readouterr().out
    assert code == 0
    assert "Gilgit-Baltistan" in output
    assert "budget_brief_25_26.pdf" in output
    assert "page 1" in output


def test_cli_tells_you_where_the_pdfs_go(tmp_path, capsys):
    code = main(["defence", "--data", str(tmp_path)])
    error = capsys.readouterr().err
    assert code == 1
    assert "finance_bill_25_26.pdf" in error
    assert str(tmp_path) in error


def test_eval_without_pdfs_exits(tmp_path, capsys):
    code = eval_main(["--data", str(tmp_path)])
    error = capsys.readouterr().err
    assert code == 1
    assert "receipts_memorandum_25_26.pdf" in error
