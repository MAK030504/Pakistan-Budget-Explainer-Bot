# Pakistan Budget Explainer

This tool answers a question about Pakistan's federal budget for 2025–26 from the budget PDFs on your machine. It finds the passage that best matches the question and cites the file, the page, and a section heading when the page has one. With no API key it quotes that passage. It does not fill in figures from outside the documents.

## Install

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Put the PDFs in `data/`

The books are not stored in git. Create a `data` directory in the repository root and save these four files under these exact names:

- `data/annual_budget_25_26.pdf` — Annual Budget Statement
- `data/budget_brief_25_26.pdf` — Budget in Brief
- `data/receipts_memorandum_25_26.pdf` — Explanatory Memorandum on Federal Receipts
- `data/finance_bill_25_26.pdf` — Finance Bill 2025 commentary (the memorandum this project originally shipped)

The Finance Division publishes the first three for 2025–26:

- Annual Budget Statement: https://www.finance.gov.pk/budget/budget_2025_26/abs_eng_10062025.pdf
- Budget in Brief: https://www.finance.gov.pk/budget/budget_2025_26/budget_in_brief_10062025.pdf
- Explanatory Memorandum on Federal Receipts: https://www.finance.gov.pk/budget/budget_2025_26/explanatory_memorandum_on_federal_receipts_10062025.pdf

Download each one and rename it to the filename above. The finance-bill commentary is not on that page. Place your copy at `data/finance_bill_25_26.pdf`. The tool still answers from whichever of the four files are present, and it says so when one is missing.

## Ask a question

From the repository root:

```bash
python -m pakistan_budget "What is the budget estimate for Defence Affairs and Services?"
```

The command prints the line it used, then a citation such as `annual_budget_25_26.pdf, SUMMARY OF EXPENDITURE, page 17`.

An optional language model can rewrite that line into a shorter reply. Copy `.env.example` to `.env` and set `BUDGET_LLM_API_KEY`. The call goes to an OpenAI-compatible chat endpoint (`BUDGET_LLM_BASE_URL` and `BUDGET_LLM_MODEL`). The model is told to use only the cited line. If the key is missing, or the call fails, the tool prints the line from the PDF instead.

## Run the eval

```bash
python -m pakistan_budget.eval
```

This asks the questions in `evals/questions.json` and prints the citation for each one. A question is marked ok when that citation is one of the expected documents. The PDFs have to be in `data/`, or the command stops and lists the missing files.

## Run the tests

```bash
pytest
```

The tests check chunking and retrieval with a small text fixture and a tiny PDF the loader test builds. They do not open the full budget books.

## Layout

`pakistan_budget/loader.py` reads the four PDFs and keeps each page in reading order. `chunker.py` splits a page into passages of about 200 words, which is the limit the original indexer used, and drops scraps of 50 characters or fewer. `retriever.py` ranks those passages with TF-IDF on this machine. `answer.py` returns the matching line and its citation. `eval.py` runs the question list.
