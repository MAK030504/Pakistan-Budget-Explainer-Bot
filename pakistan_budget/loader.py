"""Read the federal budget PDFs from a local directory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

# Same filenames the original scripts indexed from the repository root.
BUDGET_FILES = (
    "annual_budget_25_26.pdf",
    "budget_brief_25_26.pdf",
    "finance_bill_25_26.pdf",
    "receipts_memorandum_25_26.pdf",
)

_HEADING = re.compile(
    r"^(table\s*[-–:]?\s*\d+\b.*|part\s*[-–]?\s*[ivx0-9]+\b.*|chapter\s+\w+.*)$",
    re.IGNORECASE,
)
_ROW_TOLERANCE = 2.5


@dataclass(frozen=True)
class Page:
    source: str
    page: int
    text: str
    section: str | None = None


def budget_dir(path: str | Path = "data") -> Path:
    return Path(path)


def missing_budget_files(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    return [name for name in BUDGET_FILES if not (root / name).is_file()]


def missing_message(data_dir: str | Path, missing: list[str] | None = None) -> str:
    root = Path(data_dir)
    names = missing if missing is not None else list(BUDGET_FILES)
    listed = "\n".join(f"  {root / name}" for name in names)
    return (
        "No budget PDFs were found to search.\n"
        f"Place the 2025–26 files in {root} using these names:\n"
        f"{listed}"
    )


def load_documents(data_dir: str | Path = "data") -> list[Page]:
    """Load whichever of the four budget PDFs are present.

    Raises FileNotFoundError when the directory has none of them.
    """
    root = Path(data_dir)
    missing = missing_budget_files(root)
    if len(missing) == len(BUDGET_FILES):
        raise FileNotFoundError(missing_message(root, missing))

    pages: list[Page] = []
    for name in BUDGET_FILES:
        path = root / name
        if path.is_file():
            pages.extend(load_pdf(path))
    if not pages:
        raise FileNotFoundError(
            f"The PDFs in {root} did not contain any extractable text."
        )
    return pages


def load_pdf(path: str | Path) -> list[Page]:
    import pymupdf

    pdf_path = Path(path)
    document = pymupdf.open(pdf_path)
    pages: list[Page] = []
    try:
        for index, pdf_page in enumerate(document):
            text = reading_order_text(pdf_page)
            if not text.strip():
                continue
            pages.append(
                Page(
                    source=pdf_path.name,
                    page=index + 1,
                    text=text,
                    section=section_from_text(text),
                )
            )
    finally:
        document.close()
    return pages


def reading_order_text(pdf_page) -> str:
    """Join each visual row from left to right.

    These budget books are tables. PyMuPDF's plain text often reads a whole
    column before the next, which separates a label from the figure on its row.
    """
    raw = pdf_page.get_text("dict")
    items: list[tuple[float, float, str]] = []
    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            text = "".join(span.get("text", "") for span in line.get("spans", []))
            text = text.strip()
            if not text:
                continue
            x0, y0, _, _ = line["bbox"]
            items.append((y0, x0, text))
    items.sort(key=lambda item: (item[0], item[1]))

    rows: list[list[str]] = []
    row_y: float | None = None
    for y0, _, text in items:
        if row_y is None or abs(y0 - row_y) > _ROW_TOLERANCE:
            rows.append([text])
            row_y = y0
        else:
            rows[-1].append(text)
    return "\n".join(" ".join(row) for row in rows if any(part.strip() for part in row))


def section_from_text(text: str) -> str | None:
    """Use a table or chapter heading from the top of the page, when one is present."""
    lines = [" ".join(line.split()) for line in text.splitlines()]
    lines = [line for line in lines if line]
    for line in lines[:20]:
        if line.isdigit():
            continue
        if _HEADING.match(line):
            return line[:160]
        letters = [char for char in line if char.isalpha()]
        if not letters:
            continue
        upper_ratio = sum(char.isupper() for char in letters) / len(letters)
        if 12 <= len(line) <= 100 and upper_ratio >= 0.75 and not re.search(r"\d{3,}", line):
            return line[:160]
    return None
