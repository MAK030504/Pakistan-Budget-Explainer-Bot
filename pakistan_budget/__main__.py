"""Ask one question from the command line."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from pakistan_budget.explain import explain, warn_if_incomplete


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Answer a question about Pakistan's 2025–26 federal budget and cite the page."
    )
    parser.add_argument(
        "question",
        nargs="+",
        help='Question to ask. Example: python -m pakistan_budget "What is the defence allocation?"',
    )
    parser.add_argument(
        "--data",
        default="data",
        help="Directory that holds the four budget PDFs (default: data).",
    )
    args = parser.parse_args(argv)
    question = " ".join(args.question).strip()
    if not question:
        parser.error("Pass a question.")
    try:
        warn_if_incomplete(args.data)
        result = explain(question, args.data)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(result.text)
    print()
    if result.citations:
        print("Cited from:")
        for citation in result.citations:
            print(f"- {citation.label()}")
    else:
        print("Cited from: (none)")
    if result.used_llm:
        print()
        print("The reply above is a rewrite of the cited passage.")
    return 0


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=value lines into the environment without overriding real variables."""
    env_path = Path(path)
    if not env_path.is_file():
        return
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


if __name__ == "__main__":
    raise SystemExit(main())
