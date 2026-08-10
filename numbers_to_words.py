#!/usr/bin/env python3
"""Convert numbers in text to words with num2words.

Rules (matching the requested behavior):
  - Currency like $10,000  -> "Ten Thousand Dollars"
  - Bare digits like 1234567 -> "one two three four five six seven"

Requires:
  pip install num2words

Usage:
  python3 numbers_to_words.py "Send me $10,000 or call 1234567"
  python3 numbers_to_words.py -f input.txt -o output.txt
  echo "Balance is $1,250.50" | python3 numbers_to_words.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    from num2words import num2words
except ImportError as e:
    raise SystemExit("Install num2words: pip install num2words") from e


# $10,000 / $10,000.50 / $5
CURRENCY_RE = re.compile(r"\$\s*([\d,]+(?:\.\d{1,2})?)")
# Remaining digit runs (after currency replacement)
DIGITS_RE = re.compile(r"\d+")


def _parse_amount(raw: str) -> float:
    return float(raw.replace(",", ""))


def currency_to_words(raw: str) -> str:
    """Convert '10,000' or '10,000.50' to 'Ten Thousand Dollars' style."""
    amount = _parse_amount(raw)
    dollars = int(amount)
    cents = int(round((amount - dollars) * 100))

    dollar_words = num2words(dollars, lang="en").replace("-", " ").replace(",", "")
    dollar_words = " ".join(w.capitalize() for w in dollar_words.split())

    unit = "Dollar" if dollars == 1 else "Dollars"
    out = f"{dollar_words} {unit}"

    if cents:
        cent_words = num2words(cents, lang="en").replace("-", " ").replace(",", "")
        cent_words = " ".join(w.capitalize() for w in cent_words.split())
        cent_unit = "Cent" if cents == 1 else "Cents"
        out = f"{out} And {cent_words} {cent_unit}"

    return out


def digits_to_words(raw: str) -> str:
    """Convert '1234567' -> 'one two three four five six seven'."""
    return " ".join(num2words(int(ch), lang="en") for ch in raw)


def numbers_to_words(text: str) -> str:
    """Replace currency amounts and remaining digit runs in `text`."""

    def _currency_sub(match: re.Match[str]) -> str:
        return currency_to_words(match.group(1))

    text = CURRENCY_RE.sub(_currency_sub, text)

    def _digits_sub(match: re.Match[str]) -> str:
        return digits_to_words(match.group(0))

    return DIGITS_RE.sub(_digits_sub, text)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert $amounts to currency words and digits to spoken digit words"
    )
    parser.add_argument(
        "text",
        nargs="*",
        help="Text to convert (optional if using -f or stdin)",
    )
    parser.add_argument(
        "-f",
        "--file",
        type=Path,
        help="Input text file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Write result to this file (default: stdout)",
    )
    args = parser.parse_args()

    if args.file:
        source = args.file.read_text(encoding="utf-8")
    elif args.text:
        source = " ".join(args.text)
    elif not sys.stdin.isatty():
        source = sys.stdin.read()
    else:
        parser.error("Provide text args, --file, or pipe stdin")

    result = numbers_to_words(source)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
        print(f"Wrote {args.output}", file=sys.stderr)
    else:
        sys.stdout.write(result)
        if not result.endswith("\n"):
            sys.stdout.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
