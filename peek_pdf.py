"""Inspect raw text and look for edge cases (missing values, weird months)."""
import pdfplumber
import sys
import re

paths = [
    r"r:\Code\TIDE BACKEND\TIDAL_DATA\HALDIA\2024.pdf",
    r"r:\Code\TIDE BACKEND\TIDAL_DATA\HALDIA\2000.pdf",
    r"r:\Code\TIDE BACKEND\TIDAL_DATA\DIAMOND HARBOUR\2023.pdf",
    r"r:\Code\TIDE BACKEND\TIDAL_DATA\DIAMOND HARBOUR\2000.pdf",
]

for path in paths:
    print("=" * 80)
    print(f"FILE: {path}")
    print("=" * 80)
    with pdfplumber.open(path) as pdf:
        print(f"Pages: {len(pdf.pages)}")
        page = pdf.pages[0]
        text = page.extract_text() or ""
        # Show header
        lines = text.split("\n")
        for ln in lines[:6]:
            print(repr(ln))
        # Look for any non-numeric tokens that could indicate missing data
        body = "\n".join(lines[6:])
        weird_tokens = set()
        for tok in re.findall(r"\S+", body):
            if not re.fullmatch(r"-?\d+(\.\d+)?|-+", tok):
                weird_tokens.add(tok)
        print(f"\nNon-numeric tokens in body: {sorted(weird_tokens)[:20]}")
        # Count tokens per page (should be 24*days_in_month + days_in_month for day numbers)
        nums = re.findall(r"-?\d+\.\d+|\b\d+\b", body)
        print(f"Total numeric tokens on page 1: {len(nums)}")
        # Last page
        last = pdf.pages[-1].extract_text() or ""
        print(f"\nLast page header:")
        for ln in last.split("\n")[:4]:
            print(repr(ln))
