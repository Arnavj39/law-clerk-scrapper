#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch runner that reuses the canonical logic in main.py

Why this file is tiny:
- To keep behavior 1:1 with main.py (column mapping, strict Bing query, face
  detection/embeddings, clustering, CSV outputs), we delegate directly to
  main._cli_batch(). That way any future fixes in main.py automatically apply
  to batch runs too.

Usage:
  python batch_process.py LC_Law_Clerks-3.xlsx

Notes:
- Exits with a friendly message if the Excel file path is missing or wrong.
- Requires main.py to live in the same folder and contain _cli_batch().
- All outputs go to ./batch_outputs/ just like main.py.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    # Import the single source of truth for batch logic
    from main import _cli_batch  # type: ignore
except Exception as e:  # pragma: no cover
    print("[batch_process] Could not import from main.py — make sure it's present and error‑free.\n"
          f"Underlying error: {e}")
    sys.exit(1)


def _usage() -> None:
    print(
        """
Usage:
  python batch_process.py <excel_file.xlsx>
        """.strip()
    )


def run(sheet_path: str) -> None:
    """Run the same batch pipeline as main.py for the given Excel file."""
    p = Path(sheet_path)
    if not p.exists():
        print(f"[batch_process] File not found: {sheet_path}")
        _usage()
        sys.exit(2)
    if p.suffix.lower() not in {".xlsx", ".xlsm", ".xltx", ".xltm"}:
        print(f"[batch_process] Not an Excel file: {sheet_path}")
        _usage()
        sys.exit(3)

    # Delegate to main.py's batch implementation so column detection, strict
    # search queries, clustering, and CSV output stay identical.
    _cli_batch(sheet_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        _usage()
        sys.exit(1)
    run(sys.argv[1])
