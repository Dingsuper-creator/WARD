#!/usr/bin/env python3
"""Evaluate checkpoint predictions and six longitudinal diagnostics offline."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ward_eval.entrypoints import eval_main

if __name__ == "__main__":
    raise SystemExit(eval_main())
