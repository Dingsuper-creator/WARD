#!/usr/bin/env python3
"""Run checkpoint inference or prepare and execute semantic judge jobs."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from ward_eval.entrypoints import run_main

if __name__ == "__main__":
    raise SystemExit(run_main())
