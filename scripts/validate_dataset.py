#!/usr/bin/env python3
"""
Dataset validation script.
Usage:
  python scripts/validate_dataset.py
  python scripts/validate_dataset.py --splits
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_validator import validate, validate_splits

if __name__ == "__main__":
    ok = validate("data/adversarial_cases.jsonl")
    if ok and os.path.exists("data/dev_cases.jsonl"):
        ok = ok and validate_splits("data/adversarial_cases.jsonl", "data/dev_cases.jsonl", "data/heldout_cases.jsonl")
    sys.exit(0 if ok else 1)
