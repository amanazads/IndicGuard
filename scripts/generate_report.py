#!/usr/bin/env python3
"""
Report generation script.
Usage:
  python scripts/generate_report.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.report import generate_report

if __name__ == "__main__":
    generate_report(verbose=True)
