#!/usr/bin/env python3
"""
Dataset generation/validation script.
Usage:
  python scripts/generate_dataset.py --validate
  python scripts/generate_dataset.py --stats
"""

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dataset_validator import validate


def main():
    parser = argparse.ArgumentParser(
        description="IndicGuard dataset tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_dataset.py --validate
  python scripts/generate_dataset.py --validate --path data/adversarial_cases.jsonl
        """
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the adversarial dataset",
    )
    parser.add_argument(
        "--path",
        default="data/adversarial_cases.jsonl",
        help="Path to the dataset JSONL file",
    )
    args = parser.parse_args()

    if args.validate:
        ok = validate(args.path)
        sys.exit(0 if ok else 1)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
