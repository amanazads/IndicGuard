#!/usr/bin/env python3
"""
Benchmark runner script.
Usage:
  python scripts/run_benchmark.py
  python scripts/run_benchmark.py --models gemini_baseline --categories V1 V5
  python scripts/run_benchmark.py --languages english hinglish --limit 10
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed — load .env manually
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

from src.benchmark import run_benchmark
from src.models import load_model_configs


def main():
    parser = argparse.ArgumentParser(
        description="IndicGuard Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all models on all cases
  python scripts/run_benchmark.py

  # Run specific model on V5 cases
  python scripts/run_benchmark.py --models qwen_local --categories V5

  # Quick smoke test: first 5 English cases
  python scripts/run_benchmark.py --languages english --limit 5
        """
    )
    parser.add_argument("--models", nargs="+", help="Model names to run (default: all configured)")
    parser.add_argument("--categories", nargs="+", help="Filter by categories (V1-V8)")
    parser.add_argument("--languages", nargs="+", help="Filter by languages")
    parser.add_argument("--split", choices=["all", "dev", "heldout"], default="all", help="Dataset split (all, dev, heldout)")
    parser.add_argument("--limit", type=int, help="Limit number of cases per model")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent workers (default: 1)")
    parser.add_argument("--results", default="results/raw_responses.jsonl", help="Output file")
    parser.add_argument("--list-models", action="store_true", help="List configured models and exit")
    args = parser.parse_args()

    if args.list_models:
        configs = load_model_configs()
        print("\nConfigured models:")
        for c in configs:
            print(f"  {c.name:20s} | {c.provider:8s} | {c.model}")
        return

    dataset_map = {
        "all": "data/adversarial_cases.jsonl",
        "dev": "data/dev_cases.jsonl",
        "heldout": "data/heldout_cases.jsonl",
    }
    dataset_path = dataset_map.get(args.split, "data/adversarial_cases.jsonl")

    from src.benchmark import load_cases
    all_cases = load_cases(dataset_path)
    case_ids = None
    if args.limit:
        filtered = all_cases
        if args.categories:
            filtered = [c for c in filtered if c["category"] in args.categories]
        if args.languages:
            filtered = [c for c in filtered if c["language"] in args.languages]
        case_ids = [c["id"] for c in filtered[:args.limit]]

    results = run_benchmark(
        model_names=args.models,
        case_ids=case_ids,
        categories=args.categories if not case_ids else None,
        languages=args.languages if not case_ids else None,
        dataset_path=dataset_path,
        results_path=args.results,
        workers=args.workers,
        verbose=True,
    )

    print(f"\nTotal responses: {len(results)}")
    errors = [r for r in results if r.get("error")]
    if errors:
        print(f"Errors: {len(errors)}")
    print(f"\nNext step: Start evaluation UI")
    print(f"  streamlit run app.py")


if __name__ == "__main__":
    main()
