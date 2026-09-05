#!/usr/bin/env python3
"""
Automated batch evaluation script using LLM-as-a-Judge.
Usage:
    python scripts/run_judge.py
    python scripts/run_judge.py --limit 10
    python scripts/run_judge.py --model gemini_baseline --workers 5
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.judge import (
    JudgeEvaluator,
    save_judge_evaluations,
    load_judge_evaluations,
    compute_judge_human_alignment,
)
from src.human_eval import load_evaluations as load_human_evaluations
from src.metrics import compute_metrics, save_metrics
from src.report import generate_findings_report


def main():
    parser = argparse.ArgumentParser(description="Run Automated LLM-as-a-Judge evaluation on IndicGuard responses.")
    parser.add_argument("--responses", default="results/raw_responses.jsonl", help="Path to raw responses JSONL")
    parser.add_argument("--cases", default="data/adversarial_cases.jsonl", help="Path to adversarial cases JSONL")
    parser.add_argument("--output", default="results/judge_evaluations.jsonl", help="Output JSONL path for judge evaluations")
    parser.add_argument("--model", default=None, help="Filter responses to a specific evaluated model name")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of responses to evaluate")
    parser.add_argument("--workers", type=int, default=5, help="Number of concurrent worker threads")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing evaluations instead of appending/updating")
    args = parser.parse_args()

    responses_path = ROOT / args.responses
    cases_path = ROOT / args.cases
    output_path = ROOT / args.output

    if not responses_path.exists():
        print(f"❌ Responses file not found: {responses_path}")
        sys.exit(1)
    if not cases_path.exists():
        print(f"❌ Cases file not found: {cases_path}")
        sys.exit(1)

    # Load cases
    cases = []
    with open(cases_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line.strip()))
    case_map = {c["id"]: c for c in cases if "id" in c}

    # Load responses
    responses = []
    with open(responses_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                responses.append(json.loads(line.strip()))

    if args.model:
        responses = [r for r in responses if r.get("model") == args.model]

    if args.limit:
        responses = responses[:args.limit]

    print(f"🛡️  IndicGuard — Automated LLM-as-a-Judge")
    print(f"══════════════════════════════════════════════════════════════")
    print(f"Target Responses: {len(responses)}")
    print(f"Adversarial Cases: {len(cases)}")
    print(f"Judge Provider:   Google Gemini Flash (Multilingual Legal Rubrics)")
    print(f"Concurrency:      {args.workers} workers")
    print(f"Output File:      {output_path.relative_to(ROOT)}")
    print(f"──────────────────────────────────────────────────────────────")

    judge = JudgeEvaluator()

    # Progress tracking callback
    def on_progress(completed: int, total: int, record: dict):
        status_icon = "🚨 VIOLATION" if record.get("violation") else ("✅ SAFE" if record.get("violation") is False else "❓ UNCLEAR")
        print(f"[{completed}/{total}] {record.get('case_id')} ({record.get('model')}) ➜ {status_icon} ({record.get('category')}, {record.get('latency_seconds')}s)")

    start_time = time.perf_counter()
    evaluations = judge.evaluate_batch(
        cases=cases,
        responses=responses,
        max_workers=args.workers,
        progress_callback=on_progress,
    )
    total_elapsed = time.perf_counter() - start_time

    # Save to file
    save_judge_evaluations(evaluations, path=str(output_path), append=not args.overwrite)
    print(f"──────────────────────────────────────────────────────────────")
    print(f"✨ Successfully evaluated {len(evaluations)} responses in {total_elapsed:.2f}s!")

    # Summary Statistics
    total_valid = [e for e in evaluations if e.get("violation") is not None]
    violations = [e for e in total_valid if e.get("violation") is True]
    compliant = [e for e in total_valid if e.get("violation") is False]

    v_rate = (len(violations) / len(total_valid) * 100) if total_valid else 0.0
    c_rate = (len(compliant) / len(total_valid) * 100) if total_valid else 0.0

    print(f"\n📊 Summary Evaluation Results:")
    print(f"  • Total Evaluated:   {len(evaluations)}")
    print(f"  • Compliant (Safe):  {len(compliant)} ({c_rate:.2f}%)")
    print(f"  • Violations:        {len(violations)} ({v_rate:.2f}%)")

    # Check Judge-Human Alignment if human evaluations exist
    human_evals = load_human_evaluations()
    if human_evals:
        alignment = compute_judge_human_alignment(evaluations, human_evals)
        if alignment.get("status") == "ok":
            print(f"\n🤝 Judge vs. Human Validation Alignment:")
            print(f"  • Paired Cases:       {alignment['paired_count']}")
            print(f"  • Raw Agreement:      {alignment['raw_agreement']}%")
            print(f"  • Cohen's Kappa (κ):  {alignment['cohens_kappa']}")
            print(f"  • Precision / Recall: {alignment['precision']}% / {alignment['recall']}% (F1: {alignment['f1_score']}%)")

    # Update metrics and dynamic report
    metrics = compute_metrics(evaluations_path=str(output_path))
    save_metrics(metrics)
    generate_findings_report(metrics_path="results/metrics.json", output_path="docs/findings.md")
    print(f"\n✅ Updated results/metrics.json and docs/findings.md")


if __name__ == "__main__":
    main()
