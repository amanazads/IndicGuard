"""
Benchmark runner: orchestrates test cases → model responses → results storage.
Human evaluation is the evaluation mechanism; no LLM judge is used.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.models import ModelConfig, ModelRunner, get_runner, load_model_configs, get_benchmark_config

PROMPT_PATH = "prompts/baseline_system_prompt.txt"
DATASET_PATH = "data/adversarial_cases.jsonl"
RESULTS_PATH = "results/raw_responses.jsonl"


def load_prompt_template(path: str = PROMPT_PATH) -> str:
    with open(path) as f:
        return f.read()


def fill_prompt(template: str, config: dict[str, Any]) -> str:
    """Substitute placeholders in the baseline system prompt."""
    return template.format(
        LENDER=config.get("lender", "FinServe Capital"),
        NAME=config.get("name", "Ravi Kumar"),
        DPD=config.get("dpd", 45),
        PRODUCT=config.get("product", "personal loan"),
        AMOUNT=config.get("amount", "₹52,000"),
    )


def load_cases(path: str = DATASET_PATH) -> list[dict[str, Any]]:
    cases = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def build_conversation(case: dict[str, Any]) -> list[dict[str, str]]:
    """
    Convert borrower_turns into a conversation list.
    Multi-turn: alternates user/assistant (no assistant responses present,
    since we are asking the model to respond to the LAST user turn).
    """
    turns = case["borrower_turns"]
    conversation = []
    for i, turn in enumerate(turns):
        conversation.append({"role": "user", "content": turn})
        # Add placeholder assistant turns between user turns (all-user chains)
        # so the model receives proper alternating structure
        if i < len(turns) - 1:
            conversation.append({
                "role": "assistant",
                "content": "[Agent acknowledges and continues the call]",
            })
    return conversation


def save_response(record: dict[str, Any], path: str = RESULTS_PATH) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_case(
    case: dict[str, Any],
    runner: ModelRunner,
    prompt_template: str,
    benchmark_config: dict[str, Any],
) -> dict[str, Any]:
    """Run a single test case through a model and return the raw response record."""
    meta = case.get("metadata", {})
    system_prompt = fill_prompt(prompt_template, {
        "lender": benchmark_config.get("default_lender", "FinServe Capital"),
        "name": benchmark_config.get("default_name", "Ravi Kumar"),
        "dpd": meta.get("dpd", benchmark_config.get("default_dpd", 45)),
        "product": meta.get("product", benchmark_config.get("default_product", "personal loan")),
        "amount": f"₹{meta.get('amount', benchmark_config.get('default_amount', '52,000'))}",
    })

    conversation = build_conversation(case)
    response = runner.generate(system_prompt, conversation)

    return {
        "case_id": case["id"],
        "model": runner.name,
        "language": case["language"],
        "category": case["category"],
        "category_name": case.get("category_name", ""),
        "difficulty": case.get("difficulty"),
        "attack_type": case.get("attack_type"),
        "turn_count": case.get("turn_count", 1),
        "borrower_turns": case.get("borrower_turns", []),
        "expected_behavior": case.get("expected_behavior", ""),
        "violation_condition": case.get("violation_condition", ""),
        "response": response.text,
        "status": "success" if not response.error else "error",
        "error": response.error,
        "latency_ms": int(response.latency_seconds * 1000),
        "latency_seconds": round(response.latency_seconds, 3),
        "prompt_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_benchmark(
    model_names: list[str] | None = None,
    case_ids: list[str] | None = None,
    categories: list[str] | None = None,
    languages: list[str] | None = None,
    dataset_path: str = DATASET_PATH,
    results_path: str = RESULTS_PATH,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run the benchmark for selected models and cases.

    Args:
        model_names: Subset of model names to run (None = all).
        case_ids: Subset of case IDs to run (None = all).
        categories: Filter by category (None = all).
        languages: Filter by language (None = all).
        dataset_path: Path to cases JSONL file.
        results_path: Where to append results.
        verbose: Print progress.
    """
    prompt_template = load_prompt_template()
    all_cases = load_cases(dataset_path)
    model_configs = load_model_configs()
    bench_config = get_benchmark_config()

    # Filter models
    if model_names:
        model_configs = [m for m in model_configs if m.name in model_names]

    # Filter cases
    cases = all_cases
    if case_ids:
        cases = [c for c in cases if c["id"] in case_ids]
    if categories:
        cases = [c for c in cases if c["category"] in categories]
    if languages:
        cases = [c for c in cases if c["language"] in languages]

    if not model_configs:
        print("[WARNING] No models selected.")
        return []

    if not cases:
        print("[WARNING] No cases matched the filters.")
        return []

    all_results = []

    for model_cfg in model_configs:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Model: {model_cfg.name} ({model_cfg.provider}/{model_cfg.model})")
            print(f"Running {len(cases)} cases...")
            print(f"{'='*60}")

        try:
            runner = get_runner(model_cfg)
        except Exception as e:
            print(f"[ERROR] Cannot create runner for {model_cfg.name}: {e}")
            continue

        for i, case in enumerate(cases, 1):
            if verbose:
                print(f"  [{i:3d}/{len(cases)}] {case['id']} ({case['language']}, {case['category']})", end=" ... ", flush=True)

            record = run_case(case, runner, prompt_template, bench_config)

            if record.get("error"):
                if verbose:
                    print(f"ERROR: {record['error'][:80]}")
                # If the model is completely unavailable, skip remaining cases for this model
                if "Cannot connect" in (record.get("error") or "") or "not set" in (record.get("error") or ""):
                    print(f"\n[SKIP] Stopping model {model_cfg.name}: {record['error']}")
                    break
            else:
                if verbose:
                    snippet = record["response"][:60].replace("\n", " ")
                    print(f"OK ({record['latency_seconds']}s) — {snippet}...")

            save_response(record, results_path)
            all_results.append(record)

    if verbose:
        print(f"\n\nDone. {len(all_results)} responses saved to {results_path}")

    return all_results
