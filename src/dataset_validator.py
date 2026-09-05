"""
Dataset validator for IndicGuard adversarial cases.
Validates JSONL structure, required fields, IDs, and distribution.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = [
    "id", "language", "category", "category_name",
    "difficulty", "attack_type", "turn_count",
    "scenario", "borrower_turns", "expected_behavior", "violation_condition",
]

VALID_LANGUAGES = {"english", "hindi", "hinglish", "marathi"}
VALID_CATEGORIES = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}
MIN_CASES = 150
MIN_MULTI_TURN = 40


def load_cases(path: str = "data/adversarial_cases.jsonl") -> list[dict[str, Any]]:
    cases = []
    if not Path(path).exists():
        return []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  [ERROR] Line {i}: Invalid JSON — {e}")
    return cases


def split_dataset(
    src_path: str = "data/adversarial_cases.jsonl",
    dev_path: str = "data/dev_cases.jsonl",
    heldout_path: str = "data/heldout_cases.jsonl",
) -> tuple[int, int]:
    """
    Split dataset into 80% dev and 20% held-out sets stratified by (category, language).
    """
    cases = load_cases(src_path)
    if not cases:
        raise FileNotFoundError(f"Source dataset not found at {src_path}")

    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for c in cases:
        groups[(c["category"], c["language"])].append(c)

    dev_cases: list[dict[str, Any]] = []
    heldout_cases: list[dict[str, Any]] = []

    for key, cell_cases in sorted(groups.items()):
        # 4 out of 5 cases to dev (80%), 1 out of 5 to held-out (20%)
        split_idx = int(len(cell_cases) * 0.8)
        if split_idx == len(cell_cases) and len(cell_cases) > 1:
            split_idx = len(cell_cases) - 1
        dev_cases.extend(cell_cases[:split_idx])
        heldout_cases.extend(cell_cases[split_idx:])

    Path(dev_path).parent.mkdir(parents=True, exist_ok=True)
    with open(dev_path, "w") as f:
        for c in dev_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    Path(heldout_path).parent.mkdir(parents=True, exist_ok=True)
    with open(heldout_path, "w") as f:
        for c in heldout_cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    return len(dev_cases), len(heldout_cases)


def validate_splits(
    src_path: str = "data/adversarial_cases.jsonl",
    dev_path: str = "data/dev_cases.jsonl",
    heldout_path: str = "data/heldout_cases.jsonl",
    verbose: bool = True,
) -> bool:
    """Validate that dev and held-out splits are disjoint, complete, and balanced."""
    if not Path(dev_path).exists() or not Path(heldout_path).exists():
        split_dataset(src_path, dev_path, heldout_path)

    all_cases = load_cases(src_path)
    dev_cases = load_cases(dev_path)
    heldout_cases = load_cases(heldout_path)

    all_ids = {c["id"] for c in all_cases}
    dev_ids = {c["id"] for c in dev_cases}
    heldout_ids = {c["id"] for c in heldout_cases}

    errors: list[str] = []

    # Check disjointness
    overlap = dev_ids.intersection(heldout_ids)
    if overlap:
        errors.append(f"Overlap between dev and held-out IDs: {overlap}")

    # Check completeness
    combined_ids = dev_ids.union(heldout_ids)
    if combined_ids != all_ids:
        errors.append(f"Dev + Heldout ({len(combined_ids)}) != Total Cases ({len(all_ids)})")

    # Check distributions in dev and heldout
    for name, split_cases in [("Dev", dev_cases), ("Heldout", heldout_cases)]:
        cats = {c["category"] for c in split_cases}
        langs = {c["language"] for c in split_cases}
        if cats != VALID_CATEGORIES:
            errors.append(f"{name} set missing categories: {VALID_CATEGORIES - cats}")
        if langs != VALID_LANGUAGES:
            errors.append(f"{name} set missing languages: {VALID_LANGUAGES - langs}")

    if verbose:
        print("\nSplit validation")
        print("----------------")
        print(f"Dev cases:     {len(dev_cases)} ({len(dev_cases)/len(all_cases):.1%})")
        print(f"Heldout cases: {len(heldout_cases)} ({len(heldout_cases)/len(all_cases):.1%})")
        print(f"Overlap:       {len(overlap)}")
        if errors:
            print(f"Errors: {errors}")
            print("Status: FAIL")
        else:
            print("Status: PASS")

    return len(errors) == 0


def validate(path: str = "data/adversarial_cases.jsonl", verbose: bool = True) -> bool:
    """
    Validate the adversarial dataset.
    Returns True if all checks pass, False otherwise.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if not Path(path).exists():
        print(f"[FATAL] Dataset not found: {path}")
        return False

    cases = load_cases(path)
    total = len(cases)

    # --- Required fields ---
    ids_seen: set[str] = set()
    for case in cases:
        cid = case.get("id", f"<unknown@{cases.index(case)}>")

        for field in REQUIRED_FIELDS:
            if field not in case:
                errors.append(f"{cid}: Missing required field '{field}'")

        # Unique IDs
        if cid in ids_seen:
            errors.append(f"Duplicate ID: {cid}")
        ids_seen.add(cid)

        # Language
        lang = case.get("language", "")
        if lang not in VALID_LANGUAGES:
            errors.append(f"{cid}: Invalid language '{lang}'")

        # Category
        cat = case.get("category", "")
        if cat not in VALID_CATEGORIES:
            errors.append(f"{cid}: Invalid category '{cat}'")

        # Difficulty
        diff = case.get("difficulty", "")
        if diff not in VALID_DIFFICULTIES:
            warnings.append(f"{cid}: Unusual difficulty '{diff}'")

        # borrower_turns must be non-empty list
        turns = case.get("borrower_turns", [])
        if not isinstance(turns, list) or len(turns) == 0:
            errors.append(f"{cid}: 'borrower_turns' must be a non-empty list")

        # expected_behavior must be non-empty string
        eb = case.get("expected_behavior", "")
        if not eb or not eb.strip():
            errors.append(f"{cid}: 'expected_behavior' is empty")

        # violation_condition must be non-empty string
        vc = case.get("violation_condition", "")
        if not vc or not vc.strip():
            errors.append(f"{cid}: 'violation_condition' is empty")

    # --- Distribution checks ---
    lang_counts = Counter(c.get("language", "") for c in cases)
    cat_counts = Counter(c.get("category", "") for c in cases)
    multi_turn = sum(1 for c in cases if c.get("turn_count", 1) > 1)
    single_turn = total - multi_turn

    if verbose:
        print("\nDataset validation")
        print("------------------")
        print(f"Total cases: {total}")
        print("\nLanguages:")
        for lang in sorted(VALID_LANGUAGES):
            n = lang_counts.get(lang, 0)
            print(f"  {lang.capitalize():12s}: {n}")

        print("\nCategories:")
        for cat in sorted(VALID_CATEGORIES):
            n = cat_counts.get(cat, 0)
            print(f"  {cat}: {n}")

        print(f"\nMulti-turn cases:  {multi_turn}")
        print(f"Single-turn cases: {single_turn}")

    # Minimum cases
    if total < MIN_CASES:
        errors.append(f"Only {total} cases found, minimum is {MIN_CASES}")

    # Multi-turn minimum
    if multi_turn < MIN_MULTI_TURN:
        errors.append(f"Only {multi_turn} multi-turn cases found, minimum is {MIN_MULTI_TURN}")

    # All languages present
    for lang in VALID_LANGUAGES:
        if lang_counts.get(lang, 0) == 0:
            errors.append(f"No cases for language: {lang}")

    # All categories present
    for cat in VALID_CATEGORIES:
        if cat_counts.get(cat, 0) == 0:
            errors.append(f"No cases for category: {cat}")

    if verbose:
        if warnings:
            print(f"\nWarnings ({len(warnings)}):")
            for w in warnings:
                print(f"  [WARN] {w}")

        if errors:
            print(f"\nErrors ({len(errors)}):")
            for e in errors:
                print(f"  [ERROR] {e}")
            print("\nStatus: FAIL")
        else:
            print("\nStatus: PASS")

    return len(errors) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate IndicGuard dataset")
    parser.add_argument("--path", default="data/adversarial_cases.jsonl")
    parser.add_argument("--check-splits", action="store_true", help="Validate dev/heldout splits")
    args = parser.parse_args()

    ok = validate(args.path)
    if args.check_splits or Path("data/dev_cases.jsonl").exists():
        ok = ok and validate_splits()

    sys.exit(0 if ok else 1)
