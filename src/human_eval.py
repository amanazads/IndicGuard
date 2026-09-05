"""
Human evaluation storage and retrieval.
Handles annotation JSONL and inter-rater agreement calculations.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EVALUATIONS_PATH = "results/human_evaluations.jsonl"
VALIDATION_PATH = "data/heldout_cases.jsonl"

VALID_VIOLATION = {True, False, None}  # None = unclear
VALID_CATEGORIES = {"V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "None"}
VALID_SEVERITY = {"Low", "Medium", "High", "Critical", "N/A"}
VALID_BEHAVIOR_FOLLOWED = {"Yes", "Partial", "No"}


def load_validation_cases(path: str = VALIDATION_PATH) -> list[dict[str, Any]]:
    """Load the designated human-validation subset of adversarial cases."""
    p = Path(path)
    if not p.exists():
        p = Path(__file__).resolve().parent.parent / path
    if not p.exists():
        # Fallback to heldout_cases
        p = Path(__file__).resolve().parent.parent / "data" / "heldout_cases.jsonl"
    if not p.exists():
        return []
    cases = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    cases.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return cases


def save_evaluation(
    case_id: str,
    model: str,
    rater_id: str,
    violation: bool | None,
    category: str,
    severity: str,
    expected_behavior_followed: str = "Yes",
    evidence: str = "",
    notes: str = "",
    quality: str = "Safe",
    correct_behavior: bool | None = None,
    path: str = EVALUATIONS_PATH,
) -> dict[str, Any]:
    """Save a single human evaluation annotation."""
    record = {
        "case_id": case_id,
        "model": model,
        "rater_id": rater_id,
        "violation": violation,
        "category": category,
        "severity": severity,
        "expected_behavior_followed": expected_behavior_followed,
        "evidence": evidence,
        "notes": notes,
        "quality": quality,
        "correct_behavior": correct_behavior if correct_behavior is not None else (expected_behavior_followed == "Yes"),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def load_evaluations(path: str = EVALUATIONS_PATH) -> list[dict[str, Any]]:
    """Load all human evaluations from JSONL."""
    if not Path(path).exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def get_evaluated_keys(evaluations: list[dict[str, Any]]) -> set[tuple[str, str]]:
    """Return set of (case_id, model) pairs that have been evaluated."""
    return {(e["case_id"], e["model"]) for e in evaluations}


def compute_agreement(
    evaluations: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Compute inter-rater agreement statistics.
    Only operates on (case_id, model) pairs that have been annotated by ≥2 raters.
    """
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for e in evaluations:
        groups[(e["case_id"], e["model"])].append(e)

    # Filter to pairs with multiple raters
    multi_rated = {k: v for k, v in groups.items() if len(v) >= 2}

    if not multi_rated:
        return {
            "status": "insufficient_data",
            "message": "Insufficient annotations",
            "multi_rated_pairs": 0,
            "raw_agreement": None,
            "category_agreement": None,
            "violation_agreement": None,
            "cohens_kappa": None,
        }

    # Pairwise agreement
    total_pairs = 0
    raw_agree = 0
    violation_agree = 0
    violation_pairs = 0
    category_agree = 0
    category_pairs = 0

    for evals in multi_rated.values():
        for i in range(len(evals)):
            for j in range(i + 1, len(evals)):
                a, b = evals[i], evals[j]
                total_pairs += 1
                if a.get("violation") == b.get("violation"):
                    raw_agree += 1
                if a.get("violation") is not None and b.get("violation") is not None:
                    violation_pairs += 1
                    if a.get("violation") == b.get("violation"):
                        violation_agree += 1
                if a.get("category") and b.get("category"):
                    category_pairs += 1
                    if a.get("category") == b.get("category"):
                        category_agree += 1

    raw_agreement = raw_agree / total_pairs if total_pairs > 0 else None
    violation_agreement = violation_agree / violation_pairs if violation_pairs > 0 else None
    category_agreement = category_agree / category_pairs if category_pairs > 0 else None

    # Cohen's kappa for violation (binary)
    kappa = None
    if violation_pairs >= 2:
        kappa = _cohens_kappa_violation(multi_rated)

    return {
        "status": "ok",
        "multi_rated_pairs": len(multi_rated),
        "total_annotation_pairs": total_pairs,
        "raw_agreement": round(raw_agreement, 4) if raw_agreement is not None else None,
        "violation_agreement": round(violation_agreement, 4) if violation_agreement is not None else None,
        "category_agreement": round(category_agreement, 4) if category_agreement is not None else None,
        "cohens_kappa": round(kappa, 4) if kappa is not None else None,
    }


def _cohens_kappa_violation(
    multi_rated: dict[tuple[str, str], list[dict[str, Any]]],
) -> float | None:
    """Pairwise Cohen's kappa for binary violation labels."""
    paired: list[tuple[int, int]] = []
    for evals in multi_rated.values():
        definite = [e for e in evals if e.get("violation") is not None]
        if len(definite) >= 2:
            a_val = 1 if definite[0]["violation"] else 0
            b_val = 1 if definite[1]["violation"] else 0
            paired.append((a_val, b_val))

    if not paired or len(paired) < 2:
        return None

    n = len(paired)
    p_o = sum(1 for a, b in paired if a == b) / n
    p_a1 = sum(a for a, _ in paired) / n
    p_b1 = sum(b for _, b in paired) / n
    p_e = (p_a1 * p_b1) + ((1 - p_a1) * (1 - p_b1))
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)
