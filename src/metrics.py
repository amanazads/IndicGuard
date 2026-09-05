"""
Metrics engine for IndicGuard.
Computes violation rates, compliance rates, language deltas, judge-human alignment, and statistical comparisons from evaluation data.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

JUDGE_EVALUATIONS_PATH = "results/judge_evaluations.jsonl"
HUMAN_EVALUATIONS_PATH = "results/human_evaluations.jsonl"
EVALUATIONS_PATH = JUDGE_EVALUATIONS_PATH if Path(JUDGE_EVALUATIONS_PATH).exists() and Path(JUDGE_EVALUATIONS_PATH).stat().st_size > 0 else HUMAN_EVALUATIONS_PATH
RESPONSES_PATH = "results/raw_responses.jsonl"
CASES_PATH = "data/adversarial_cases.jsonl"

CATEGORIES = ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8"]
LANGUAGES = ["english", "hindi", "hinglish", "marathi"]
INDIC_LANGUAGES = ["hindi", "hinglish", "marathi"]


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return records


def _pct(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(100.0 * num / den, 2)


def compute_metrics(
    evaluations_path: str | None = None,
    responses_path: str = RESPONSES_PATH,
    cases_path: str = CASES_PATH,
) -> dict[str, Any]:
    """
    Compute all benchmark metrics from evaluations.
    Prioritizes automated judge evaluations, with fallback to human annotations.
    """
    if evaluations_path is None:
        if Path(JUDGE_EVALUATIONS_PATH).exists() and Path(JUDGE_EVALUATIONS_PATH).stat().st_size > 0:
            evaluations_path = JUDGE_EVALUATIONS_PATH
        else:
            evaluations_path = HUMAN_EVALUATIONS_PATH

    evals = _load_jsonl(evaluations_path)
    responses = _load_jsonl(responses_path)
    cases = _load_jsonl(cases_path)
    case_map = {c["id"]: c for c in cases if "id" in c}

    if not evals:
        return {
            "status": "no_evaluations",
            "message": "No evaluations found. Run LLM Judge or complete Human Evaluation first.",
            "total_responses": len(responses),
            "total_evaluations": 0,
        }

    # De-duplicate: use latest evaluation per (case_id, model, rater_id)
    latest: dict[tuple, dict] = {}
    for e in evals:
        key = (e["case_id"], e["model"], e.get("rater_id", ""))
        latest[key] = e

    evals = list(latest.values())

    # Enrich each evaluation with case metadata if missing
    for e in evals:
        case = case_map.get(e["case_id"], {})
        if not e.get("language") and case.get("language"):
            e["language"] = case["language"]
        if not e.get("target_category") and case.get("category"):
            e["target_category"] = case["category"]
        if not e.get("category"):
            e["category"] = case.get("category", "V1")
        if not e.get("difficulty") and case.get("difficulty"):
            e["difficulty"] = case["difficulty"]
        if not e.get("attack_type") and case.get("attack_type"):
            e["attack_type"] = case["attack_type"]
        if "turn_count" not in e and "turn_count" in case:
            e["turn_count"] = case["turn_count"]

    # Helper: only include definite violations (not None/unclear)
    definite = [e for e in evals if e.get("violation") is not None]

    total = len(definite)
    violations = sum(1 for e in definite if e["violation"] is True)
    compliant = total - violations

    metrics: dict[str, Any] = {
        "status": "ok",
        "total_responses": len(responses),
        "total_evaluations": len(evals),
        "total_definite": total,
        "overall_violation_rate": _pct(violations, total),
        "overall_compliance_rate": _pct(compliant, total),
        "violations_count": violations,
        "compliant_count": compliant,
    }

    # --- 1. By Category ---
    cat_metrics: dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_evals = [e for e in definite if e.get("category") == cat or e.get("target_category") == cat]
        v = sum(1 for e in cat_evals if e["violation"])
        c = len(cat_evals) - v
        cat_metrics[cat] = {
            "total": len(cat_evals),
            "violations": v,
            "compliant": c,
            "violation_rate": _pct(v, len(cat_evals)),
            "compliance_rate": _pct(c, len(cat_evals)),
        }
    metrics["by_category"] = cat_metrics

    # --- 2. By Language ---
    lang_metrics: dict[str, dict] = {}
    for lang in LANGUAGES:
        lang_evals = [e for e in definite if e.get("language") == lang]
        v = sum(1 for e in lang_evals if e["violation"])
        c = len(lang_evals) - v
        lang_metrics[lang] = {
            "total": len(lang_evals),
            "violations": v,
            "compliant": c,
            "violation_rate": _pct(v, len(lang_evals)),
            "compliance_rate": _pct(c, len(lang_evals)),
        }
    metrics["by_language"] = lang_metrics

    # --- 3. English vs Indic Deltas (Percentage Points) ---
    en_rate = lang_metrics["english"]["compliance_rate"]
    indic_total = sum(lang_metrics[l]["total"] for l in INDIC_LANGUAGES)
    indic_compliant = sum(lang_metrics[l]["compliant"] for l in INDIC_LANGUAGES)
    indic_rate = _pct(indic_compliant, indic_total)

    metrics["english_safety_rate"] = en_rate
    metrics["indic_safety_rate"] = indic_rate
    metrics["indic_delta"] = round(indic_rate - en_rate, 2) if (en_rate is not None and indic_rate is not None) else None
    metrics["hindi_delta"] = round((lang_metrics["hindi"]["compliance_rate"] or 0) - (en_rate or 0), 2) if (en_rate is not None and lang_metrics["hindi"]["compliance_rate"] is not None) else None
    metrics["hinglish_delta"] = round((lang_metrics["hinglish"]["compliance_rate"] or 0) - (en_rate or 0), 2) if (en_rate is not None and lang_metrics["hinglish"]["compliance_rate"] is not None) else None
    metrics["marathi_delta"] = round((lang_metrics["marathi"]["compliance_rate"] or 0) - (en_rate or 0), 2) if (en_rate is not None and lang_metrics["marathi"]["compliance_rate"] is not None) else None

    # --- 4. By Model ---
    models_seen = sorted({e.get("model", "") for e in definite if e.get("model")})
    model_metrics: dict[str, dict] = {}
    for model in models_seen:
        model_evals = [e for e in definite if e.get("model") == model]
        v = sum(1 for e in model_evals if e["violation"])
        c = len(model_evals) - v

        per_lang = {}
        for lang in LANGUAGES:
            le = [e for e in model_evals if e.get("language") == lang]
            lv = sum(1 for e in le if e["violation"])
            lc = len(le) - lv
            per_lang[lang] = {
                "total": len(le),
                "violations": lv,
                "compliant": lc,
                "violation_rate": _pct(lv, len(le)),
                "compliance_rate": _pct(lc, len(le)),
            }

        m_en_cr = per_lang["english"]["compliance_rate"]
        m_indic_t = sum(per_lang[l]["total"] for l in INDIC_LANGUAGES)
        m_indic_c = sum(per_lang[l]["compliant"] for l in INDIC_LANGUAGES)
        m_indic_cr = _pct(m_indic_c, m_indic_t)
        m_indic_delta = round(m_indic_cr - m_en_cr, 2) if (m_en_cr is not None and m_indic_cr is not None) else None

        model_metrics[model] = {
            "total": len(model_evals),
            "violations": v,
            "compliant": c,
            "violation_rate": _pct(v, len(model_evals)),
            "compliance_rate": _pct(c, len(model_evals)),
            "english_compliance": m_en_cr,
            "indic_compliance": m_indic_cr,
            "indic_delta": m_indic_delta,
            "by_language": per_lang,
        }
    metrics["by_model"] = model_metrics

    # --- 5. By Difficulty ---
    diff_metrics: dict[str, dict] = {}
    for diff in ["easy", "medium", "hard"]:
        de = [e for e in definite if e.get("difficulty") == diff]
        v = sum(1 for e in de if e["violation"])
        c = len(de) - v
        diff_metrics[diff] = {
            "total": len(de),
            "violations": v,
            "compliant": c,
            "violation_rate": _pct(v, len(de)),
            "compliance_rate": _pct(c, len(de)),
        }
    metrics["by_difficulty"] = diff_metrics

    # --- 6. By Attack Type ---
    attack_types = sorted({e.get("attack_type", "unknown") for e in definite if e.get("attack_type")})
    attack_metrics: dict[str, dict] = {}
    for at in attack_types:
        ae = [e for e in definite if e.get("attack_type") == at]
        v = sum(1 for e in ae if e["violation"])
        c = len(ae) - v
        attack_metrics[at] = {
            "total": len(ae),
            "violations": v,
            "compliant": c,
            "violation_rate": _pct(v, len(ae)),
            "compliance_rate": _pct(c, len(ae)),
        }
    metrics["by_attack_type"] = attack_metrics

    # --- 7. Single vs Multi-turn ---
    single = [e for e in definite if e.get("turn_count", 1) == 1]
    multi = [e for e in definite if e.get("turn_count", 1) > 1]
    sv = sum(1 for e in single if e["violation"])
    mv = sum(1 for e in multi if e["violation"])
    sc = len(single) - sv
    mc = len(multi) - mv
    metrics["single_turn"] = {
        "total": len(single),
        "violations": sv,
        "compliant": sc,
        "violation_rate": _pct(sv, len(single)),
        "compliance_rate": _pct(sc, len(single)),
    }
    metrics["multi_turn"] = {
        "total": len(multi),
        "violations": mv,
        "compliant": mc,
        "violation_rate": _pct(mv, len(multi)),
        "compliance_rate": _pct(mc, len(multi)),
    }

    # --- 8. Judge vs Human Alignment ---
    human_evals = _load_jsonl(HUMAN_EVALUATIONS_PATH)
    judge_evals = _load_jsonl(JUDGE_EVALUATIONS_PATH)
    if human_evals and judge_evals:
        from src.judge import compute_judge_human_alignment
        metrics["judge_human_alignment"] = compute_judge_human_alignment(judge_evals, human_evals)
    else:
        metrics["judge_human_alignment"] = {"status": "insufficient_data", "message": "Requires both Judge and Human evaluations."}

    return metrics


def save_metrics(metrics: dict[str, Any], path: str = "results/metrics.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def load_metrics(path: str = "results/metrics.json") -> dict[str, Any] | None:
    if not Path(path).exists():
        return None
    with open(path) as f:
        return json.load(f)
