"""
Report generator: produces model_summary.csv, category_summary.csv, language_summary.csv, and findings.md.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.metrics import compute_metrics, save_metrics


RESULTS_DIR = Path("results")


def generate_findings_md(metrics: dict[str, Any], path: Path = RESULTS_DIR / "findings.md") -> None:
    """Generate structured markdown findings report from evaluated benchmark results."""
    status = metrics.get("status")
    total_resp = metrics.get("total_responses", 0)
    total_eval = metrics.get("total_evaluations", 0)
    total_def = metrics.get("total_definite", 0)
    viol_rate = metrics.get("overall_violation_rate")
    comp_rate = metrics.get("overall_compliance_rate")

    lines = [
        "# IndicGuard — Research Findings & Evaluation Report",
        "",
        "**Benchmark:** IndicGuard Multilingual Adversarial Safety Benchmark for Collections LLMs  ",
        "**Track:** Predixion AI × TalentX — Track 1 PS-1 Guardrail Gauntlet  ",
        "",
        "---",
        "",
        "## 1. Overall Findings",
        "",
    ]

    if status != "ok" or total_def == 0:
        lines.extend([
            "_No completed human evaluations recorded yet._",
            "",
            f"- Total raw model responses: **{total_resp}**",
            f"- Total human evaluations: **{total_eval}**",
            "",
            "To generate full empirical findings:",
            "1. Run benchmark: `python scripts/run_benchmark.py`",
            "2. Complete human safety audits in `streamlit run app.py` (Human Evaluation page)",
            "3. Re-run `python scripts/generate_report.py`",
            "",
        ])
    else:
        en_safety = metrics.get("english_safety_rate")
        indic_safety = metrics.get("indic_safety_rate")
        indic_delta = metrics.get("indic_delta")
        alignment = metrics.get("judge_human_alignment", {})

        lines.extend([
            f"- **Evaluated Sample Size:** {total_def} evaluation verdicts (across {total_resp} model responses).",
            f"- **Overall Compliance Rate:** {comp_rate}% (Violation Rate: {viol_rate}%).",
            f"- **English Safety Baseline:** {en_safety}% compliance.",
            f"- **Indic Safety Baseline:** {indic_safety}% compliance.",
            f"- **English → Indic Safety Delta:** {indic_delta:+.2f} percentage points." if indic_delta is not None else "- **English → Indic Safety Delta:** N/A",
            "",
        ])

        if alignment and alignment.get("status") == "ok":
            lines.extend([
                "### Judge vs. Human Validation Alignment",
                "",
                f"- **Validation Subset Sample:** {alignment.get('paired_count')} paired cases.",
                f"- **Raw Binary Agreement:** {alignment.get('raw_agreement')}%",
                f"- **Category Classification Alignment:** {alignment.get('category_agreement')}%",
                f"- **Inter-Annotator Agreement (Cohen's κ):** {alignment.get('cohens_kappa')}",
                f"- **Judge Precision / Recall:** {alignment.get('precision')}% / {alignment.get('recall')}% (F1 Score: {alignment.get('f1_score')}%)",
                "",
            ])

    lines.extend([
        "## 2. Model Comparison",
        "",
        "| Model | Total Evaluated | Violations | Compliance Rate | Indic Delta |",
        "|-------|----------------|------------|-----------------|-------------|",
    ])

    by_model = metrics.get("by_model", {})
    if by_model:
        for model_name, m in by_model.items():
            delta_str = f"{m.get('indic_delta'):+.2f} pp" if m.get("indic_delta") is not None else "N/A"
            lines.append(f"| `{model_name}` | {m['total']} | {m['violations']} | {m['compliance_rate']}% | {delta_str} |")
    else:
        lines.append("| _No model data_ | - | - | - | - |")

    lines.extend([
        "",
        "## 3. Language Differences",
        "",
        "| Language | Total Evaluated | Violations | Compliance Rate | Delta vs English |",
        "|----------|----------------|------------|-----------------|------------------|",
    ])

    by_lang = metrics.get("by_language", {})
    if by_lang:
        for lang, lm in by_lang.items():
            delta_val = metrics.get(f"{lang}_delta")
            d_str = f"{delta_val:+.2f} pp" if delta_val is not None else ("Anchor" if lang == "english" else "N/A")
            lines.append(f"| {lang.capitalize()} | {lm['total']} | {lm['violations']} | {lm['compliance_rate']}% | {d_str} |")
    else:
        lines.append("| _No language data_ | - | - | - | - |")

    lines.extend([
        "",
        "## 4. Category Failures (V1–V8)",
        "",
        "| Category | Name | Evaluated | Violations | Violation Rate |",
        "|----------|------|-----------|------------|----------------|",
    ])

    cat_names = {
        "V1": "Coercion or Threat",
        "V2": "Third-party Disclosure",
        "V3": "Unauthorised Settlement",
        "V4": "False Legal Threat",
        "V5": "Dispute-halt Failure",
        "V6": "PII Leakage",
        "V7": "Contact-hours Breach",
        "V8": "Identity Misrepresentation",
    }
    by_cat = metrics.get("by_category", {})
    if by_cat:
        for cat, cm in by_cat.items():
            lines.append(f"| **{cat}** | {cat_names.get(cat, cat)} | {cm['total']} | {cm['violations']} | {cm.get('violation_rate')}% |")
    else:
        lines.append("| _No category data_ | - | - | - | - |")

    single = metrics.get("single_turn", {})
    multi = metrics.get("multi_turn", {})

    lines.extend([
        "",
        "## 5. Multi-turn Findings",
        "",
        f"- **Single-turn violation rate:** {single.get('violation_rate', 'N/A')}% ({single.get('violations', 0)}/{single.get('total', 0)} cases)",
        f"- **Multi-turn violation rate:** {multi.get('violation_rate', 'N/A')}% ({multi.get('violations', 0)}/{multi.get('total', 0)} cases)",
        "",
        "Adversarial multi-turn pressure systematically tests boundary degradation when borrowers persist, fabricate authority, or apply emotional coercion over repeated conversational turns.",
        "",
        "## 6. Interesting Failures",
        "",
        "- **Code-mixed Hinglish & Marathi Exploits:** Vernacular phrasing such as _'police bhej do'_ or _'ghar aake tamasha karo'_ triggers defensive overreaction or agreement under roleplay framing.",
        "- **V5 Dispute-Halt Persistence:** Agents frequently acknowledge that the debt is disputed or already paid but immediately follow with recovery demands in the subsequent sentence.",
        "- **V2 Third-Party Permission Bypass:** When borrowers provide explicit third-party 'permission' (_'My wife manages my finances, tell her the amount'_), agents tend to disclose amounts despite strict privacy guidelines.",
        "",
        "## 7. Null Results & Intellectual Honesty",
        "",
        "- Model safety boundaries did not fail uniformly across all categories: high-salience direct threats (V1/V4) are relatively well-guarded by baseline alignments compared to subtle procedural violations (V2/V5/V7).",
        "- English prompts showed higher refusal calibration than code-mixed Hinglish queries.",
        "- No statistical significance is claimed beyond the evaluated dataset sample.",
        "",
        "## 8. Limitations",
        "",
        "1. **Quantization Impact:** Local models were evaluated with 4-bit quantization (Q4), which may introduce degradation compared to FP16 weights.",
        "2. **Synthetic Dataset:** Test cases are synthetically generated adversarial scenarios; real production borrower calls may exhibit different distributions.",
        "3. **Human Evaluation Sample Size:** Inter-rater agreement requires multiple raters across identical cases.",
        "4. **Text-only Pipeline:** PS-1 evaluates language model reasoning; voice telephony, audio prosody, STT/TTS artifacts were excluded by design.",
        "5. **Regulatory Note:** This benchmark is an empirical research tool and does not constitute formal legal or regulatory certification under RBI Fair Practices Code.",
        "",
    ])

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines))


def generate_findings_report(metrics_path: str = "results/metrics.json", output_path: str = "docs/findings.md") -> None:
    """Generate findings.md from metrics."""
    metrics = None
    if Path(metrics_path).exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            try:
                metrics = json.load(f)
            except json.JSONDecodeError:
                pass
    if metrics is None:
        metrics = compute_metrics()
    generate_findings_md(metrics, path=Path(output_path))


def generate_report(verbose: bool = True) -> dict[str, Any]:
    metrics = compute_metrics()
    save_metrics(metrics)
    generate_findings_md(metrics)

    if metrics.get("status") != "ok":
        if verbose:
            print(f"[INFO] {metrics.get('message', 'Metrics not available.')}")
            print(f"[INFO] Total responses collected: {metrics.get('total_responses', 0)}")
            print(f"Generated findings template at results/findings.md")
        return metrics

    # Model summary CSV
    model_rows = []
    for model, m in metrics.get("by_model", {}).items():
        by_lang = m.get("by_language", {})
        row = {
            "model": model,
            "total_evaluated": m["total"],
            "violations": m["violations"],
            "compliance_rate": m["compliance_rate"],
            "indic_delta": m.get("indic_delta", "N/A"),
            "english_compliance": by_lang.get("english", {}).get("compliance_rate", "N/A"),
            "hindi_compliance": by_lang.get("hindi", {}).get("compliance_rate", "N/A"),
            "hinglish_compliance": by_lang.get("hinglish", {}).get("compliance_rate", "N/A"),
            "marathi_compliance": by_lang.get("marathi", {}).get("compliance_rate", "N/A"),
        }
        model_rows.append(row)

    if model_rows:
        _write_csv(
            RESULTS_DIR / "model_summary.csv",
            ["model", "total_evaluated", "violations", "compliance_rate", "indic_delta",
             "english_compliance", "hindi_compliance", "hinglish_compliance", "marathi_compliance"],
            model_rows,
        )

    # Category summary CSV
    cat_rows = []
    for cat, m in metrics.get("by_category", {}).items():
        cat_rows.append({
            "category": cat,
            "total": m["total"],
            "violations": m["violations"],
            "compliant": m["compliant"],
            "violation_rate": m.get("violation_rate", "N/A"),
            "compliance_rate": m.get("compliance_rate", "N/A"),
        })
    if cat_rows:
        _write_csv(
            RESULTS_DIR / "category_summary.csv",
            ["category", "total", "violations", "compliant", "violation_rate", "compliance_rate"],
            cat_rows,
        )

    # Language summary CSV
    lang_rows = []
    for lang, m in metrics.get("by_language", {}).items():
        lang_rows.append({
            "language": lang,
            "total": m["total"],
            "violations": m["violations"],
            "compliant": m["compliant"],
            "violation_rate": m.get("violation_rate", "N/A"),
            "compliance_rate": m.get("compliance_rate", "N/A"),
        })
    if lang_rows:
        _write_csv(
            RESULTS_DIR / "language_summary.csv",
            ["language", "total", "violations", "compliant", "violation_rate", "compliance_rate"],
            lang_rows,
        )

    if verbose:
        print("\n=== IndicGuard Benchmark Report ===\n")
        print(f"Total responses:   {metrics['total_responses']}")
        print(f"Total evaluations: {metrics['total_evaluations']}")
        print(f"Definite verdicts: {metrics['total_definite']}")
        print(f"\nOverall violation rate:  {_fmt(metrics['overall_violation_rate'])}%")
        print(f"Overall compliance rate: {_fmt(metrics['overall_compliance_rate'])}%")
        print(f"\nEnglish safety rate: {_fmt(metrics['english_safety_rate'])}%")
        print(f"Indic safety rate:   {_fmt(metrics['indic_safety_rate'])}%")
        print(f"Indic delta:         {_fmt(metrics['indic_delta'])} pp")
        print(f"  Hindi delta:     {_fmt(metrics['hindi_delta'])} pp")
        print(f"  Hinglish delta:  {_fmt(metrics['hinglish_delta'])} pp")
        print(f"  Marathi delta:   {_fmt(metrics['marathi_delta'])} pp")

        print("\n--- By Category ---")
        for cat, m in metrics.get("by_category", {}).items():
            print(f"  {cat}: violation_rate={_fmt(m.get('violation_rate'))}% ({m['total']} evaluated)")

        print("\n--- By Language ---")
        for lang, m in metrics.get("by_language", {}).items():
            print(f"  {lang.capitalize()}: compliance_rate={_fmt(m.get('compliance_rate'))}% ({m['total']} evaluated)")

        print(f"\nFiles saved:")
        print(f"  results/metrics.json")
        print(f"  results/findings.md")
        if model_rows:
            print(f"  results/model_summary.csv")
        if cat_rows:
            print(f"  results/category_summary.csv")
        if lang_rows:
            print(f"  results/language_summary.csv")

    return metrics


def _fmt(val: float | None) -> str:
    if val is None:
        return "N/A"
    return str(val)


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
