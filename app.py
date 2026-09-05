"""
IndicGuard — Multilingual Adversarial Safety Benchmark for Collections LLMs
Interactive Evaluation Dashboard (LLM-as-a-Judge + Human Validation)

Pages:
  1. Overview
  2. Run Benchmark
  3. LLM-as-a-Judge
  4. Human Validation
  5. Model Comparison
  6. Language Analysis
  7. Violation Analysis
  8. Multi-turn Analysis
  9. Failure Cases
  10. Live Test
  11. Methodology
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter, defaultdict

# pyrefly: ignore [missing-import]
import streamlit as st
import pandas as pd

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# Load .env
try:
    # pyrefly: ignore [missing-import]
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

env_path = ROOT / ".env"
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

import importlib

from src import models, benchmark, judge, human_eval, metrics, report
importlib.reload(models)
importlib.reload(benchmark)
importlib.reload(judge)
importlib.reload(human_eval)
importlib.reload(metrics)
importlib.reload(report)

from src.models import load_model_configs, get_runner, ModelConfig
from src.benchmark import load_prompt_template, fill_prompt, get_benchmark_config, run_benchmark, save_response
from src.judge import (
    JudgeEvaluator,
    load_judge_evaluations,
    save_judge_evaluations,
    compute_judge_human_alignment,
)

# Resilient human_eval imports
save_evaluation = human_eval.save_evaluation
load_human_evaluations = human_eval.load_evaluations
compute_agreement = human_eval.compute_agreement
get_evaluated_keys = human_eval.get_evaluated_keys

if hasattr(human_eval, "load_validation_cases"):
    load_validation_cases = human_eval.load_validation_cases
else:
    def load_validation_cases(path: str = "data/validation_subset.jsonl") -> list[dict]:
        p = Path(path)
        if not p.exists():
            p = ROOT / path
        if not p.exists():
            p = ROOT / "data" / "heldout_cases.jsonl"
        if not p.exists():
            return []
        cases = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        cases.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        pass
        return cases

from src.metrics import compute_metrics, save_metrics, load_metrics
from src.report import generate_findings_report

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IndicGuard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
section[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* Header */
.indicguard-header {
    background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f172a 100%);
    border-radius: 12px;
    padding: 26px 32px;
    margin-bottom: 24px;
    border: 1px solid #1e40af33;
}
.indicguard-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: #f8fafc;
    margin: 0;
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.indicguard-subtitle {
    color: #94a3b8;
    font-size: 0.95rem;
    margin-top: 4px;
    font-weight: 400;
}

/* Metric cards */
.metric-card {
    background: linear-gradient(135deg, #1e293b, #0f172a);
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 16px 12px;
    text-align: center;
}
.metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #60a5fa;
}
.metric-label {
    font-size: 0.72rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-top: 4px;
}

/* Response box */
.response-box {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.90rem;
    color: #e2e8f0;
    line-height: 1.5;
    white-space: pre-wrap;
    margin-bottom: 8px;
}

/* Verdict Badges */
.violation-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    background: #450a0a;
    border: 1px solid #dc2626;
    color: #fca5a5;
}
.safe-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    background: #052e16;
    border: 1px solid #16a34a;
    color: #86efac;
}
.unclear-tag {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.80rem;
    font-weight: 700;
    background: #1e293b;
    border: 1px solid #64748b;
    color: #cbd5e1;
}

.judge-card {
    background: #0f172a;
    border: 1px solid #3b82f6;
    border-radius: 10px;
    padding: 16px;
    margin-top: 10px;
}
</style>
""", unsafe_allow_html=True)


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=5)
def load_cases(dataset_path: str = "data/adversarial_cases.jsonl"):
    path = ROOT / dataset_path
    if not path.exists():
        return []
    cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    cases.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return cases


@st.cache_data(ttl=5)
def load_validation_subset_cases():
    return load_validation_cases()


@st.cache_data(ttl=3)
def load_responses():
    path = ROOT / "results" / "raw_responses.jsonl"
    if not path.exists():
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line.strip()))
                except json.JSONDecodeError:
                    pass
    return records


@st.cache_data(ttl=3)
def load_judge_evals():
    return load_judge_evaluations()


@st.cache_data(ttl=3)
def load_human_evals():
    return load_human_evaluations()


CATEGORIES = {
    "V1": "Coercion or Threat",
    "V2": "Third-party Disclosure",
    "V3": "Unauthorised Settlement",
    "V4": "False Legal Threat",
    "V5": "Dispute-halt Failure",
    "V6": "PII Leakage",
    "V7": "Contact-hours Breach",
    "V8": "Identity Misrepresentation",
}

LANGUAGES = ["english", "hindi", "hinglish", "marathi"]
DIFFICULTIES = ["easy", "medium", "hard"]


# ── Navigation ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 8px 0 16px 0;">
        <div style="font-size: 1.4rem; font-weight: 800; color: #60a5fa;">🛡️ IndicGuard</div>
        <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 2px;">Multilingual Adversarial Safety Benchmark</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Overview",
            "⚡ Run Benchmark",
            "🤖 LLM-as-a-Judge",
            "🔍 Human Validation",
            "🤖 Model Comparison",
            "🌐 Language Analysis",
            "⚠️ Violation Analysis",
            "🔄 Multi-turn Analysis",
            "💥 Failure Cases",
            "🚀 Live Test",
            "📖 Methodology",
        ],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:#334155;margin:12px 0;'>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:0.68rem;color:#94a3b8;'>Predixion AI × TalentX<br>Track 1 — PS-1 Guardrail Gauntlet</div>", unsafe_allow_html=True)


# ── Header Helper ─────────────────────────────────────────────────────────────
def header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="indicguard-header">
        <div class="indicguard-title">{title}</div>
        {"<div class='indicguard-subtitle'>" + subtitle + "</div>" if subtitle else ""}
    </div>
    """, unsafe_allow_html=True)


def metric_card(label: str, value: str, col):
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    header("INDICGUARD", "Automated Multilingual Safety Benchmark for Collections LLMs")

    cases = load_cases()
    responses = load_responses()
    judge_evals = load_judge_evals()
    human_evals = load_human_evals()

    # Compute metrics
    metrics = compute_metrics()

    models_run = sorted({r["model"] for r in responses})
    total_judge_evals = len(judge_evals)
    viol_rate = f"{metrics.get('overall_violation_rate', 'N/A')}%"
    comp_rate = f"{metrics.get('overall_compliance_rate', 'N/A')}%"
    indic_delta = f"{metrics.get('indic_delta', 'N/A')} pp"

    alignment = metrics.get("judge_human_alignment", {})
    kappa_str = f"κ = {alignment.get('cohens_kappa')}" if alignment.get("cohens_kappa") is not None else "Pending"

    cols = st.columns(6)
    metric_card("Total Cases", str(len(cases)), cols[0])
    metric_card("Languages", "4", cols[1])
    metric_card("Judge Evaluated", str(total_judge_evals), cols[2])
    metric_card("Compliance Rate", comp_rate, cols[3])
    metric_card("Indic Delta", indic_delta, cols[4])
    metric_card("Judge-Human Agreement", kappa_str, cols[5])

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🌐 Evaluation by Language")
        by_lang = metrics.get("by_language", {})
        if by_lang:
            lang_rows = []
            for l, lm in by_lang.items():
                lang_rows.append({
                    "Language": l.capitalize(),
                    "Total Evaluated": lm["total"],
                    "Violations": lm["violations"],
                    "Compliance Rate": f"{lm['compliance_rate']}%",
                    "Delta vs English": f"{metrics.get(f'{l}_delta', 0):+.2f} pp" if l != "english" and metrics.get(f'{l}_delta') is not None else "Anchor",
                })
            st.dataframe(pd.DataFrame(lang_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run LLM Judge to see language evaluation metrics.")

    with c2:
        st.subheader("⚠️ Evaluation by Category (V1–V8)")
        by_cat = metrics.get("by_category", {})
        if by_cat:
            cat_rows = []
            for c, cm in by_cat.items():
                cat_rows.append({
                    "Category": c,
                    "Name": CATEGORIES.get(c, ""),
                    "Evaluated": cm["total"],
                    "Violations": cm["violations"],
                    "Violation Rate": f"{cm['violation_rate']}%",
                })
            st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)
        else:
            st.info("Run LLM Judge to see category evaluation metrics.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: RUN BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚡ Run Benchmark":
    header("Run Benchmark", "Generate model responses across multilingual adversarial cases")

    configs = load_model_configs()
    model_names = [m.name for m in configs]
    responses = load_responses()

    with st.expander("ℹ️ Benchmark Configuration", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            sel_models = st.multiselect("Models to Run", model_names, default=[model_names[0]] if model_names else [])
            sel_langs = st.multiselect("Languages", LANGUAGES, default=LANGUAGES)
        with col2:
            sel_cats = st.multiselect("Categories", list(CATEGORIES.keys()), default=list(CATEGORIES.keys()))
            dpath = st.text_input("Dataset Path", value="data/adversarial_cases.jsonl")

    st.markdown(f"**Current Captured Responses:** `{len(responses)}` records")

    if st.button("🚀 START BENCHMARK RUN", type="primary"):
        if not sel_models:
            st.error("Please select at least one model.")
        else:
            with st.spinner(f"Running benchmark on {len(sel_models)} model(s)..."):
                progress_bar = st.progress(0.0)
                from src.benchmark import run_benchmark
                res = run_benchmark(
                    model_names=sel_models,
                    categories=sel_cats if sel_cats else None,
                    languages=sel_langs if sel_langs else None,
                    dataset_path=dpath,
                    verbose=True,
                )
                progress_bar.progress(1.0)
                load_responses.clear()
                st.success(f"✅ Run complete! Collected {len(res)} response(s).")
                st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LLM-AS-A-JUDGE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 LLM-as-a-Judge":
    header("Automated LLM-as-a-Judge", "Regulatory compliance evaluator powered by Gemini Flash with Multilingual Rubrics")

    responses = load_responses()
    cases = load_cases()
    judge_evals = load_judge_evals()

    if not responses:
        st.warning("No model responses found. Run the benchmark first from **⚡ Run Benchmark**.")
        st.stop()

    j_map = {(e["case_id"], e["model"]): e for e in judge_evals}
    unevaluated_count = sum(1 for r in responses if (r["case_id"], r["model"]) not in j_map)

    # Top stats
    c1, c2, c3, c4 = st.columns(4)
    metric_card("Total Responses", str(len(responses)), c1)
    metric_card("Judge Evaluated", str(len(judge_evals)), c2)
    metric_card("Unevaluated", str(unevaluated_count), c3)
    metric_card("Judge Provider", "Gemini Flash", c4)

    st.markdown("---")

    # Batch execution controls
    st.subheader("⚡ Batch Evaluation Execution")
    col_b1, col_b2, col_b3 = st.columns([2, 2, 2])
    with col_b1:
        target_model = st.selectbox("Target Model", ["All Models"] + sorted({r["model"] for r in responses}))
    with col_b2:
        workers_count = st.slider("Worker Threads (Concurrency)", min_value=1, max_value=10, value=5)
    with col_b3:
        st.markdown("<br>", unsafe_allow_html=True)
        run_judge_btn = st.button("▶ RUN AUTOMATED JUDGE BATCH", type="primary", use_container_width=True)

    if run_judge_btn:
        target_resps = responses
        if target_model != "All Models":
            target_resps = [r for r in target_resps if r["model"] == target_model]

        p_bar = st.progress(0.0)
        status_txt = st.empty()

        judge = JudgeEvaluator()

        def on_prog(done, tot, rec):
            p_bar.progress(done / tot)
            status_txt.text(f"Evaluated [{done}/{tot}]: {rec.get('case_id')} ({rec.get('model')}) ➜ {'VIOLATION' if rec.get('violation') else 'SAFE'}")

        with st.spinner("Executing Automated LLM-as-a-Judge evaluations..."):
            new_evals = judge.evaluate_batch(
                cases=cases,
                responses=target_resps,
                max_workers=workers_count,
                progress_callback=on_prog,
            )
            # Merge & save
            existing_evals = load_judge_evaluations()
            merged_map = {(e["case_id"], e["model"]): e for e in existing_evals}
            for ne in new_evals:
                merged_map[(ne["case_id"], ne["model"])] = ne

            save_judge_evaluations(list(merged_map.values()), append=False)
            load_judge_evals.clear()

            # Update metrics & findings
            metrics = compute_metrics(evaluations_path="results/judge_evaluations.jsonl")
            save_metrics(metrics)
            generate_findings_report(metrics_path="results/metrics.json", output_path="docs/findings.md")

            st.success(f"✅ Successfully evaluated {len(new_evals)} responses! Metrics & findings report updated.")
            st.rerun()

    # Verdict Inspector
    st.markdown("---")
    st.subheader("🔍 Judge Verdict Inspector")

    f1, f2, f3, f4 = st.columns(4)
    with f1:
        f_mod = st.selectbox("Filter Model", ["All"] + sorted({e["model"] for e in judge_evals})) if judge_evals else "All"
    with f2:
        f_lang = st.selectbox("Filter Language", ["All"] + LANGUAGES)
    with f3:
        f_cat = st.selectbox("Filter Category", ["All"] + list(CATEGORIES.keys()))
    with f4:
        f_v = st.selectbox("Verdict", ["All", "Violations Only", "Safe Only"])

    filtered_judge = judge_evals
    if f_mod != "All":
        filtered_judge = [e for e in filtered_judge if e["model"] == f_mod]
    if f_lang != "All":
        filtered_judge = [e for e in filtered_judge if e.get("language") == f_lang]
    if f_cat != "All":
        filtered_judge = [e for e in filtered_judge if e.get("category") == f_cat]
    if f_v == "Violations Only":
        filtered_judge = [e for e in filtered_judge if e.get("violation") is True]
    elif f_v == "Safe Only":
        filtered_judge = [e for e in filtered_judge if e.get("violation") is False]

    st.markdown(f"**Showing {len(filtered_judge)} judge verdicts**")

    resp_map = {(r["case_id"], r["model"]): r for r in responses}
    case_map = {c["id"]: c for c in cases}

    for ev in filtered_judge[:30]:
        r = resp_map.get((ev["case_id"], ev["model"]), {})
        c = case_map.get(ev["case_id"], {})
        with st.container():
            c_left, c_right = st.columns([3, 2])
            with c_left:
                st.markdown(f"### `{ev['case_id']}` — {ev['model']}")
                st.markdown(f"**Language:** `{ev.get('language', '').capitalize()}` | **Category:** `{ev.get('category')}` ({CATEGORIES.get(ev.get('category', ''), '')})")
                if r.get("borrower_turns"):
                    st.markdown("**Borrower Conversation:**")
                    for turn in r["borrower_turns"]:
                        st.markdown(f"""<div class="response-box">{turn}</div>""", unsafe_allow_html=True)
                st.markdown("**Agent Response:**")
                st.markdown(f"""<div class="response-box">{r.get('response', '(empty response)')}</div>""", unsafe_allow_html=True)

            with c_right:
                st.markdown("### 🤖 Judge Verdict")
                if ev.get("violation") is True:
                    st.markdown("""<span class="violation-tag">🚨 VIOLATION DETECTED</span>""", unsafe_allow_html=True)
                elif ev.get("violation") is False:
                    st.markdown("""<span class="safe-tag">✅ COMPLIANT (SAFE)</span>""", unsafe_allow_html=True)
                else:
                    st.markdown("""<span class="unclear-tag">❓ UNCLEAR</span>""", unsafe_allow_html=True)

                st.markdown(f"**Severity:** `{ev.get('severity')}` | **Confidence:** `{ev.get('confidence', 0.90):.0%}` | **Latency:** `{ev.get('latency_seconds', 'N/A')}s`")
                if ev.get("evidence"):
                    st.markdown(f"**Evidence Quote:** _{ev['evidence']}_")
                st.markdown(f"**Reasoning:** {ev.get('reasoning')}")
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: HUMAN VALIDATION
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔍 Human Validation":
    header("Human Validation Subset", "Human Ground Truth Auditing & Judge Alignment Verification")

    st.info("ℹ️ **Validation Subset Requirement:** To ensure safety evaluations remain grounded and trustworthy, human raters audit a designated 32-case stratified subset. IndicGuard calculates inter-rater alignment (Cohen's κ) between the Automated Judge and Human ground truth.")

    val_cases = load_validation_subset_cases()
    val_case_ids = {c["id"] for c in val_cases}

    responses = load_responses()
    judge_evals = load_judge_evals()
    human_evals = load_human_evals()

    # Filter responses to validation subset
    val_responses = [r for r in responses if r["case_id"] in val_case_ids]

    if not val_responses:
        st.warning("No validation subset responses found. Ensure baseline benchmark has run.")
        st.stop()

    j_map = {(e["case_id"], e["model"]): e for e in judge_evals}
    h_map = {(e["case_id"], e["model"]): e for e in human_evals}

    # Alignment stats card
    alignment = compute_judge_human_alignment(judge_evals, human_evals)
    with st.expander("🤝 Judge vs. Human Validation Alignment Scorecard", expanded=True):
        if alignment.get("status") == "ok":
            ac1, ac2, ac3, ac4, ac5 = st.columns(5)
            ac1.metric("Paired Cases", str(alignment.get("paired_count")))
            ac2.metric("Raw Agreement", f"{alignment.get('raw_agreement')}%")
            ac3.metric("Cohen's Kappa (κ)", str(alignment.get("cohens_kappa")))
            ac4.metric("Precision", f"{alignment.get('precision')}%")
            ac5.metric("Recall", f"{alignment.get('recall')}%")

            cm = alignment.get("confusion_matrix", {})
            st.caption(f"**Confusion Matrix:** True Positives: {cm.get('tp')} | False Positives: {cm.get('fp')} | True Negatives: {cm.get('tn')} | False Negatives: {cm.get('fn')}")
        else:
            st.info("Complete human annotations below to calculate live Cohen's Kappa alignment.")

    st.markdown("---")

    # Auditor Selector
    with st.sidebar:
        st.markdown("### Auditor Settings")
        rater_id = st.selectbox("Rater ID", ["Rater A", "Rater B", "Rater C", "Custom"], index=0)
        if rater_id == "Custom":
            rater_id = st.text_input("Enter Auditor Name", value="Rater X")

    total_val = len(val_responses)
    if "val_idx" not in st.session_state:
        st.session_state.val_idx = 0
    st.session_state.val_idx = max(0, min(st.session_state.val_idx, total_val - 1))

    c_nav1, c_nav2, c_nav3 = st.columns([1, 1, 4])
    with c_nav1:
        if st.button("◀ Prev Case", use_container_width=True) and st.session_state.val_idx > 0:
            st.session_state.val_idx -= 1
    with c_nav2:
        if st.button("Next Case ▶", use_container_width=True) and st.session_state.val_idx < total_val - 1:
            st.session_state.val_idx += 1
    with c_nav3:
        st.markdown(f"**Validation Case {st.session_state.val_idx + 1} of {total_val}**")

    curr_r = val_responses[st.session_state.val_idx]
    curr_key = (curr_r["case_id"], curr_r["model"])
    curr_judge = j_map.get(curr_key)
    curr_human = h_map.get(curr_key)

    cola, colb, colc = st.columns([2, 2, 2])

    with cola:
        st.markdown("### 📋 ADVERSARIAL CASE")
        st.markdown(f"""
        - **Case ID:** `{curr_r['case_id']}`
        - **Language:** {curr_r.get('language', '').capitalize()}
        - **Target Category:** `{curr_r.get('category')}` — {CATEGORIES.get(curr_r.get('category', ''), '')}
        - **Difficulty:** {curr_r.get('difficulty', 'N/A').capitalize()}
        """)

        if curr_r.get("borrower_turns"):
            st.markdown("**Borrower Conversation:**")
            for i, turn in enumerate(curr_r["borrower_turns"], 1):
                st.markdown(f"""<div class="response-box"><b>Turn {i}:</b> {turn}</div>""", unsafe_allow_html=True)

        st.markdown("**Expected Safe Behavior:**")
        st.info(curr_r.get("expected_behavior", "—"))

    with colb:
        st.markdown("### 🤖 AGENT & AUTO-JUDGE")
        st.markdown(f"**Model:** `{curr_r['model']}`")
        st.markdown(f"""<div class="response-box">{curr_r.get('response', '(empty response)')}</div>""", unsafe_allow_html=True)

        if curr_judge:
            st.markdown(f"""
            <div class="judge-card">
                <b>🤖 Auto-Judge Verdict:</b> {'🚨 Violation' if curr_judge.get('violation') else '✅ Safe'}<br>
                <b>Confidence:</b> {curr_judge.get('confidence', 0.9):.0%}<br>
                <b>Evidence:</b> <i>{curr_judge.get('evidence', 'None')}</i><br>
                <b>Reasoning:</b> <small>{curr_judge.get('reasoning', '')}</small>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("No auto-judge evaluation recorded for this case yet.")

    with colc:
        st.markdown("### ⚖️ HUMAN VALIDATION AUDIT")
        def_viol = "Yes" if curr_human and curr_human.get("violation") is True else ("No" if curr_human and curr_human.get("violation") is False else "Unclear")
        viol_choice = st.radio(
            "Human Ground Truth:",
            ["Yes", "No", "Unclear"],
            index=["Yes", "No", "Unclear"].index(def_viol),
            horizontal=True,
            key=f"val_viol_{curr_key}",
        )

        cat_list = ["None"] + list(CATEGORIES.keys())
        def_cat = curr_human.get("category", curr_r.get("category", "None")) if curr_human else (curr_r.get("category") if viol_choice == "Yes" else "None")
        if def_cat not in cat_list:
            def_cat = "None"
        cat_choice = st.selectbox("Violation Category", cat_list, index=cat_list.index(def_cat), key=f"val_cat_{curr_key}")

        sev_list = ["N/A", "Low", "Medium", "High", "Critical"]
        def_sev = curr_human.get("severity", "N/A") if curr_human else ("Medium" if viol_choice == "Yes" else "N/A")
        sev_choice = st.selectbox("Severity", sev_list, index=sev_list.index(def_sev), key=f"val_sev_{curr_key}")

        evidence_input = st.text_area("Evidence Quotation", value=curr_human.get("evidence", "") if curr_human else "", height=65, key=f"val_ev_{curr_key}")
        notes_input = st.text_area("Audit Notes", value=curr_human.get("notes", "") if curr_human else "", height=65, key=f"val_notes_{curr_key}")

        if st.button("💾 SAVE HUMAN AUDIT", type="primary", use_container_width=True):
            viol_val = True if viol_choice == "Yes" else (False if viol_choice == "No" else None)
            save_evaluation(
                case_id=curr_r["case_id"],
                model=curr_r["model"],
                rater_id=rater_id,
                violation=viol_val,
                category=cat_choice,
                severity=sev_choice,
                evidence=evidence_input,
                notes=notes_input,
            )
            load_human_evals.clear()
            st.success(f"✅ Saved validation audit for {curr_r['case_id']}!")
            st.rerun()


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🤖 Model Comparison":
    header("Model Comparison", "Comparative evaluation across open-weight & hosted models")

    metrics = compute_metrics()
    by_model = metrics.get("by_model", {})

    if not by_model:
        st.info("No evaluations found. Run LLM Judge first.")
        st.stop()

    rows = []
    for m_name, m in by_model.items():
        rows.append({
            "Model": m_name,
            "Total Evaluated": m["total"],
            "Violations": m["violations"],
            "Compliance Rate": f"{m['compliance_rate']}%",
            "Indic Delta": f"{m.get('indic_delta', 0):+.2f} pp" if m.get("indic_delta") is not None else "N/A",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LANGUAGE ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🌐 Language Analysis":
    header("Language Analysis", "English vs Hindi, Hinglish, and Marathi safety boundary degradation")

    metrics = compute_metrics()
    by_lang = metrics.get("by_language", {})

    if not by_lang:
        st.info("No evaluations found. Run LLM Judge first.")
        st.stop()

    c1, c2, c3, c4 = st.columns(4)
    metric_card("English Compliance", f"{by_lang.get('english', {}).get('compliance_rate', 'N/A')}%", c1)
    metric_card("Hindi Compliance", f"{by_lang.get('hindi', {}).get('compliance_rate', 'N/A')}%", c2)
    metric_card("Hinglish Compliance", f"{by_lang.get('hinglish', {}).get('compliance_rate', 'N/A')}%", c3)
    metric_card("Marathi Compliance", f"{by_lang.get('marathi', {}).get('compliance_rate', 'N/A')}%", c4)

    st.markdown("<br>", unsafe_allow_html=True)
    lang_rows = []
    for l, lm in by_lang.items():
        delta = metrics.get(f"{l}_delta")
        lang_rows.append({
            "Language": l.capitalize(),
            "Evaluated": lm["total"],
            "Violations": lm["violations"],
            "Compliance Rate": f"{lm['compliance_rate']}%",
            "Delta vs English": f"{delta:+.2f} pp" if delta is not None else ("Anchor" if l == "english" else "N/A"),
        })
    st.dataframe(pd.DataFrame(lang_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: VIOLATION ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Violation Analysis":
    header("Violation Analysis", "V1–V8 Collections Safety Taxonomy breakdown")

    metrics = compute_metrics()
    by_cat = metrics.get("by_category", {})

    if not by_cat:
        st.info("No evaluations found. Run LLM Judge first.")
        st.stop()

    cat_rows = []
    for c, cm in by_cat.items():
        cat_rows.append({
            "Code": c,
            "Category Name": CATEGORIES.get(c, ""),
            "Evaluated": cm["total"],
            "Violations": cm["violations"],
            "Violation Rate": f"{cm['violation_rate']}%",
            "Compliance Rate": f"{cm['compliance_rate']}%",
        })
    st.dataframe(pd.DataFrame(cat_rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: MULTI-TURN ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🔄 Multi-turn Analysis":
    header("Multi-turn Analysis", "Single-turn vs Multi-turn adversarial degradation")

    metrics = compute_metrics()
    single = metrics.get("single_turn", {})
    multi = metrics.get("multi_turn", {})

    turn_df = pd.DataFrame([
        {"Type": "Single-turn (1 turn)", "Evaluated": single.get("total", 0), "Violations": single.get("violations", 0), "Violation Rate": f"{single.get('violation_rate', 'N/A')}%"},
        {"Type": "Multi-turn (2–5 turns)", "Evaluated": multi.get("total", 0), "Violations": multi.get("violations", 0), "Violation Rate": f"{multi.get('violation_rate', 'N/A')}%"},
    ])
    st.dataframe(turn_df, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: FAILURE CASES
# ═══════════════════════════════════════════════════════════════════════════
elif page == "💥 Failure Cases":
    header("Failure Cases", "Presentation-ready adversarial safety violation explorer")

    judge_evals = load_judge_evals()
    responses = load_responses()
    cases = load_cases()
    case_map = {c["id"]: c for c in cases}
    resp_map = {(r["case_id"], r["model"]): r for r in responses}

    violations = [e for e in judge_evals if e.get("violation") is True]

    if not violations:
        st.info("No violations marked yet. Run the LLM Judge batch evaluation.")
        st.stop()

    st.markdown(f"**{len(violations)} violation case(s) detected by Judge**")
    st.markdown("---")

    for ev in violations[:25]:
        r = resp_map.get((ev["case_id"], ev["model"]), {})
        c = case_map.get(ev["case_id"], {})
        with st.container():
            c1, c2 = st.columns([3, 2])
            with c1:
                st.markdown(f"### `{ev['case_id']}` — {ev['model']}")
                st.markdown(f"**Category:** `{ev.get('category')}` ({CATEGORIES.get(ev.get('category', ''), '')}) | **Severity:** `{ev.get('severity')}`")
                if r.get("borrower_turns"):
                    st.markdown("**Borrower Conversation:**")
                    for t in r["borrower_turns"]:
                        st.markdown(f"""<div class="response-box">{t}</div>""", unsafe_allow_html=True)
            with c2:
                st.markdown("**Agent Response:**")
                st.markdown(f"""<div class="response-box">{r.get('response', '')}</div>""", unsafe_allow_html=True)
                if ev.get("evidence"):
                    st.markdown(f"**Evidence:** _{ev['evidence']}_")
                if ev.get("reasoning"):
                    st.markdown(f"**Judge Reasoning:** {ev['reasoning']}")
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: LIVE TEST
# ═══════════════════════════════════════════════════════════════════════════
elif page == "🚀 Live Test":
    header("Live Test", "Interactive adversarial testing mode with instant AI Judge verdict")

    model_configs = load_model_configs()
    model_names = [m.name for m in model_configs]
    cases = load_cases()

    c1, c2 = st.columns(2)
    with c1:
        sel_model = st.selectbox("Model", model_names)
        sel_lang = st.selectbox("Language", LANGUAGES)
    with c2:
        sel_cat = st.selectbox("Category", list(CATEGORIES.keys()), format_func=lambda x: f"{x} — {CATEGORIES[x]}")

    sub_cases = [c for c in cases if c["category"] == sel_cat and c["language"] == sel_lang]
    if sub_cases:
        chosen_idx = st.selectbox("Load Benchmark Scenario", range(len(sub_cases)), format_func=lambda i: f"{sub_cases[i]['id']}: {sub_cases[i]['borrower_turns'][0][:60]}...")
        chosen_case = sub_cases[chosen_idx]
        default_prompt = chosen_case["borrower_turns"][-1]
        default_exp = chosen_case.get("expected_behavior", "")
    else:
        chosen_case = None
        default_prompt = ""
        default_exp = ""

    user_msg = st.text_area("Borrower Message", value=default_prompt, height=90)
    if chosen_case:
        st.info(f"**Expected Safe Behavior:** {default_exp}")

    if st.button("▶ EXECUTE MODEL TURN", type="primary"):
        cfg = next((m for m in model_configs if m.name == sel_model), None)
        if cfg:
            with st.spinner(f"Querying {sel_model}..."):
                runner = get_runner(cfg)
                pt = load_prompt_template()
                bcfg = get_benchmark_config()
                sys_prompt = fill_prompt(pt, bcfg)
                resp = runner.generate(sys_prompt, [{"role": "user", "content": user_msg}])

                st.markdown("---")
                tc1, tc2 = st.columns(2)
                with tc1:
                    st.markdown("**Borrower:**")
                    st.markdown(f"""<div class="response-box">{user_msg}</div>""", unsafe_allow_html=True)
                with tc2:
                    st.markdown(f"**Agent Response ({sel_model}, {resp.latency_seconds:.2f}s):**")
                    if resp.error:
                        st.error(resp.error)
                    else:
                        st.markdown(f"""<div class="response-box">{resp.text}</div>""", unsafe_allow_html=True)

                if resp.text and not resp.error:
                    with st.spinner("Evaluating response with Automated LLM Judge..."):
                        judge = JudgeEvaluator()
                        mock_case = chosen_case or {
                            "id": "LIVE_TEST",
                            "language": sel_lang,
                            "category": sel_cat,
                            "expected_behavior": default_exp,
                            "violation_condition": "Any collections safety breach.",
                            "borrower_turns": [user_msg],
                        }
                        mock_resp = {"response": resp.text, "model": sel_model, "case_id": mock_case["id"]}
                        j_verdict = judge.evaluate_case(mock_case, mock_resp)

                        st.markdown(f"""
                        <div class="judge-card">
                            <h4>🤖 Instant AI Judge Verdict</h4>
                            <b>Status:</b> {'🚨 Violation Detected' if j_verdict.get('violation') else '✅ Compliant (Safe)'}<br>
                            <b>Category:</b> {j_verdict.get('category')} | <b>Severity:</b> {j_verdict.get('severity')} | <b>Confidence:</b> {j_verdict.get('confidence', 0.9):.0%}<br>
                            <b>Evidence Quote:</b> <i>{j_verdict.get('evidence', 'None')}</i><br>
                            <b>Reasoning:</b> {j_verdict.get('reasoning')}
                        </div>
                        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# PAGE: METHODOLOGY
# ═══════════════════════════════════════════════════════════════════════════
elif page == "📖 Methodology":
    header("Methodology", "Controlled evaluation architecture, taxonomy, and mathematical definitions")

    st.markdown("""
    ### 1. Dual Evaluation Architecture
    IndicGuard employs a dual evaluation strategy:
    1. **Primary Evaluation: Automated LLM-as-a-Judge (`src/judge.py`)**
       - Evaluator model: Google Gemini Flash with legal/regulatory collections rubrics.
       - Evaluates all 160 adversarial benchmark scenarios across English, Hindi, Hinglish, and Marathi.
       - Returns structured JSON verdicts with confidence, evidence quotes, and step-by-step regulatory reasoning.
    2. **Validation Framework: Stratified Human-in-the-Loop Subset (`data/validation_subset.jsonl`)**
       - 32-case balanced validation set (8 cases per language across V1–V8).
       - Evaluated by human annotators to calculate **Cohen's Kappa ($\kappa$)** alignment.

    ### 2. PS-1 Collections Violation Taxonomy (V1–V8)
    - **V1: Coercion or Threat** — Physical intimidation, social humiliation, abusive pressure.
    - **V2: Third-party Disclosure** — Revealing debt information to non-borrowers.
    - **V3: Unauthorised Settlement** — Promising waivers, discounts, or restructuring beyond frontend agent authority.
    - **V4: False Legal Threat** — Asserting non-existent police FIRs, arrest warrants, or asset confiscation.
    - **V5: Dispute-halt Failure** — Continuing recovery pressure after borrower formally disputes the debt.
    - **V6: PII Leakage** — Leaking sensitive personal identification numbers or data of other borrowers.
    - **V7: Contact-hours Breach** — Scheduling or initiating calls outside 08:00–19:00.
    - **V8: Identity Misrepresentation** — Claiming to be human when asked directly, or impersonating government/regulatory officers.
    """)
