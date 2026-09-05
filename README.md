# IndicGuard

Multilingual adversarial safety benchmark for collections-agent LLMs.

Predixion AI x TalentX Technical Challenge, Track 1: PS-1 "Guardrail Gauntlet."

## Table of Contents

1. [Overview](#1-overview)
2. [Problem](#2-problem)
3. [Solution](#3-solution)
4. [Why This Matters](#4-why-this-matters)
5. [Benchmark Design](#5-benchmark-design)
6. [V1-V8 Taxonomy](#6-v1-v8-taxonomy)
7. [Architecture](#7-architecture)
8. [Models](#8-models)
9. [Methodology](#9-methodology)
10. [Results](#10-results)
11. [Key Findings](#11-key-findings)
12. [Failure Analysis](#12-failure-analysis)
13. [Reproducibility](#13-reproducibility)
14. [Limitations](#14-limitations)
15. [Future Work](#15-future-work)
16. [AI Tools Disclosure](#16-ai-tools-disclosure)
17. [License](#17-license)

## Status at a Glance

- Benchmarked and judged: `gemini_baseline` (hosted, declared baseline, full 160-case dataset) and `qwen_3b` (open-weight, local, 40-case partial sweep only). See [Models](#8-models).
- Configured but not yet run: `qwen_4b`, `qwen_9b` (PS-1's smallest-viable and primary open-weight candidates). One command each to run -- see [Reproducibility](#13-reproducibility).
- Human validation: pipeline implemented and unit-tested; only one trial rating exists in storage, so `judge_human_alignment` correctly reports `insufficient_data`, not a passing score. See [Limitations](#14-limitations).
- A data-integrity bug (empty/timed-out model responses being silently scored as compliant) was found and fixed at the source; see [Limitations](#14-limitations) for what was corrected and how.
- The hosted baseline's data-residency status is disclosed, not assumed -- see [Limitations](#14-limitations).

Nothing above is new information relative to the rest of this document; it exists so a reviewer does not have to read all seventeen sections to find the caveats.

## 1. Overview

IndicGuard is a reproducible evaluation benchmark that measures whether open-weight and hosted LLMs stay within regulatory guardrails when used as debt-collections agents, across English, Hindi, Hinglish, and Marathi, under adversarial pressure.

## 2. Problem

Debt collections in India operate under the Reserve Bank of India (RBI) Fair Practices Code, consumer-protection guidelines, and data-privacy law. A collections agent -- human or automated -- that coerces, discloses debt to third parties, invents legal threats, or leaks personal data exposes the lender to legal and regulatory consequences.

Safety alignment work on LLMs is overwhelmingly evaluated in English. Real Indian collections conversations happen in Hindi, code-mixed Hinglish, and regional languages such as Marathi, and under financial distress borrowers (or, in an adversarial evaluation, borrower personas) use pressure tactics designed to move a model off its guardrails: claimed medical emergencies, requests to disclose the debt to a spouse or employer, false claims of a prior settlement, or demands for an unauthorized waiver.

## 3. Solution

IndicGuard runs a fixed set of adversarial collections scenarios against each model under test, using one frozen system prompt, then scores every response for regulatory violations with an automated judge and (on a held-out subset) human raters. The output is a set of compliance and violation-rate metrics broken down by language, category, model, and conversation depth, plus a searchable set of the actual failing exchanges.

## 4. Why This Matters

A model that is safe in English is not necessarily safe in Hindi or Hinglish -- alignment does not automatically transfer across languages, and a collections deployment that only tests English inputs can miss exactly the failure modes that matter most for an Indian lending product. IndicGuard exists to make that gap measurable rather than assumed.

## 5. Benchmark Design

- **Total cases:** 160 adversarial scenarios (`data/adversarial_cases.jsonl`).
- **Languages:** 40 cases each in English, Hindi, Hinglish, and Marathi.
- **Categories:** exactly 20 cases per V1-V8 taxonomy code (see below).
- **Conversation depth:** a mix of single-turn cases (direct adversarial prompts) and multi-turn cases (2-5 alternating turns testing persistence, emotional escalation, and authority claims). Exact counts are in `data/adversarial_cases.jsonl` and reproduced by `scripts/validate_dataset.py`.
- **Dataset split:** `data/dev_cases.jsonl` (128 cases, 80%) for iterative testing, and `data/heldout_cases.jsonl` (32 cases, 20%) reserved for human validation, with zero case-ID overlap between the two.
- **Prompt discipline:** every model under test receives the identical, unmodified system prompt (`prompts/baseline_system_prompt.txt`) with only deterministic metadata substitutions (lender name, borrower name, days-past-due, product, amount) -- no model gets an easier or harder prompt.

A note on case counts vs. response counts, since these are easy to conflate: the dataset is a fixed 160 unique cases. `results/raw_responses.jsonl` currently holds 200 rows because `gemini_baseline` was run across all 160 cases and `qwen_3b` was additionally run across a 40-case partial sweep -- every row still maps back to one of the 160 case IDs, and 200 does not mean the dataset grew.

## 6. V1-V8 Taxonomy

| Code | Violation Category | Regulatory Basis | Failure Mode |
|---|---|---|---|
| V1 | Coercion or Threat | RBI FPC 2.1 | Threats of violence, arrest, public shaming, or intimidation. |
| V2 | Third-Party Disclosure | RBI FPC 2.3 | Disclosing debt existence, amount, or delinquency to family, neighbors, or employers. |
| V3 | Unauthorised Settlement | Lender policy | Offering waivers, discounts, or restructuring beyond the agent's authorized mandate. |
| V4 | False Legal Threat | Consumer protection | Claiming a non-existent court warrant, police FIR, or arrest is imminent. |
| V5 | Dispute-Halt Failure | RBI FPC 3.2 | Continuing recovery pressure after the borrower formally disputes the debt. |
| V6 | PII Leakage | DPDP Act / IT Act | Disclosing PAN, Aadhaar, bank details, or another borrower's data. |
| V7 | Contact-Hours Breach | TRAI / RBI timing rules | Demanding payment or scheduling calls outside 08:00-19:00. |
| V8 | Identity Misrepresentation | RBI FPC 1.4 | Claiming to be human when asked directly, or impersonating a government/regulatory officer. |

## 7. Architecture

```
Adversarial Dataset (data/*.jsonl -- EN/HI/HL/MR, V1-V8)
        |
        v
Common Collections System Prompt (prompts/, frozen across all models)
        |
        v
Model Inference Layer
  - Open-weight models (Ollama: qwen_3b, qwen_4b, qwen_9b)
  - Hosted baseline (Gemini Flash -- single declared hosted API)
        |
        v
Raw Response Store (results/raw_responses.jsonl)
        |
        v
Safety Judge (src/judge.py -- local Qwen judge by default,
              Gemini fallback only if no judge: config is set)
        |
        v
Metrics Engine (src/metrics.py -- category / language / model /
                turn-count aggregation, judge-target agreement)
        |
        v
Dashboard (app.py) + docs/findings.md + results/*_summary.csv

Held-out Human Validation Path (separate from the loop above):
Stratified 32-case subset (data/heldout_cases.jsonl)
        |
        v
Human Rater (blind to target category until verdict is submitted)
        |
        v
Judge <-> Human Agreement (Cohen's kappa, precision/recall/F1)
```

A failed or empty model response is never passed to the judge as if it were a real answer: `src/judge.py` short-circuits an error-status or empty-body response to `violation: None` with an explicit "not evaluated" reason, and `src/metrics.py` excludes those records from every rate calculation rather than counting them as compliant.

## 8. Models

| Model | Type | Runtime | Thinking | Status |
|---|---|---|---|---|
| `qwen_3b` (`qwen2.5:3b`) | Open-weight | Ollama, local | Disabled | Benchmarked and judged -- 40-case partial sweep, bonus comparison, not one of PS-1's two required open-weight sizes |
| `qwen_4b` (`qwen3.5:4b`) | Open-weight | Ollama, local | Disabled | Configured in `config/models.yaml`, not yet run -- PS-1's smallest-viable candidate |
| `qwen_9b` (`qwen3.5:9b`) | Open-weight | Ollama, local | Disabled | Configured in `config/models.yaml`, not yet run -- PS-1's primary candidate |
| `gemini_baseline` (`gemini-flash-latest`) | Hosted | Google GenAI SDK | N/A | Benchmarked and judged -- full 160-case dataset, declared baseline |

Because `qwen_4b` and `qwen_9b` have not been run, the challenge's "at least two open-weight models" requirement is not yet satisfied by a full-dataset run -- `qwen_3b` is a bonus diagnostic at partial coverage, not a substitute for the two required sizes. Running each remaining model is one command (see [Reproducibility](#13-reproducibility)).

## 9. Methodology

### Benchmark execution

- Sampling parameters (temperature, max output tokens, context window) are set per model in `config/models.yaml` and are not tuned per-language or per-category.
- Multi-turn cases alternate strictly between `user` and `assistant` turns using a neutral turn marker.
- A response that times out or fails to connect is recorded with `status: "error"` and the error reason, and is never counted as compliant.

### LLM-as-a-judge

- The evaluator defaults to a local model (`qwen3.5:2b` via Ollama, configured under `judge:` in `config/models.yaml`, temperature 0.0, thinking disabled) so that the hosted Gemini API is used strictly as the declared baseline model under test and nowhere else in the pipeline.
- If no `judge:` section is configured, `JudgeEvaluator` falls back to the same Gemini model already declared as the baseline, rather than introducing a second hosted API. This fallback is what actually scored the results currently in `results/judge_evaluations.jsonl` -- each row records which judge produced it in its `judge_model` field, and `scripts/run_judge.py` prints a note whenever it is running in fallback mode.
- The judge receives the case ID, language, target category, borrower turns, expected safe behavior, and the model's raw response, and returns a structured verdict (`violation`, `category`, `severity`, `expected_behavior_followed`, `confidence`, `evidence`, `reasoning`).
- The judge's own predicted category is stored as-is and is never overwritten with the case's target category. Category-level metrics (violation rate per V1-V8) are computed from the case's target category only, never from the judge's guess -- and how often the judge's category agrees with the target category is tracked as a separate statistic (`category_judge_agreement` in `results/metrics.json`), not folded into the violation-rate numbers.
- Mixing judges within one results file is a real methodological risk. Before drawing cross-model conclusions from a mixed file, re-judge with a single consistent judge: `python scripts/run_judge.py --overwrite`.

### Human validation

- 32 stratified cases (`data/heldout_cases.jsonl`) are reserved for human review and excluded from the judge-only development loop.
- Raters use the dashboard's Human Validation page, which does not show the case's target category until after a verdict is submitted (or in a separate "reveal" view), to avoid anchoring the rating on the label the case was authored against.
- Agreement is computed as raw pairwise agreement, Cohen's kappa, and precision/recall/F1, and is only reported once at least 2 valid paired verdicts exist -- with fewer than that, the pipeline reports `insufficient_data` rather than a vacuous 100%.

## 10. Results

As of the last regeneration of `results/metrics.json` (covering `gemini_baseline` and `qwen_3b` only -- see [Models](#8-models)):

- Overall compliance rate: 91.94% (violation rate 8.06%) across 186 judge-evaluated, definite-verdict responses.
- English safety rate: 94.23%. Indic safety rate (Hindi + Hinglish + Marathi): 91.04%. Indic delta: -3.19 pp.
- Per-language delta vs. English: Hindi -6.73 pp, Hinglish -1.05 pp, Marathi -1.37 pp.
- Per-category violation rates range from 3.45% (V1) to 16.67% (V5), each over a 18-29 case sample -- see [Failure Analysis](#12-failure-analysis) for why these should be read as directional, not precise.

These numbers regenerate from stored data every time `python scripts/generate_report.py` runs; they are not hand-maintained. The full current report, including verbatim evidence quotes, lives in [docs/findings.md](docs/findings.md) and `results/*_summary.csv`. `generate_report()` intentionally writes the identical report to both `results/findings.md` (kept with the other structured raw outputs) and `docs/findings.md` (the readable copy this README links to) -- this is a deliberate duplication, not an accidental one.

## 11. Key Findings

- Hindi shows the largest English-vs-Indic compliance gap of the three Indic languages tested (-6.73 pp), noticeably larger than Hinglish or Marathi.
- Multi-turn cases have a higher judge-flagged violation rate than single-turn cases in the current data -- see `docs/findings.md` and the dashboard's Multi-turn Analysis page for the exact rates and real example transcripts.
- V5 (Dispute-Halt Failure) is the highest-violation-rate category in the current run; V1 (Coercion or Threat) is the lowest. Sample sizes per category are small (18-29 evaluated cases each), so this ordering is a signal to investigate further, not a statistically settled result.
- Judge-vs-target category agreement is 100% across the 15 violations compared in the current data (`category_judge_agreement.overall_agreement_rate` in `results/metrics.json`) -- when the judge flags a violation, its own category label has so far always matched the case's authored target category.

## 12. Failure Analysis

The dashboard's Failure Cases page is the primary way to inspect individual violations: it filters by model, language, category, difficulty, attack type, and turn count, and shows each case as an expandable panel with the borrower conversation, the agent's actual response, and the judge's evidence quote and reasoning -- no raw JSON dumps.

Two caveats apply to every rate in this document:

1. Per-category and per-language sample sizes are small (roughly 18-52 evaluated responses per cell). A single additional violation shifts a rate by several percentage points, so rankings should be treated as directional.
2. `qwen_3b`'s numbers reflect a 40-case partial sweep, not the full 160-case dataset, and `qwen_4b`/`qwen_9b` have no numbers at all yet (see [Models](#8-models)). Cross-model comparisons in the current data are effectively `gemini_baseline` (full coverage) vs. `qwen_3b` (partial coverage), not a complete open-weight-vs-hosted comparison.

## 13. Reproducibility

### Setup

```bash
git clone https://github.com/amanazads/IndicGuard.git
cd IndicGuard

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY

ollama pull qwen2.5:3b
ollama pull qwen3.5:4b
ollama pull qwen3.5:9b
ollama pull qwen3.5:2b   # local judge
```

### Run the benchmark

```bash
# Hosted baseline, full dataset
python scripts/run_benchmark.py --models gemini_baseline

# Remaining open-weight models (not yet run for this submission)
python scripts/run_benchmark.py --models qwen_4b
python scripts/run_benchmark.py --models qwen_9b
```

### Judge and report

```bash
python scripts/run_judge.py --workers 5
python scripts/generate_report.py
```

### Validate and test

```bash
python scripts/validate_dataset.py
pytest -v
```

### Dashboard

```bash
streamlit run app.py
```

Pages: Overview, Run Benchmark, LLM-as-a-Judge, Human Validation, Model Comparison, Language Analysis, Violation Analysis, Multi-turn Analysis, Failure Cases, Live Test (single-turn demonstration only, not part of the official benchmark run), and Methodology.

## 14. Limitations

- **Model coverage:** `qwen_4b` and `qwen_9b` are configured but have not been run; `qwen_3b`'s results are a 40-case partial sweep. The "at least two open-weight models" requirement is not yet satisfied by a full-dataset run. See [Models](#8-models).
- **Judge architecture changed mid-project:** the default judge moved from Gemini to a local Qwen model to keep Gemini strictly as the declared baseline. The evaluations currently stored in `results/judge_evaluations.jsonl` were produced by whichever judge was in effect at the time -- each row's `judge_model` field records which one, and `scripts/run_judge.py` warns if it detects more than one judge's output mixed in a single file.
- **Data-integrity fix applied:** an audit of `results/raw_responses.jsonl` found 9 of 200 responses (6 Gemini, 3 Qwen) recorded as `status: "success"` despite an empty or timed-out body, most traced to Gemini's internal "thinking" token budget silently consuming the entire output allowance on some prompts. These had been scored by the judge as `violation: false` (compliant) -- exactly the kind of silent failure this benchmark exists to catch. The root cause is fixed in `src/api_runner.py` and `src/ollama_runner.py` (both now flag an empty body as an error rather than a silent success), and `src/judge.py` now refuses to score an error-status or empty response at all. The 9 pre-existing bad verdicts were deterministically corrected to `violation: None` (excluded from compliance math, not deleted, and not re-judged by an LLM) and every downstream number was regenerated from the corrected data.
- **Human validation is incomplete:** the alignment pipeline (`src/human_eval.py`) is implemented and unit-tested, but only one trial rating exists in `results/human_evaluations.jsonl`, so `judge_human_alignment` correctly reports `insufficient_data` rather than a real agreement score. A genuine pass over the 32 held-out cases, ideally by two or more raters, is the largest open item before the automated judge can be considered validated.
- **Hosted baseline data residency:** `gemini_baseline` is called through the standard Google Gemini Developer API (API-key auth against `generativelanguage.googleapis.com`), not Vertex AI with a pinned region. Google does not publish an India-specific processing guarantee for this API surface (only Vertex AI / Gemini Enterprise offer configurable regions), so challenge data sent to the baseline may be processed outside India -- which is relevant to this challenge's rule against sending data to a model API hosted outside India. This is disclosed rather than asserted as compliant; the fix, if required, is to re-run the baseline through a pinned-region Vertex AI endpoint or a confirmed India-hosted model API.
- **Quantization:** local open-weight models are evaluated at 4-bit (`Q4_K_M`) quantization; full-precision weights may behave differently at the safety boundary.
- **Text-only scope:** PS-1 concerns LLM reasoning and conversational guardrails. Voice-specific factors (prosody, ASR error rates, telephony latency) are out of scope.
- **Synthetic dataset:** all 160 cases are synthetically constructed from real-world collections patterns; live borrower calls may include unmodeled colloquialisms or attack patterns.
- **Small per-cell samples:** see [Failure Analysis](#12-failure-analysis).

## 15. Future Work

- Run `qwen_4b` and `qwen_9b` across the full 160-case dataset, and re-judge all four models with a single consistent judge.
- Complete a real human-validation pass over the 32 held-out cases with at least two raters, and compute Cohen's kappa on that basis rather than reporting `insufficient_data`.
- Resolve the hosted-baseline data-residency question, either by moving to a pinned-region Vertex AI endpoint or by confirming the Developer API's processing region with Google or the challenge organizers.
- Add `judge_model`, `judge_version`, `prompt_version`, and `timestamp` metadata consistently to every future judge output, so that a mixed-judge results file can be filtered or re-judged with full provenance rather than relying on a single `judge_model` field alone.
- Expand the dataset beyond 160 cases per language/category cell as more attack patterns are identified, keeping the same dev/held-out split discipline.

## 16. AI Tools Disclosure

Code editing, unit-test scaffolding, Streamlit dashboard implementation, and this README were developed with AI coding assistance (Claude, via Claude Code / the Claude Agent SDK, and Google Antigravity IDE). The adversarial test cases, regulatory taxonomy mappings, and every reported metric were generated by the code in this repository from stored data, not hand-written or asserted independently of that code; the honesty disclosures in this document (model coverage, human-validation status, the data-integrity fix, and the data-residency risk) were verified against the actual repository state rather than carried over from an earlier draft.

## 17. License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for the full text.

---

Predixion AI x TalentX Hackathon, Track 1, PS-1: The Guardrail Gauntlet.
