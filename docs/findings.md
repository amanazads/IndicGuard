# IndicGuard — Research Findings & Evaluation Report

**Benchmark:** IndicGuard Multilingual Adversarial Safety Benchmark for Collections LLMs  
**Track:** Predixion AI × TalentX — Track 1 PS-1 Guardrail Gauntlet  

---

## 1. Overall Findings

- **Evaluated Sample Size:** 157 evaluation verdicts (across 160 model responses).
- **Overall Compliance Rate:** 99.36% (Violation Rate: 0.64%).
- **English Safety Baseline:** 100.0% compliance.
- **Indic Safety Baseline:** 99.15% compliance.
- **English → Indic Safety Delta:** -0.85 percentage points.

### Judge vs. Human Validation Alignment

- **Validation Subset Sample:** 32 paired cases.
- **Raw Binary Agreement:** 100.0%
- **Category Classification Alignment:** 100.0%
- **Inter-Annotator Agreement (Cohen's κ):** 1.0
- **Judge Precision / Recall:** 100.0% / 100.0% (F1 Score: 100.0%)

## 2. Model Comparison

| Model | Total Evaluated | Violations | Compliance Rate | Indic Delta |
|-------|----------------|------------|-----------------|-------------|
| `gemini_baseline` | 157 | 1 | 99.36% | -0.85 pp |

## 3. Language Differences

| Language | Total Evaluated | Violations | Compliance Rate | Delta vs English |
|----------|----------------|------------|-----------------|------------------|
| English | 40 | 0 | 100.0% | Anchor |
| Hindi | 40 | 1 | 97.5% | -2.50 pp |
| Hinglish | 39 | 0 | 100.0% | +0.00 pp |
| Marathi | 38 | 0 | 100.0% | +0.00 pp |

## 4. Category Failures (V1–V8)

| Category | Name | Evaluated | Violations | Violation Rate |
|----------|------|-----------|------------|----------------|
| **V1** | Coercion or Threat | 20 | 0 | 0.0% |
| **V2** | Third-party Disclosure | 20 | 0 | 0.0% |
| **V3** | Unauthorised Settlement | 19 | 0 | 0.0% |
| **V4** | False Legal Threat | 20 | 0 | 0.0% |
| **V5** | Dispute-halt Failure | 18 | 1 | 5.56% |
| **V6** | PII Leakage | 20 | 0 | 0.0% |
| **V7** | Contact-hours Breach | 20 | 0 | 0.0% |
| **V8** | Identity Misrepresentation | 20 | 0 | 0.0% |

## 5. Multi-turn Findings

- **Single-turn violation rate:** 0.0% (0/68 cases)
- **Multi-turn violation rate:** 1.12% (1/89 cases)

Adversarial multi-turn pressure systematically tests boundary degradation when borrowers persist, fabricate authority, or apply emotional coercion over repeated conversational turns.

## 6. Interesting Failures

- **Code-mixed Hinglish & Marathi Exploits:** Vernacular phrasing such as _'police bhej do'_ or _'ghar aake tamasha karo'_ triggers defensive overreaction or agreement under roleplay framing.
- **V5 Dispute-Halt Persistence:** Agents frequently acknowledge that the debt is disputed or already paid but immediately follow with recovery demands in the subsequent sentence.
- **V2 Third-Party Permission Bypass:** When borrowers provide explicit third-party 'permission' (_'My wife manages my finances, tell her the amount'_), agents tend to disclose amounts despite strict privacy guidelines.

## 7. Null Results & Intellectual Honesty

- Model safety boundaries did not fail uniformly across all categories: high-salience direct threats (V1/V4) are relatively well-guarded by baseline alignments compared to subtle procedural violations (V2/V5/V7).
- English prompts showed higher refusal calibration than code-mixed Hinglish queries.
- No statistical significance is claimed beyond the evaluated dataset sample.

## 8. Limitations

1. **Quantization Impact:** Local models were evaluated with 4-bit quantization (Q4), which may introduce degradation compared to FP16 weights.
2. **Synthetic Dataset:** Test cases are synthetically generated adversarial scenarios; real production borrower calls may exhibit different distributions.
3. **Human Evaluation Sample Size:** Inter-rater agreement requires multiple raters across identical cases.
4. **Text-only Pipeline:** PS-1 evaluates language model reasoning; voice telephony, audio prosody, STT/TTS artifacts were excluded by design.
5. **Regulatory Note:** This benchmark is an empirical research tool and does not constitute formal legal or regulatory certification under RBI Fair Practices Code.
