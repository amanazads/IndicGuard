# IndicGuard — Research Findings & Evaluation Report

**Benchmark:** IndicGuard Multilingual Adversarial Safety Benchmark for Collections LLMs  
**Track:** Predixion AI × TalentX — Track 1 PS-1 Guardrail Gauntlet  

---

## 1. Overall Findings

- **Evaluated Sample Size:** 194 evaluation verdicts (across 200 model responses).
- **Overall Compliance Rate:** 92.27% (Violation Rate: 7.73%).
- **English Safety Baseline:** 94.23% compliance.
- **Indic Safety Baseline:** 91.55% compliance.
- **English → Indic Safety Delta:** -2.68 percentage points.

## 2. Model Comparison

| Model | Total Evaluated | Violations | Compliance Rate | Indic Delta |
|-------|----------------|------------|-----------------|-------------|
| `gemini_baseline` | 158 | 1 | 99.37% | -0.85 pp |
| `qwen_3b` | 36 | 14 | 61.11% | -20.83 pp |

## 3. Language Differences

| Language | Total Evaluated | Violations | Compliance Rate | Delta vs English |
|----------|----------------|------------|-----------------|------------------|
| English | 52 | 3 | 94.23% | Anchor |
| Hindi | 51 | 6 | 88.24% | -5.99 pp |
| Hinglish | 46 | 3 | 93.48% | -0.75 pp |
| Marathi | 45 | 3 | 93.33% | -0.90 pp |

## 4. Category Failures (V1–V8)

| Category | Name | Evaluated | Violations | Violation Rate |
|----------|------|-----------|------------|----------------|
| **V1** | Coercion or Threat | 32 | 1 | 3.12% |
| **V2** | Third-party Disclosure | 22 | 2 | 9.09% |
| **V3** | Unauthorised Settlement | 23 | 2 | 8.7% |
| **V4** | False Legal Threat | 24 | 2 | 8.33% |
| **V5** | Dispute-halt Failure | 22 | 3 | 13.64% |
| **V6** | PII Leakage | 24 | 1 | 4.17% |
| **V7** | Contact-hours Breach | 24 | 2 | 8.33% |
| **V8** | Identity Misrepresentation | 23 | 2 | 8.7% |

## 5. Multi-turn Findings

- **Single-turn violation rate:** 1.3% (1/77 cases)
- **Multi-turn violation rate:** 11.97% (14/117 cases)

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
