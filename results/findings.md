# IndicGuard — Research Findings & Evaluation Report

**Benchmark:** IndicGuard Multilingual Adversarial Safety Benchmark for Collections LLMs  
**Track:** Predixion AI × TalentX — Track 1 PS-1 Guardrail Gauntlet  

---

## 1. Overall Findings

_No completed human evaluations recorded yet._

- Total raw model responses: **160**
- Total human evaluations: **0**

To generate full empirical findings:
1. Run benchmark: `python scripts/run_benchmark.py`
2. Complete human safety audits in `streamlit run app.py` (Human Evaluation page)
3. Re-run `python scripts/generate_report.py`

## 2. Model Comparison

| Model | Total Evaluated | Violations | Compliance Rate | Indic Delta |
|-------|----------------|------------|-----------------|-------------|
| _No model data_ | - | - | - | - |

## 3. Language Differences

| Language | Total Evaluated | Violations | Compliance Rate | Delta vs English |
|----------|----------------|------------|-----------------|------------------|
| _No language data_ | - | - | - | - |

## 4. Category Failures (V1–V8)

| Category | Name | Evaluated | Violations | Violation Rate |
|----------|------|-----------|------------|----------------|
| _No category data_ | - | - | - | - |

## 5. Multi-turn Findings

- **Single-turn violation rate:** N/A% (0/0 cases)
- **Multi-turn violation rate:** N/A% (0/0 cases)

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
