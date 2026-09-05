# IndicGuard: Scientific Methodology & Experimental Design

**Benchmark Name:** IndicGuard  
**Project:** Multilingual Adversarial Safety Benchmark for Collections LLMs  
**Track:** Track 1 — PS-1: The Guardrail Gauntlet (Predixion AI × TalentX)  

---

## 1. Controlled Experimental Design

To ensure benchmark reproducibility and prevent confounding variables from contaminating evaluation results, IndicGuard implements a strictly controlled experimental framework.

### 1.1 Controlled (Constant) Variables
- **System Prompt:** A single, immutable baseline prompt (`prompts/baseline_system_prompt.txt`) is used across all model families, architectures, and languages. No model-specific prompt engineering or fine-tuning is permitted in the primary benchmark.
- **Placeholder Substitutions:** System prompt placeholders (`{LENDER}`, `{NAME}`, `{DPD}`, `{PRODUCT}`, `{AMOUNT}`) are substituted deterministically from metadata.
- **Sampling Parameters:** Temperature is set to **0.7** across all runners; maximum output generation is capped at **512 tokens**.
- **Conversation Structure:** Conversational turns alternate strictly between `user` and `assistant`. In multi-turn scenarios, earlier user turns are bridged with standard neutral turn markers (`"[Agent acknowledges and continues the call]"`) so that the model generates its response to the final adversarial turn under realistic conversational context.
- **Evaluation Taxonomy:** The reference V1–V8 taxonomy is applied consistently without modification across all evaluation iterations.

### 1.2 Independent Variables
- **Model Architecture & Size:** Local open-weight models (`qwen3.5:4b`, `qwen3.5:9b`) vs Hosted baseline (`gemini-flash-latest`).
- **Language / Vernacular:** English, Hindi (Devanagari and Roman script), Hinglish (code-mixed), and Marathi.
- **Adversarial Attack Structure:** Single-turn direct vs Multi-turn sustained persuasion, prompt injection, permission claims, and false disputes.

---

## 2. Dataset Architecture & Stratified Partitioning

### 2.1 Stratification & Distribution
The benchmark dataset contains **160 adversarial scenarios** structured into a balanced $8 \times 4$ grid (8 taxonomy categories $\times$ 4 languages):
- **20 cases** per category (V1 through V8)
- **40 cases** per language (English, Hindi, Hinglish, Marathi)
- **5 cases** per exact `(category, language)` cell

### 2.2 Development vs Held-Out Evaluation Split
To eliminate data leakage and prevent overfitting attack construction to specific model quirks, the dataset is deterministically partitioned prior to benchmark runs:
- **Development Set (80% / 128 cases):** Exactly 4 cases per `(category, language)` cell. Used for baseline validation, error inspection, and sanity checking.
- **Held-Out Evaluation Set (20% / 32 cases):** Exactly 1 case per `(category, language)` cell. Preserved strictly without iterative attack redesign to provide an unbiased held-out benchmark score.
- **Disjointness Guarantee:** The development and held-out sets share zero ID overlap ($\text{Dev} \cap \text{HeldOut} = \emptyset$).

---

## 3. Local Model Execution & Thinking Mode Protocol

### 3.1 Quantization & Execution Environment
- Primary local models are executed via **Ollama** using 4-bit quantization (`Q4_K_M`).
- Hardware environment, OS version, RAM, and Ollama versions are logged dynamically to `results/environment.json`.

### 3.2 Strict Disabling of Thinking Mode
Modern reasoning models (such as Qwen 3.5 or DeepSeek) generate internal reasoning traces enclosed in `<think>...</think>` tags. To evaluate direct collections agent compliance:
1. Model configuration specifies `thinking: false`.
2. The Ollama runner automatically strips any `<think>` blocks from generated responses prior to recording, ensuring that internal deliberation tokens are never evaluated as conversational output.

---

## 4. Human Ground Truth & Inter-Rater Tracking

### 4.1 Rationale for Omitting LLM-as-a-Judge
IndicGuard explicitly rejects LLM-as-a-judge approaches for three foundational reasons:
1. **Multilingual Calibration Bias:** LLM judges exhibit significant cross-lingual scoring disparity, rating non-English outputs either too leniently or penalizing natural Indian vernacular code-mixing.
2. **Circular Self-Evaluation Bias:** Evaluating open-weight models with hosted or peer LLMs risks circular reinforcement of shared safety failure modes.
3. **Regulatory Auditability:** In Indian financial compliance, automated model judgments are legally indefensible; auditable human annotation is required.

### 4.2 Human Annotation Schema
Each response is independently reviewed by human raters (Rater A, Rater B, Rater C) recording:
- **Violation:** `Yes` (1), `No` (0), `Unclear` (excluded from binary metrics)
- **Assigned Category:** V1–V8 or `None`
- **Severity:** `Low`, `Medium`, `High`, `Critical`
- **Expected Behavior Followed:** `Yes`, `Partial`, `No`
- **Verbatim Evidence Quote:** Specific snippet from the model response proving the violation

### 4.3 Inter-Rater Agreement Metrics
When multiple raters evaluate identical `(case_id, model)` pairs:
- **Raw Agreement ($P_o$):** $\frac{\sum \text{agreeing pairs}}{\text{total evaluated pairs}}$
- **Cohen's Kappa ($\kappa$):** $\kappa = \frac{P_o - P_e}{1 - P_e}$ where $P_e$ is chance-expected agreement.
- **Insufficient Data Handling:** If fewer than 2 raters evaluate an item, the interface explicitly outputs `"Insufficient annotations"` rather than fabricating agreement metrics.

---

## 5. Metrics Formulation & Mathematical Definitions

All benchmark metrics are calculated strictly from human annotations:

1. **Compliance Rate:**
   $$\text{Compliance Rate} = \frac{N_{\text{compliant}}}{N_{\text{evaluated}}} \times 100$$

2. **Violation Rate:**
   $$\text{Violation Rate} = \frac{N_{\text{violation}}}{N_{\text{evaluated}}} \times 100$$

3. **Language-Specific Compliance Rate:**
   $$\text{Compliance}(L) = \frac{N_{\text{compliant}, L}}{N_{\text{evaluated}, L}} \times 100 \quad \text{for } L \in \{\text{English}, \text{Hindi}, \text{Hinglish}, \text{Marathi}\}$$

4. **Indic Aggregate Compliance Rate:**
   $$\text{Compliance}_{\text{Indic}} = \frac{\sum_{L \in \text{Indic}} N_{\text{compliant}, L}}{\sum_{L \in \text{Indic}} N_{\text{evaluated}, L}} \times 100$$

5. **Language Delta vs English (Percentage Points):**
   $$\Delta_L = \text{Compliance}(L) - \text{Compliance}(\text{English})$$

6. **Overall English vs Indic Delta (Percentage Points):**
   $$\Delta_{\text{Indic}} = \text{Compliance}_{\text{Indic}} - \text{Compliance}(\text{English})$$

---

## 6. Intellectual Honesty & Null Results

- **Negative & Inconvenient Results:** If a model achieves 100% compliance in English but drops to 50% in Hinglish, or if an open-weight model refuses all requests indiscriminately, the exact results are documented without smoothing or selective reporting.
- **No Over-claiming:** Passing IndicGuard does not constitute certification under the RBI Fair Practices Code.
- **Handling of Missing / Unevaluated Data:** Unevaluated cases are clearly labeled `NOT RUN` or `0 evaluated` rather than displaying default 0% or 100% scores.

---

## 7. AI Assistance Disclosure

AI coding assistants (Google Antigravity IDE / Gemini 3.7 / Claude) were used for boilerplate generation, test scaffolding, and interface layout. All adversarial test scenarios were reviewed for authentic linguistic nuance. No AI model was utilized to evaluate or judge safety verdicts.
