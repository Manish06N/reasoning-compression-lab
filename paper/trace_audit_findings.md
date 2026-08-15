# Structured Trace Audit Report ($\ge 200$ Stratified Samples)

**Audit Scope:** 200 stratified mathematical reasoning problems across MATH-500 difficulty levels.
**Evaluated Checkpoints:** `Qwen-7B` and `Llama-8B` across BF16, FP8, AWQ-4, and GPTQ-4.

---

## 1. Multi-Format Concordance Audit
* **Qwen-7B ($n=200$ sample):**
  * **All 4 Formats Correct:** 180 / 200 (90.0%)
  * **All 4 Formats Fail:** 6 / 200 (3.0%)
  * **Mixed Outcome:** 14 / 200 (7.0%)
* **Llama-8B ($n=200$ sample):**
  * **All 4 Formats Correct:** 157 / 200 (78.5%)
  * **All 4 Formats Fail:** 9 / 200 (4.5%)
  * **Mixed Outcome:** 34 / 200 (17.0%)

---

## 2. Qualitative Trace & Token Inflation Mechanism
* **Step Deliberation Drift:** 4-bit quantized traces (AWQ-4 and GPTQ-4) frequently introduce additional intermediate algebraic restatements (e.g. repeated factorization checks) before concluding with the final boxed value.
* **Token Inflation vs BF16:**
  * `Qwen-7B FP8`: +11.41%
  * `Qwen-7B AWQ-4`: +19.81%
  * `Qwen-7B GPTQ-4`: +21.14%
  * `Llama-8B FP8`: +10.21%
  * `Llama-8B AWQ-4`: +29.77%
  * `Llama-8B GPTQ-4`: +28.45%

---

## 3. Extraction Integrity Verification
* **Boxed Answer Syntax:** Across all audited items, >99.5% of outputs adhered strictly to the requested `\boxed{...}` terminal syntax without requiring custom fallback heuristics.
