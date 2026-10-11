6. Results (1.5 pages, ~900 words)
6.1 Main Results (~300w)

Official test set: F1=0.7996 (4th place)

Comparison to baselines:

Single best model: Qwen-235B 5-shot RAG (F1=0.7936)

ACE gain: +0.006 F1

Repair gain: ~+0.015 F1 (compare INITIAL_BROKEN vs. REPAIRED_PARTIAL)

Graphics needed:
✅ Table 4: Test set results (all your submissions from competition_logs.md, sorted by F1)
✅ Figure 2: Precision-Recall curve (or scatter plot showing P/R for each model)

6.2 Ablation Studies (~300w)

RAG impact by model size:

Qwen-235B: 0-shot (F1=0.7525) → 5-shot RAG (F1=0.7936) = +4.1%

GPT-5: 0-shot (F1=0.7719) → 5-shot RAG (F1=0.7308) = -4.1% (hallucination)

gpt-5-nano: 0-shot (F1=0.5953) → 5-shot RAG (F1=0.4995) = -9.6% (severe leakage)

Ensemble strategies:

Simple union (no adjudication): [run this on dev if time permits]

ACE (with adjudication): F1=0.7996

Repair mechanism:

All models: +1-2% F1 after repair

Token efficiency: 99% reduction

Graphics needed:
✅ Figure 3: Bar chart showing RAG impact by model size (F1 delta)

6.3 Error Analysis on Dev Set (~300w) [DO THIS BEFORE SUBMISSION]
Recommended experiments (1-2 days):

Error type taxonomy:

False negatives: Missed observations (categorize by value_type)

False positives: Hallucinated observations (categorize by model source)

Value errors: Correct ID but wrong value

Transcript difficulty analysis:

Short (< 100 tokens) vs. long (> 300 tokens) transcripts

Simple (< 5 observations) vs. complex (> 10 observations)

Adjudication effectiveness:

How many observations did adjudication remove? (precision gain)

How many valid observations were incorrectly removed? (recall loss)

Graphics needed:
✅ Table 5: Error breakdown by category (FN, FP, value errors) on dev set
✅ Figure 4: Confusion matrix or heatmap (observation types × error types)

7. Discussion (1 page, ~600 words)
7.1 Why ACE Works (~200w)

Complementary strengths: GPT-5 (precision) + Qwen-235B RAG (recall)

Adjudication as hallucination filter

Medical domain: High-stakes → consensus reduces risk

7.2 RAG Hallucination Phenomenon (~200w)

Hypothesis: Small models lack capacity to distinguish in-context examples from target transcript

Evidence: gpt-5-nano "leaked" observations from RAG examples (F1 dropped 9.6%)

Solution: Adjudication removed these false positives

Implication: RAG + consensus is safer than RAG alone for small models

7.3 Targeted Repair Efficiency (~100w)

99% token reduction → cost savings

Focused repair avoids introducing new errors

Generalizable to other structured extraction tasks

7.4 Limitations (~100w)

Synthetic SYNUR data (may not generalize to real dictations)

Computational cost: ACE requires 2 large model calls + adjudication

Test labels unavailable → cannot analyze test-specific failure modes

8. Conclusion (0.5 pages, ~300 words)
Summary: ACE + targeted repair achieved F1=0.7996 (4th place)

Key insights: (1) Consensus mitigates hallucinations, (2) RAG helps large models but hurts small models, (3) Targeted repair is token-efficient

Future work: (1) Apply ACE to real clinical data, (2) Investigate adjudication prompt design, (3) Explore cost-accuracy trade-offs

9. References (1 page)
Must-cite:

SYNUR overview paper [provided citation]

SYNUR dataset paper (Corbeil et al. 2025) [provided citation]

Clinical NLP surveys

Consensus-based clinical extraction
​

LLM hallucination mitigation
​

Structured medical extraction

Optional:

RAG methods [general NLP papers]

Few-shot learning in clinical NLP



## Advanced Error Analysis
### 4.1. Precision Collapse and Hallucinations
- **Pattern**: RAG model produces 10x more "Wrong ID" errors (840) than 0-shot (89).
- **The "Example Anchoring" Effect**: Model "leaks" symptoms from retrieved few-shot examples into the current patient profile.
- **Key Hallucinated IDs**: `Cognitive Status [130]`, `Respiratory Interventions [3]`, and `Oxygen Saturation [10]`.

### 4.2. Recall Bottlenecks
- Analysis of missed entities (>20% of cases): `Secondary Diagnosis [31]` and `Gastrointestinal symptoms [148]`.
- Difficulty of multi-step inference vs. direct extraction.

### 4.3. Entity-Specific Challenges
- **Cognitive Status (ID 130)**: High incidence of both value errors and hallucinations.
- **Strict String Matching**: Low F1 on `STRING` value types (0.08) due to lack of fuzzy matching in official eval.

## Discussion
- **The Case Against RAG for Small Large Language Models (sLLMs)**: In clinical extraction, extraneous context from other patients (RAG) acts as noise rather than helpful context.
- **Zero-shot Faithfulness**: Strong alignment with the target schema and instruction-following is more critical than external examples for this specific task.

## 6. Conclusion
- Recommendations for clinical NLP: Prioritize zero-shot instruction tuning and post-processing repair over retrieval-based methods for structured extraction tasks.
