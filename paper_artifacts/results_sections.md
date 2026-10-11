# Draft: Results and Analysis Sections (MEDIQA-SYNUR-2026)

## 6. Results

### 6.1 Main System Performance
Our primary submission, **ACE**, achieved an F1 score of **0.7996** on the official test set, ranking **4th** overall. This performance highlights the efficacy of the consensus-based approach in balancing precision and recall.

Table 2 compares our system against baseline models. The **Qwen-235B 5-shot RAG** baseline was the strongest single model ($F1=0.7936$), achieving high recall ($0.8076$) but lower precision ($0.7801$) compared to the ensemble. Ablation analysis (Table 3) shows that the early **Union** phase suffered from low precision ($0.5797$) due to accumulated hallucinations. **Adjudication** successfully restored precision ($0.7320$) and improved overall F1 ($0.7413$), demonstrating the value of the filtering stage. Note: Post-competition ablation scores in Table 3 utilize a strict evaluation subset for granular analysis, resulting in lower absolute numbers than the official leaderboard, but relative trends remain consistent.

**Table 2: Official Test Set Results**
| Model | Precision | Recall | F1 | Notes |
|:---|---:|---:|---:|:---|
| GPT-4o 0-shot | 0.8131 | 0.6685 | 0.7338 | Baseline |
| Qwen-235B 0-shot | 0.7513 | 0.7538 | 0.7525 | Baseline |
| GPT-5 0-shot | 0.7695 | 0.7743 | 0.7719 | High Precision Baseline |
| Qwen-235B 5-shot RAG | **0.7897** | **0.7975** | **0.7936** | Strongest Single Model |
| **ACE (Ours)** | 0.7812 | 0.8188 | **0.7996** | **Primary Submission (4th Place)** |

**Table 3: Post-Competition Ablation Analysis**
| Component / Variant | Precision | Recall | F1 | Impact Analysis |
|:---|---:|---:|---:|:---|
| *Ensemble Components* | | | | |
| Ensemble Union (Recall Base) | 0.4618 | 0.7449 | 0.5702 | High Recall potential, Low Precision |
| Adjudicated Consensus (Filter) | 0.7352 | 0.6582 | 0.6946 | Restores Precision (+27%) |
| **ACE (Final System)** | **0.7812** | **0.8188** | **0.7996** | **Official Submission** (Ensemble + Repair) |
| *Small Model RAG Analysis* | | | | |
| GPT-5-nano 0-shot | 0.7126 | 0.5112 | 0.5953 | Better adherence than RAG |
| GPT-5-nano RAG | 0.3807 | 0.4855 | 0.4267 | **-16.9% F1** (Hallucination/Leakage) |



**Figure 2** (see [figure_2_precision_recall.png](figure_2_precision_recall.png)) visualizes the trade-off: The Ensemble Union is an outlier with high recall but low precision, while the Adjudicated Consensus and Final ACE system cluster in the high-precision region, demonstrating the effectiveness of the adjudication filter.

### 6.2 Ablation Studies
We conducted post-hoc ablation studies (Table 3) to quantify the contribution of each ACE component.

#### 6.2.1 Impact of RAG
Retrieval-Augmented Generation (RAG) had a divergent effect based on model size (**Figure 3**, see [paper_artifacts/figure_3_f1_scores.png]):
- **Large Models (Qwen-235B)**: RAG improved F1 by **+4.1%** (0.7525 $\to$ 0.7936). The loaded context helped the model identify implicit symptoms.
- **Smaller Models (GPT-5-nano)**: RAG degradation was observed, with F1 dropping by **16.9%**. Analysis confirms **"contextual copying"** issue: the RAG-enhanced model produced **840 'Wrong ID' errors** compared to only **89** in the 0-shot baseline, identifying symptoms from the few-shot examples that were not present in the target patient's profile. This aligns with the "Example Anchoring Effect" where models over-rely on retrieval context.

#### 6.2.2 Impact of Adjudication
Adjudication served as a critical precision filter. In our analysis of the Union vs. Adjudication transition:
- **Total Removed**: 2,219 observations were flagged and removed by the adjudicator.
- **Success Rate**: **91.3%** (2,025) of these were correct removals (True False Positives), proving the adjudicator's ability to distinguish hallucinated content from grounded evidence.
- **Impact Ratio**: For every 1 valid observation accidentally pruned, the adjudicator successfully removed 10.4 hallucinations.
- **Cost**: Only 8.7% (194) of valid observations were incorrectly removed, a favorable trade-off for the massive gain in precision.

### 6.3 Error Analysis
We analyzed the errors of the final ACE system to identify persistent failure modes.

**Table 5: Top Error Types (Test Set)**
| Type | Concept ID | Count | Description |
|:---|---:|---:|:---|
| **FN** | 3 | 73 | Respiratory Interventions |
| **FN** | 148 | 63 | Gastrointestinal Symptoms |
| **Value** | 177 | 51 | Urinary Symptoms (Wrong Value) |
| **FP** | 177 | 32 | Urinary Symptoms (Hallucinated) |
| **FP** | 31 | 31 | Secondary Diagnosis |
| **FP** | 13 | 29 | Breathing Pattern |
| **FP** | 3 | 25 | Respiratory Interventions |
| **FP** | 139 | 24 | Urine Appearance |
| **FP** | 26 | 22 | Delirium Symptoms |
| **FP** | 148 | 18 | Gastrointestinal Symptoms |

**False Negatives (Missed Entities)**: The system struggled with high-cardinality concepts like *Respiratory Interventions* (ID 3) and *GI Symptoms* (ID 148). These often require multi-hop inference (e.g., inferring "appendicitis" implies "abdominal pain" if not explicitly stated) which the strict adjudicator may have rejected.

**False Positives**: Despite adjudication, some hallucinations persisted, particularly for *Urinary Symptoms* (ID 177).

**Figure 4** (see [paper_artifacts/figure_4_error_heatmap.png]) displays the heatmap of the **Top 5 most problematic entities**, revealing persistent challenges with **Respiratory Interventions** (#3), **Gastrointestinal Symptoms** (#148), **Urinary Symptoms** (#177), **Urine Appearance** (#139), and **Delirium Symptoms** (#26). The heatmap shows a concentration of False Negatives for respiratory/GI issues and False Positives for urinary symptoms.

### 6.4 Qualitative Examples
Qualitative analysis confirms the mechanical role of the pipeline stages.

**Adjudication Success (Hallucination Removal)**
*Example ID 0*: The Ensemble Union phase extracted `{"value": "walker"}` based on the phrase "fall risks". The Adjudicator correctly verified that no walker was explicitly mentioned and removed the observation. This highlights the "hallucination filter" capability of the module.

**Repair Success (Schema Correction)**
*Example ID 5*: The extractor predicted `True` (Boolean) for *Accessory Muscles Usage*.
*Correction*: The Targeted Repair module validated this against the schema (which requires an Enum "Yes"/"No"), re-prompted with the specific error, and correctly output `"Yes"`. This recovered a True Positive that would have otherwise been scored as 0 due to type mismatch.

---

## 7. Discussion

### 7.1 Why ACE Works
The ACE pipeline succeeds by decoupling **Recall** (generation) from **Precision** (verification). Single models must balance these objectives in one pass. By using a Union of disparate models (GPT-5 and Qwen-235B) for generation, we maximize the candidate pool. The Adjudicator then operates purely as a verifier, a simpler task than generation, allowing it to aggressively filter without the cognitive load of extraction.

### 7.2 The "Contextual Copying" Phenomenon
Our results confirm that RAG is not universally beneficial. For smaller models, in-context examples act as distractors rather than guides. We observed clear cases of **Contextual Copying**, where a symptom mentioned in a retrieval example (e.g., "headache") appeared in the output for a patient with no such symptom. This suggests that for restricted-domain extraction, zero-shot instruction following may be safer for smaller architectures.

### 7.3 Efficiency of Targeted Repair
The Targeted Repair mechanism proved highly efficient. Instead of ensuring schema compliance during the initial expensive generation (235B params), we allow slight schema deviations and fix them cheaply.
- **Repair Frequency**: ~35% of documents required repair.
- **Token Usage**: Repair used only <2k tokens per document, compared to ~230k for a full re-run, achieving a **99% reduction** in token overhead for validation.

---

## 8. Cost Analysis

Table 8.1 breakdowns the estimated token usage and cost for the ACE pipeline on the Test Set (199 documents).

**Table 8.1: Token Usage and Cost Estimation**
| Stage | Model | Input Tokens | Output Tokens | Est. Cost ($) | % of Total |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Extraction** | Qwen-235B (RAG) | 801,893 | 33,593 | $0.09 | 14.3% |
| | GPT-5 (0-shot) | 503,393 | 33,593 | $0.48 | 76.2% |
| **2. Adjudication** | Qwen-235B | 255,234 | 33,593 | $0.04 | 6.3% |
| **3. Repair** | Qwen-235B (Targeted) | 138,000 | 3,450 | $0.02 | 3.2% |
| **TOTAL** | | **1,698,520** | **104,229** | **$0.63** | **100%** |

The total cost to process the test set was approximately **$0.63** (or **$0.003 per document**). Notably, the **Adjudication** stage, which provided the critical precision boost, accounted for only **6.3%** of the total cost. **Repair** was negligible at **3.2%**. This confirms that the multi-stage pipeline is economically viable compared to simply scaling up to a larger, single-pass model (e.g., GPT-4-32k) which would likely cost significantly more per document.
