# MEDIQA-SYNUR 2026: Advanced Error Analysis & RAG Hallucinations

## 1. Quantitative Overview

| Metric | GPT-5-Nano (0-shot) | GPT-5-Nano (5-shot RAG) |
| :--- | :--- | :--- |
| **F1 Score** | 0.6920 | 0.5059 |
| **Precision** | 0.8134 | 0.4526 |
| **Recall** | 0.6022 | 0.5734 |
| **Total Predicted Obs** | 1029 | 1761 |
| **Wrong ID (Hallucination)** | 89 | 840 |
| **Wrong Value (Known ID)** | 91 | 104 |

> [!IMPORTANT]
> **Precision Collapse in RAG**: The switch from 0-shot to dynamic RAG caused a 44% drop in precision and a 10x increase in hallucinated IDs. Interestingly, Recall also slightly decreased, showing that RAK (Retrieval-Augmented Knowledge) in its current form introduced significant noise without aiding discovery.

---

## 2. Hallucination Patterns

### A. The "Knowledge Spillover" Effect
In the RAG model, there is a massive over-generation of specific entities that appear frequently in medical contexts but may not be present in the specific patient transcript.

**Top Hallucinated Entities (FP) in RAG:**
1. **Cognitive status [130]**: 43 extra (vs 9 in 0-shot)
2. **Oxygen saturation unit [178]**: 26 extra
3. **Respiratory interventions [3]**: 26 extra
4. **Urinary symptoms [177]**: 22 extra
5. **Oxygen saturation [10]**: 22 extra

**Hypothesis**: The model is "anchoring" on the 5-shot examples retrieved by RAG. If one or more retrieved examples mention oxygen saturation or cognitive issues, the model is biased toward extracting them even if the current transcript is silent on those points.

### B. Precision vs. Recall Trade-off
The 0-shot model is significantly more conservative. It only predicts 1029 observations for 1390 reference labels, whereas RAG predicts 1761 (over-predicting by ~26%). This confirms that RAG triggers "loose" extraction logic.

---

## 3. Top Missing Entities (FN)
Both models struggle with:
- **Secondary Diagnosis [31]**: 20-23 missed. This often requires complex inference from the transcript rather than simple extraction.
- **Gastrointestinal symptoms [148]**: 18-21 missed.
- **Mobility [89]**: 18-19 missed.

---

## 4. Value Errors (Known IDs)
While Hallucinations (Wrong IDs) are the main problem for RAG, Wrong Values (assigning the wrong data to a correctly identified ID) are also present:
- **Cognitive status [130]**: Consistent source of value mismatch (16 errors in RAG vs 6 in 0-shot). This suggests that "Disoriented to time" vs "Alert but confused" are often swapped or incorrectly mapped.

---

## 5. Summary for Paper
The "Advanced Error Analysis" section should highlight the **Failure of Dynamic RAG** in few-shot settings for this specific extraction task. The analysis suggests that **Zero-shot prompting with strong schema constraints** provides better faithfulness than retrieval-augmented methods that bring in potentially distracting medical evidence from other cases.
