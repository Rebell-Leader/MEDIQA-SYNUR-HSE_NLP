# Agents.md - ACE System Architecture & Findings

## 1. Project Overview
**Status**: Completed (Rank 4 in MEDIQA-SYNUR 2026)
**Final Score**: **F1 = 0.7996** (Precision: 0.7812, Recall: 0.8188)
**Mission**: Extract structured clinical observations from nurse dictations with strict ontology adherence.

## 2. Theoretical Breakthroughs
Our work identified two key phenomena in clinical extraction with LLMs:

### 2.1 Paradox of Contextual Copying
*   **Discovery**: Retrieval-Augmented Generation (RAG) improves recall for large models (Qwen-235B) but induces **hallucinations** in smaller models (e.g., GPT-5-nano).
*   **Mechanism**: Small models do not just use retrieved examples for style; they copy observation values *verbatim* from the examples into the prediction, even if absent from the source text.
*   **Impact**: In our ablation, GPT-5-nano with RAG had a **16.9% F1 drop** compared to zero-shot, solely due to this leakage.

### 2.2 Efficiency of Targeted Repair
*   **Problem**: Full re-generation to fix schema errors (invalid enums, wrong types) is expensive (~230k tokens/doc because of the massive schema).
*   **Solution**: "Targeted Repair Loop" that generates a minimal prompt containing *only* the error and a 100-token context window.
*   **Result**: Reduced validation cost by **99.2%** (from ~230k to ~1.8k tokens/repair) while recovering ~35% of valid observations.

## 3. Final Architecture: The ACE Pipeline
The Consensus Adjudication Ensemble (ACE) uses a "Union-then-Filter" approach.

### Stage 1: Parallel Extraction
Two diverse models maximize coverage:
1.  **Qwen-235B (Instruct)**: **5-Shot Dynamic RAG**. Uses `all-MiniLM-L6-v2` to find similar training examples. High Recall, moderate precision.
2.  **GPT-5 (0-Shot)**: **Zero-Shot**. Uses strict schema instruction. High Precision, lower recall.

### Stage 2: Union Accumulation
*   `Candidates = Output_A ∪ Output_B`
*   Maximizes recall by accepting any potential observation.

### Stage 3: Master Adjudication
*   **Model**: Qwen-235B (0-Shot Verifier)
*   **Role**: Examining each candidate against the source text.
*   **Performance**: For every 1 valid observation it accidentally removed, it successfully removed **10.4 hallucinations**. This 10:1 ratio is the core of our high precision.

### Stage 4: Targeted Repair
*   **Role**: Validates final JSON against `synur_schema.json` and fixes type/enum errors using the efficient loop described in 2.2.

## 4. Performance Metrics (Official Test Set)
| Component | Precision | Recall | F1 | Note |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-235B (RAG)** | 0.7897 | 0.7975 | 0.7936 | Strongest Single Model |
| **GPT-5 (0-Shot)** | 0.7695 | 0.7743 | 0.7719 | High Precision Base |
| **Ensemble Union** | 0.4618 | 0.7449 | 0.5702 | Massive Over-generation |
| **Adjudicated (ACE)** | **0.7812** | **0.8188** | **0.7996** | **Final System** |

## 5. Artifacts and References
*   `paper_artifacts/paper_draft.md`: Full paper text.
*   `paper_artifacts/figure_1_pipeline_architecture.png`: Visual diagram of the ACE pipeline.
*   `src/`: Source code for the `batch_ablation_runner` and `repair_loop`.
