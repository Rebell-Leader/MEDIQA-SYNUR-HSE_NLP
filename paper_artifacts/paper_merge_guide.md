# Paper Merge Guide

This document maps the drafted content from `results_sections.md` and `new_sections.md` to the final paper structure outlined in `paper_outline.md`.

## Terminology Standardization
*   **Leakage Terminology**: Use **"Contextual Copying"** as the formal term for the phenomenon where RAG examples are copied. Use "Example Leakage" only when describing the mechanism (e.g., "Contextual Copying is caused by example leakage").
*   **System Name**: Ensure **ACE** (Consensus Adjudication Ensemble) is used consistently.

## Merge Map

### Section 2: Related Work
*   **Base**: `paper_outline.md` Section 2.
*   **Insert**: Content from `new_sections.md` -> **"2.1 Expanded Related Work: Contextual Analysis"**.
    *   *Action*: Append this subsection to the end of Section 2.

### Section 4: Methodology
*   **Base**: `paper_outline.md` Section 4.
*   **Insert**: Content from `new_sections.md` -> **"4.3.1 Algorithm: Targeted Schema Repair"**.
    *   *Action*: Insert after Section 4.3 description.
*   **Insert**: Content from `new_sections.md` -> **"6.7 Prompt Archeology"** (Rename to **"4.5 Prompt Engineering Strategy"**).
    *   *Action*: Move the prompt definitions to distinct boxes/figures in Methodology.

### Section 6: Results & Analysis
*   **Base**: `results_sections.md` (Main Results, Ablation Studies, Error Analysis).
*   **Insert**: Content from `new_sections.md` -> **"6.5 Qualitative Analysis of RAG Pathology"**.
    *   *Resolution*: This replaces/expands the brief "Impact of RAG" in `results_sections.md` Section 6.2.1. Keep 6.2.1 quantitative, use 6.5 for the qualitative case study.
*   **Insert**: Content from `new_sections.md` -> **"6.6 Clinical Safety Implications"**.

### Section 8: Cost Analysis
*   **Base**: `results_sections.md` "8. Cost Analysis" (Table 8.1).
*   **Insert**: Content from `new_sections.md` -> **"8.2 Expanded Cost-Latency Analysis"**.
    *   *Action*: Combine. Use Table 8.1 first, then follow with the "Latency vs Accuracy" and "Pareto Logic" text.

## Consistency Checklist
*   [ ] Check that Table 8.1 total cost ($0.63) matches mentions in Abstract/Intro.
*   [ ] Ensure "Targeted Repair" logic in Section 4 matches the algorithm pseudo-code in 4.3.1.
*   [ ] Verify Figure references (Figure 2, 3, 4) point to valid placeholders or generated images.
