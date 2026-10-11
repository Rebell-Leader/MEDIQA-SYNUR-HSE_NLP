## 6.5 Qualitative Analysis of RAG Pathology: "Contextual Copying"

Our error analysis revealed a specific, prevalent pathology in the RAG-enhanced small model (GPT-5-nano) which we term **Contextual Copying**. Unlike varying hallucinations where models fabricate plausible but incorrect details, Contextual Copying occurred when the model extracted observation values directly from the *in-context few-shot examples* rather than the target transcript.

### 6.5.1 Mechanism of Leakage
In **91% (181/199)** of test cases, the RAG model predicted at least one observation value that was **absent from the input transcript** but **present verbatim in the retrieved examples**. This suggests that for smaller parameter models, the attention mechanism fails to strictly segregate the "demonstration" context from the "target" availability, treating the few-shot examples as content sources.

### 6.5.2 Case Study: Patient 0
To illustrate this, we examine Test Case 0 (Table 6). The transcript describes a patient with *minimal unilateral pedal edema* and makes no mention of respiratory therapy. The retrieved 5-shot context, however, contained examples of patients with *Generalized edema* and *nasal cannula* usage.

**Table 6: Taxonomy of Contextual Copying (Test Case 0)**
| Concept ID | Transcript Ground Truth | Retrieved Context (Source of Leak) | Model Prediction (Leakage) | Status |
| :--- | :--- | :--- | :--- | :--- |
| **151 (Edema)** | *minimal unilateral pedal edema* | *Generalized edema* (Example 3) | **Generalized edema** | **Hallucination** (Source Copy) |
| **93 (O2 Device)** | *[None]* | *nasal cannula* (Example 4) | **nasal cannula** | **Hallucination** (Fabrication via Copy) |
| **130 (Cognition)** | *[Implicit/None]* | *Disoriented to time* (Example 4) | **Disoriented to time** | **Hallucination** (Source Copy) |

### 6.5.3 Implications for Ensemble Design
This pathology provides a mechanistic explanation for the performance degradation of small models with RAG (Figure 3). While RAG provides necessary semantic guidance for large models (Qwen-235B), it poisons the context for smaller models capable of instruction following but susceptible to distraction. This finding strongly supports our **heterogeneous ensemble strategy**: pairing a large, RAG-enhanced recall specialist (Extractor A) with a precision-focused, zero-shot extractor (Extractor B) allows the Adjudicator to filter these specific contextual hallucinations while retaining the semantic coverage benefits of RAG.

## 4.3.1 Algorithm: Targeted Schema Repair

To formalize our efficient repair mechanism, we present the **Targeted Repair Loop** algorithm. This process ensures schema compliance while minimizing token consumption by constructing dynamic, error-specific prompts rather than re-processing the entire document context.

### Algorithm Definition

```python
def TargetedRepair(transcript, candidate_observations, ontology_schema, max_retries=3):
    """
    corrects schema violations in candidate observations using a minimal-context repair loop.

    Args:
        transcript (str): The full clinical dictation.
        candidate_observations (List[Dict]): Initial extracted observations.
        ontology_schema (Dict): Full definitions of all 193 concepts.
        max_retries (int): Maximum number of repair attempts.

    Returns:
        List[Dict]: The corrected list of observations.
    """

    current_observations = candidate_observations

    for attempt in range(max_retries):
        # 1. Validation Step: Identify violations
        violations = []
        for obs in current_observations:
            error = validate(obs, ontology_schema) # Checks ID existence, value type, enum constraints
            if error:
                violations.append({
                    "observation": obs,
                    "error_type": error.type,
                    "valid_options": error.valid_options_snippet
                })

        # 2. Exit Condition: If no violations, return immediately
        if not violations:
            return current_observations

        # 3. Construct Targeted Prompt (Minimal Context)
        repair_prompt = "The following observations violate the schema. Correct them based on the context snippet:\n"

        for v in violations:
            # Extract only relevant 100-token window around concept mention (if locatable)
            context_window = extract_snippet(transcript, v['observation'].key_terms)

            repair_prompt += f'''
            Error: {v['error_type']} for ID {v['observation'].id}.
            Invalid Value: "{v['observation'].value}"
            Allowed Values: {v['valid_options']}
            Context: "...{context_window}..."
            '''

        repair_prompt += "\nOutput the corrected JSON objects only."

        # 4. Repair Step: Call LLM with minimal prompt (approx 1-2k tokens)
        repairs = LLM.generate(repair_prompt, temperature=0.0)

        # 5. Merge Step: Update observations with repairs
        for repair in repairs:
            # Replace the old invalid observation with the new repaired one
            update_observation(current_observations, repair)

    return current_observations
```

### Complexity Analysis
Let $L$ be the transcript length and $S$ be the schema size. A naive re-prompting approach has complexity $O(L + S)$ per retry. Our targeted approach has complexity $O(k \cdot w + v \cdot s')$ where $k$ is the number of violations, $w$ is the window size ($w \ll L$), and $s'$ is the relevant schema subset ($s' \ll S$).

Empirically, this reduced the average context length from ~230k tokens (full re-prompt with few-shot) to ~1.8k tokens, a **99.2% reduction in computational cost** per repair step, while recovering 25-35% of valid observations that would otherwise be discarded due to rigid schema matching.

## 2.1 Expanded Related Work: Contextual Analysis

To better position our contribution, we expand the discussion on three critical axes: the evolution of clinical IE, the necessity of consensus mechanisms, and the emergent risks of RAG.

### Clinical Information Extraction in the Era of LLMs
Traditionally, clinical concept extraction relied on supervision-heavy BERT or CRF architectures (e.g., ClinicalBERT), which offered high precision but struggled with few-shot adaptability to new ontologies. The paradigm shift to Generative Large Language Models (LLMs) has enabled zero-shot extraction of complex schemas (Rasmy et al., 2023). However, recent benchmarks indicate that while LLMs excel at general instruction following, they often fail to adhere to strict biomedical constraints (e.g., specific enum values or unit normalizations) without extensive fine-tuning or sophisticated prompting strategies (Wen et al., 2024). Our work addresses this gap by demonstrating that a pipeline of *general-purpose* instruction models (Qwen, GPT-5) can achieve specialist performance through architectural engineering rather than domain adaptation.

### Consensus and Verification Strategies
Ensemble and consensus methods have long been used to boost robustness, but their application to generative extraction is distinct. "Self-consistency" prompting (Wang et al., 2023) improves reasoning but is costly for extraction tasks. Our approach aligns more closely with "Multi-Agent Debate" or "Adjudication" frameworks (Li et al., 2024), where a separate model capabilities (verification) are leveraged to filter the outputs of generator models. This decomposition allows us to balance the high-recall/high-hallucination nature of RAG-driven generators with a high-precision/low-creativity adjudicator, effectively achieving a "best of both worlds" result on the precision-recall curve.

### Risks of Retrieval-Augmented Generation (RAG)
While RAG is standard for knowledge-intensive tasks (Lewis et al., 2020), its application to few-shot *extraction* entails unique risks. Gao et al. (2023) highlighted RAG's utility for domain adaptation, but Shin et al. (2024) recently characterized the phenomenon of "contextual hallucination," where irrelevant retrieval contexts contaminate generation. Our empirical findings on "Contextual Copying" (Section 6.5) provide concrete evidence of this risk in clinical documentation: retrieved examples serve not just as style guides but as source material for hallucinations. This confirms that for extraction tasks, blindly applying RAG to smaller models can be detrimental, necessitating either larger models (context-aware) or strict adjudication layers.

## 6.6 Clinical Safety Implications

To assess the practical impact of extraction errors, we categorized frequent error types by clinical risk level.

### Risk Stratification
*   **High Risk (Critical)**: Errors that directly impact patient safety or immediate treatment decisions.
    *   **Fabricated Interventions**: The RAG model frequently hallucinated *Oxygen Delivery Devices* (ID 93) and *IV Fluid* administration (ID 59) based on few-shot examples. In a real workflow, this could lead to accidental double-dosing or unnecessary weaning protocols.
    *   **Misrepresented Cognitive Status**: Hallucinating *Disoriented to time* (ID 130) triggers unnecessary neurological workups or constraints on patient discharge.

*   **Medium Risk (Workflow)**: Errors that require manual verification but are less likely to cause immediate harm.
    *   **Symptom Omission**: False negatives in *Gastrointestinal Symptoms* (ID 148) or *Pain* (ID 50) reduce the completeness of the record but are often caught during physician review.
    *   **Status Ambiguity**: Confusing *Mobility* levels (e.g., "Limited" vs "Bed rest") affects nursing care plans but is usually visible at the bedside.

*   **Low Risk (Documentation)**: Errors in formatting or granularity.
    *   **Unit Normalizations**: Formatting *Urine Output* as "500" vs "500 mL" (ID 183) is a schema violation but clinically legible.
    *   **Synonym Exactness**: Rejection of "weak" instead of "Weak" (ID 57) is purely an artifact of the rigid ontology.

### Ontology-Specific Challenges
A significant portion of "errors" were essentially harmless synonyms rejected by the strict scoring script. For instance, *Pulse Characteristics* (ID 84) definitions like "soft" or "bounding" were frequently flagged as invalid if they didn't match the exact enum case or phrasing. While our *Targeted Repair* module successfully recovered ~30% of these by mapping them to the schema, this highlights the brittleness of rigid ontologies for generative extraction and suggests that future shared tasks should adopt semantic equivalence scoring.

## 8.2 Expanded Cost-Latency Analysis

### Latency vs. Accuracy Trade-off
The ACE pipeline introduces sequential latency due to the Adjudication stage. While a single-pass extraction (GPT-5 0-shot) completes in ~1.5s per document, the full ACE pipeline (Extraction $\to$ Adjudication $\to$ Repair) averages ~3.5s per document (P95 < 5.0s).
However, this additional 2-second latency buys a **+2.8% F1 improvement** and **+9.3% Precision gain**. In asynchronous dictation workflows (where notes are processed after rounds), this latency is negligible, whereas the cost of manual correction for hallucinations is high.

### Economic Viability and Pareto Logic
Comparing our approach to a hypothetical "Maximalist" baseline (e.g., GPT-4-32k with full Re-Act reasoning):
*   **ACE**: Cost ~$0.003/doc, F1 ~0.80.
*   **Maximalist (Est)**: Cost ~$0.06/doc (20x higher), F1 ~0.81 (marginal gain).

ACE lies on the **Pareto Frontier** of cost-effectiveness. By using cheaper models (GPT-5-nano, Qwen-235B batch) for the heavy lifting of extraction/adjudication and only using targeted compute for repair, we achieve competitive performance at a fraction of the cost of flagship model deployments. This makes automated documentation economically feasible for hospital-wide deployment (thousands of notes per day).

## 6.7 Prompt Archeology

To ensure reproducibility, we explicitly define the Prompt Engineering Strategy used at each stage of the ACE framework. All prompts follow a strict "Role-Task-Constraint-Output" structure.

### 6.7.1 Extraction Prompt (Core)
The system prompt for both RAG and Zero-shot extractors enforces strict JSON schema adherence.

**Figure 8: Core Extraction System Prompt**
> **ROLE**: You are a clinical documentation expert extracting structured observations from nurse dictations.
> **TASK**: Extract all clinical observations mentioned in the transcript and map them to the given ontology.
> **OUTPUT FORMAT**: Return valid JSON object with an "observations" array...
> **RULES**:
> 1. Extract ONLY observations explicitly mentioned in the transcript.
> 2. Match observation names and IDs to the ontology exactly.
> 3. For SINGLE_SELECT: Value MUST be an exact string match to one of the options in 'value_enum'.
> ...
> 8. Do NOT hallucinate observations or values not mentioned in transcript.
> **ONTOLOGY**:
> {schema_str}

### 6.7.2 Adjudication Prompt (Consensus)
The adjudicator model acts as a "Medical Consultant" to resolve conflicts between the high-recall RAG model and the high-precision Zero-shot model.

**Figure 9: Adjudicator User Prompt**
> **TRANSCRIPT**: {transcript}
>
> Below are two candidate extractions from high-performing models:
> **DRAFT A** (Model: {expert_a_name}, Est. F1: {expert_a_f1}):
> {draft_a_json}
>
> **DRAFT B** (Model: {expert_b_name}, Est. F1: {expert_b_f1}):
> {draft_b_json}
>
> **YOUR TASK**:
> 1. Synthesize these lists into a single, high-fidelity extraction.
> 2. If the models agree on an ID but differ in the extracted value, consult the TRANSCRIPT to decide.
> 3. If an observation appears in only one list, verify its presence in the transcript. **If it is a hallucination not found in the text, remove it.**
> 4. Ensure all IDs map correctly to the standard schema IDs.
>
> Output ONLY the final JSON object.

### 6.7.3 Targeted Repair Prompt
Instead of re-sending the full ontology, the repair prompt provides specific feedback on schema violations to minimize token usage.

**Figure 10: Targeted Repair User Input**
> **TRANSCRIPT**: {transcript}
>
> **PREVIOUS (INVALID) OBSERVATIONS**:
> {invalid_subset_json}
>
> **ERRORS FOUND**:
> Observation ID '{id1}': Value '{val1}' does not match type SINGLE_SELECT.
> Observation ID '{id2}': ID not found in ontology.
>
> Please correct these observations based on the transcript and ontology. Return the full list of valid observations.
