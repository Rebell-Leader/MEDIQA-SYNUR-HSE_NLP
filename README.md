# MEDIQA-SYNUR 2026 - Structured Clinical Observation Extraction, HSE NLP team

This repository contains the HSE NLP teams pipeline and models for the **MEDIQA-SYNUR 2026** competition. The goal is to extract structured clinical observations from nurse dictations based on a strict medical ontology.

Our primary submission, the **Consensus Adjudication Ensemble (ACE)**, ranked **4th** on the official leaderboard (F1 = 0.7996).

## 🚀 Top Results

Our best-performing models achieved high fidelity through a combination of **Dynamic RAG** and **Consensus Adjudication**.
Scores are on the official test set (micro P/R/F1 from `mediqa_synur_eval_script.py`).

| Model | Approach | F1 | Precision | Recall |
| :--- | :--- | :--- | :--- | :--- |
| **Qwen-235B** | **ACE Ensemble (v2)** | **0.7996** | 0.7812 | 0.8188 |
| Qwen-235B | Dynamic RAG (5-shot) | 0.7936 | 0.7897 | 0.7975 |
| GPT-5 | 0-shot | 0.7719 | 0.7695 | 0.7743 |
| Qwen-235B | 0-shot | 0.7525 | 0.7513 | 0.7538 |
| GPT-4o | 0-shot | 0.7338 | 0.8131 | 0.6685 |
| GPT-5-nano | 0-shot | 0.5953 | 0.7126 | 0.5112 |

The scores of every submitted run are logged in [`competition_logs.md`](competition_logs.md), and the corresponding predictions are in [`submission/`](submission/).

---

## 🛠️ Approaches

### 1. Consensus Adjudication Ensemble (ACE)
The successful strategy. It takes the union of extractions from our top two models (**Qwen-235B RAG** and **GPT-5 0-shot**) and uses a "Master Adjudicator" (Qwen-235B) to resolve conflicts and remove hallucinations by strictly verifying against the transcript.

### 2. Dynamic RAG (Few-Shot)
The well-performing strategy. Uses a vector database (Qdrant) to retrieve the 5 most similar clinical scenarios from the training set. These are injected as few-shot examples into the prompt to provide the model with context-relevant extraction patterns.

### 3. Targeted Repair Loop
A custom validation and repair system. If a model output fails ontology validation, the system generates a "Repair Prompt" containing only the erroneous IDs and the specific validation error, significantly reducing context window usage and improving repair accuracy.

---

## 📄 Paper

The system description paper and all of its supporting material live in [`paper_artifacts/`](paper_artifacts/):

- [`paper_draft.md`](paper_artifacts/paper_draft.md) - full paper text (abstract → conclusion).
- [`paper_outline.md`](paper_artifacts/paper_outline.md), [`results_sections.md`](paper_artifacts/results_sections.md), [`new_sections.md`](paper_artifacts/new_sections.md), [`paper_merge_guide.md`](paper_artifacts/paper_merge_guide.md) - section drafts and the merge plan used to assemble the paper.
- `figure_1_pipeline_architecture.png` … `figure_4_error_heatmap.png` - paper figures.
- [`paper_tables.md`](paper_artifacts/paper_tables.md), [`ablation_results.md`](paper_artifacts/ablation_results.md), [`qualitative_examples.md`](paper_artifacts/qualitative_examples.md), `error_analysis_granular.csv` - tables and analysis data used in the paper.

[`Agents.md`](Agents.md) gives a one-page summary of the final architecture and key findings.

---

## 📁 Project Structure

```text
.
├── src/                           # Core pipeline modules
│   ├── adapters/                  # Model-specific API wrappers (OpenAI, Doubleword, HF)
│   ├── batch_manager.py           # Async Batch API tracker (outputs/status_check.json)
│   ├── config.py                  # Model registry and environment config
│   ├── embeddings.py              # One-off: embed train set and upsert to Qdrant
│   ├── retrieval.py               # Dynamic 5-shot RAG retrieval
│   ├── repair.py                  # Targeted repair loop
│   ├── metrics.py, submission.py  # Evaluation helpers and pred.zip packaging
│   └── utils.py                   # Prompt building and ontology validation
├── experiments/                   # Post-competition ablation runner (V1 base → V6 ACE full)
├── analysis/                      # Error analysis, RAG leakage, cost estimation, table generation
├── paper_artifacts/               # Paper text, figures, tables and figure-generation scripts
├── submission/                    # Every submitted run (pred.jsonl + pred.zip)
├── outputs/
│   ├── results_*.jsonl            # Raw model outputs returned by the Batch APIs (+ *_repair.jsonl)
│   ├── results_ablations/         # Ablation predictions (V1-V6) and per-variant analyses
│   └── rag/                       # Cached retrieved 5-shot examples for dev and test
├── run_experiment.py              # Main experiment launcher
├── check_experiments.py           # Batch monitor: download → validate → repair → submission
├── convert_batch_to_submission.py # Raw Batch API output → pred.jsonl
├── reformat_ablation_predictions.py # Raw ablation output → evaluator format
├── run_official_eval.py           # Alignment check + official scorer wrapper
├── mediqa_synur_eval_script.py    # Official evaluation script (from the organizers)
├── ensemble_config.json           # Expert configuration for ACE mode
├── train.jsonl, dev.jsonl         # Training / development data
├── SYNUR_testset_input.jsonl      # Test transcripts (competition input)
├── test_gold.jsonl                # Test gold annotations (released after the competition)
├── synur_schema.json              # 193-concept observation ontology
├── competition_logs.md            # Leaderboard scores of every submitted run
├── README_competition.md          # Organizers' dataset card and task description
└── requirements.txt               # Python dependencies
```

Submission folders are suffixed by the state of the run when it was packaged: `INITIAL_BROKEN` (some outputs failed schema validation), `REPAIRED_PARTIAL` (after the targeted repair pass), `ENSEMBLE` / `ACE` (adjudication runs), `RAG` (dynamic 5-shot).

---

## ⚙️ Setup & Configuration

### 1. Environment Creation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Copy `.env.example` to `.env` in the root directory and fill in your keys (`.env` is git-ignored):
```env
OPENAI_API_KEY=your_openai_key
DOUBLEWORD_API_KEY=your_doubleword_key
QDRANT_URL=your_qdrant_instance_url
QDRANT_API_KEY=your_qdrant_api_key
```

API keys are only needed to re-run inference. All evaluation, analysis and figure scripts below work offline from the files in this repository.

---

## 🏃 Launching Experiments

### Running a Batch Experiment
To run our best RAG-enabled approach:
```bash
python run_experiment.py --model dw_qwen_235b --dataset test --stage 0-shot --mode batched --rag --exp-suffix RAG
```

### Running the ACE Ensemble
```bash
python run_experiment.py --model dw_qwen_235b --dataset test --stage 0-shot --mode batched --ensemble --exp-suffix ACE_v1
```

### Monitoring & Processing Results
The system is "fire-and-forget". Use the monitor to download results, perform repairs, and package for submission:
```bash
python check_experiments.py
```

### Official Evaluation
To verify results against a reference set (e.g., `dev.jsonl`):
```bash
python run_official_eval.py -p outputs/submission/YOUR_EXP_FOLDER/pred.jsonl -r dev.jsonl
```

---

## 🔁 Reproducing the Paper Results

All commands are run from the repository root and need no API keys.

**1. Official test-set scores** - re-score any submitted run against the gold annotations:
```bash
python mediqa_synur_eval_script.py -r test_gold.jsonl \
    -p submission/dw_qwen_235b_0-shot_test_ACE_v1_ENSEMBLE_REPAIRED_PARTIAL/pred.jsonl
# → P=0.7812  R=0.8188  F1=0.7996 (matches the leaderboard)
```

**2. Ablations, error analysis and tables** (inputs: `outputs/results_ablations/`, `outputs/rag/`, `test_gold.jsonl`):
```bash
python analysis/paper_prep.py               # error_analysis_granular.csv + ablation_results.md
python analysis/generate_paper_tables.py    # paper_tables.md
python paper_artifacts/calculate_stats.py   # Table 1 dataset statistics (tiktoken counts used in the paper)
python analysis/find_leakage.py             # RAG "contextual copying" analysis → analysis/rag_leakage_report.json
python analysis/find_qualitative_examples.py  # qualitative_examples.md
python paper_artifacts/analyze_errors.py    # per-concept error breakdown
python analysis/estimate_costs.py           # token / cost table for the ACE pipeline
```

**3. Figures** - run after step 2, because `paper_prep.py` also writes earlier drafts of figures 2-4:
```bash
python paper_artifacts/generate_figures.py  # figure_2_precision_recall, figure_3_f1_scores, figure_4_error_heatmap
```
Figure 1 (pipeline architecture) is a static diagram.

**4. Re-running the ablations from scratch** (requires API keys; variants are defined in `experiments/variants.py`):
```bash
# Phase 1: submit the base extractions to the Batch APIs
for v in v1_gpt5_nano v2_gpt5_nano_rag v1_dw_qwen_235b v1_gpt4o; do
    python experiments/run_dev.py --variant $v --dataset test_gold --mode batched
done
# Phase 2: once those batches finish, build V3 union → V4 adjudication → V5 repair → V6 ACE full
python experiments/batch_ablation_runner.py --action check --dataset test_gold   # re-run until "All ablation phases completed"
```

---

## 📚 Data & Licensing

The SYNUR data (`train.jsonl`, `dev.jsonl`, `SYNUR_testset_input.jsonl`, `test_gold.jsonl`, `synur_schema.json`) and the official evaluation script are provided by the MEDIQA-SYNUR 2026 organizers ([abachaa/MEDIQA-SYNUR-2026](https://github.com/abachaa/MEDIQA-SYNUR-2026)); the dataset is distributed under CDLA-Permissive-2.0. See [`README_competition.md`](README_competition.md) for the dataset card and task description.
