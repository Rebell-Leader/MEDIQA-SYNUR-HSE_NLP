# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MEDIQA-SYNUR 2026 competition pipeline for extracting structured clinical observations from nurse dictations. The system uses LLMs with RAG and ensemble methods to map transcripts to a medical ontology defined in `synur_schema.json`.

## Key Commands

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run experiment (batched mode submits to async Batch API)
python run_experiment.py --model dw_qwen_235b --dataset test --stage 0-shot --mode batched

# With RAG (dynamic 5-shot retrieval from Qdrant)
python run_experiment.py --model dw_qwen_235b --dataset test --stage 0-shot --mode batched --rag

# With ensemble adjudication
python run_experiment.py --model dw_qwen_235b --dataset test --mode batched --ensemble

# Monitor batches, download results, trigger repairs
python check_experiments.py

# Evaluate against reference
python run_official_eval.py -p outputs/submission/YOUR_FOLDER/pred.jsonl -r dev.jsonl
```

## Architecture

### Data Flow
1. `run_experiment.py` → Creates batch files → Submits via `BatchManager` → Async processing
2. `check_experiments.py` → Polls status → Downloads results → Validates → Triggers repair if needed → Creates submission

### Core Components

- **src/adapters/** - Model API wrappers (OpenAI, Doubleword). Handle batch file creation, result parsing, direct inference.
- **src/config.py** - Model registry (`MODELS` dict with `ModelConfig` dataclasses). Add new models here.
- **src/batch_manager.py** - Tracks batch jobs in `outputs/status_check.json`, handles upload/download.
- **src/utils.py** - Prompt building, schema validation, observation sanitization, dataset loading.
- **src/repair.py** - Targeted repair system. Identifies validation failures and creates minimal repair prompts with filtered schema.
- **src/retrieval.py** - Qdrant-based RAG. Retrieves similar training examples by embedding similarity.
- **src/embeddings.py** - One-time script to embed training data and upsert to Qdrant.

### Key Data Structures

Observations follow this structure:
```json
{"id": "123", "value": "...", "value_type": "SINGLE_SELECT|MULTI_SELECT|STRING|NUMERIC", "name": "..."}
```

Value types have strict constraints:
- `SINGLE_SELECT`: Must match one of `value_enum` exactly
- `MULTI_SELECT`: Must be a list, all items in `value_enum`
- `NUMERIC`: Numbers only, no units
- `STRING`: Free text

### Experiment Modes

- **batched**: Fire-and-forget via Batch API. Results tracked in `outputs/status_check.json`.
- **direct**: Synchronous inference (for debugging).
- **RAG**: Injects 5 dynamically retrieved similar examples per transcript. Cached in `outputs/rag/`.
- **ensemble**: Uses expert predictions as hints for adjudication. Config in `ensemble_config.json`.

### Output Structure

- `outputs/batches/` - Batch input files (JSONL)
- `outputs/results_*.jsonl` - Raw batch results
- `outputs/submission/` - Final submissions (pred.jsonl + zip)
- `outputs/status_check.json` - Batch job tracker

### Multi-Phase Ablation Runner

`experiments/batch_ablation_runner.py` - Automated orchestration for complex experiments:
- Extraction → Union → Adjudication → Repair → ACE Full pipeline
- Use for running multi-model ensemble experiments

## Performance Benchmarks

| Model | Approach | F1 |
|:------|:---------|---:|
| Qwen-235B | 5-shot RAG | 0.794 |
| GPT-5 | 0-shot | 0.772 |
| Qwen-235B | 0-shot | 0.753 |
| GPT-4o | 0-shot | 0.734 |

## Key Findings

- **Adjudication > Repair**: Adjudication effectively filters noise from union ensembles. Repair can regress performance (~2.5 points) by introducing hallucinations during fix phase.
- **RAG provides significant boost**: Dynamic 5-shot retrieval improves GPT-5-nano from 0.50 to 0.66 F1.
- **Problematic observation IDs**: IDs 3, 148, 177, 139, 13 have highest error rates (FN, FP, value mismatches). See `paper_artifacts/paper_tables.md` for details.

## Environment Variables

Required in `.env`:
```
OPENAI_API_KEY=...
DOUBLEWORD_API_KEY=...
QDRANT_URL=...         # For RAG
QDRANT_API_KEY=...     # For RAG
```
