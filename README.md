# MEDIQA-SYNUR 2026 - Structured Clinical Observation Extraction, HSE NLP team (avaliev)

This repository contains the HSE NLP teams pipeline and models for the **MEDIQA-SYNUR 2026** competition. The goal is to extract structured clinical observations from nurse dictations based on a strict medical ontology.

## 🚀 Top Results

Our best-performing models achieved high fidelity through a combination of **Dynamic RAG** and **Consensus Adjudication**.

| Model | Approach | Precision | F1 | Recall |
| :--- | :--- | :--- | :--- | :--- |
| Qwen-235B | Dynamic RAG (5-shot) | 0.7936 | 0.7897 | 0.7975 |
| **Qwen-235B** | **ACE Ensemble (v2)** | **0.7996** | 0.7812 | 0.8188 |
| GPT-5 | 0-shot | 0.7719 | 0.7695 | 0.7743 |
| Qwen-235B | 0-shot | 0.7525 | 0.7513 | 0.7538 |
| GPT-4o | 0-shot | 0.7338 | 0.8131 | 0.6685 |
| GPT-5-nano | 0-shot | 0.5953 | 0.7126 | 0.5112 |

---

## 🛠️ Approaches

### 1. Consensus Adjudication Ensemble (ACE)
The successful strategy. It takes the union of extractions from our top two models (**Qwen-235B RAG** and **GPT-5 0-shot**) and uses a "Master Adjudicator" (Qwen-235B) to resolve conflicts and remove hallucinations by strictly verifying against the transcript.

### 2. Dynamic RAG (Few-Shot)
The well-performing strategy. Uses a vector database (Qdrant) to retrieve the 5 most similar clinical scenarios from the training set. These are injected as few-shot examples into the prompt to provide the model with context-relevant extraction patterns.

### 3. Targeted Repair Loop
A custom validation and repair system. If a model output fails ontology validation, the system generates a "Repair Prompt" containing only the erroneous IDs and the specific validation error, significantly reducing context window usage and improving repair accuracy.

---

## 📁 Project Structure

```text
.
├── src/                    # Core logic and modules
│   ├── adapters/           # Model-specific API wrappers (OpenAI, Doubleword)
│   ├── batch_manager.py    # Async Batch API tracker
│   ├── config.py           # Model registry and environment config
│   ├── embeddings.py       # Embedding generation and Qdrant upsert
│   ├── retrieval.py        # RAG retrieval logic
│   ├── repair.py           # Automated JSON repair system
│   └── utils.py            # Prompt building and ontology validation
├── outputs/                # Experiment results and submissions
├── run_experiment.py       # Main experiment launcher
├── check_experiments.py    # Batch monitor and results processor
├── run_official_eval.py    # Alignment-verifying evaluation wrapper
├── ensemble_config.json    # Expert configuration for ACE mode
├── requirements.txt        # Python dependencies
└── .env                    # Environment variables (API Keys)
... (Here are also datasets, official eval script + config, batch checkers, etc - sorry for the mess, the main part is still ./src + run_experiment.py + check_experiments.py)
```

---

## ⚙️ Setup & Configuration

### 1. Environment Creation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuration (`.env`)
Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_key
DOUBLEWORD_API_KEY=your_doubleword_key
QDRANT_URL=your_qdrant_instance_url
QDRANT_API_KEY=your_qdrant_api_key
```

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
