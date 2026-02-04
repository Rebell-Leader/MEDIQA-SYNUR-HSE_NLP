import os
import sys
import argparse
import time
import json
from typing import List, Dict

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import MODELS, ModelConfig
from src.utils import load_schema, inject_schema_details, load_jsonl, load_dataset_by_name
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.doubleword_adapter import DoublewordAdapter
from src.batch_manager import BatchManager
from src.metrics import evaluate_predictions
from src.submission import prepare_submission

SCHEMA_PATH = "synur_schema.json"

# Dataset Map
# DATASETS map and load_dataset moved to src/utils.py

def load_examples(fs_mode: str) -> List[Dict]:
    if fs_mode == "0-shot":
        return None
    elif fs_mode == "5-shot-fixed":
        ex_path = "5_shot_examples.jsonl"
        if not os.path.exists(ex_path):
             # Create on fly if missing (using train head)
             if os.path.exists("train.jsonl"):
                 print("Generating 5-shot examples from train head...")
                 all_train = load_jsonl("train.jsonl")
                 examples = all_train[:5]
                 with open(ex_path, 'w') as f:
                     for e in examples:
                         f.write(json.dumps(e) + "\n")
                 return examples
             else:
                 print("Warning: No examples found for 5-shot.")
                 return None
        return load_jsonl(ex_path)
    return None

def run_pipeline(model_key: str, dataset_name: str, fs_mode: str, mode: str, full_data: List[Dict], suffix: str = ""):
    config = MODELS.get(model_key)
    if not config:
        print(f"Model {model_key} not found.")
        return

    print(f"\n--- Processing {model_key} [{fs_mode}] on {dataset_name} ({len(full_data)} items) ---")

    schema = load_schema(SCHEMA_PATH)
    # transcripts = [d['transcript'] for d in full_data] # Adapters now take items
    examples = load_examples(fs_mode)

    # Init Adapter
    if config.provider == "openai":
        adapter = OpenAIAdapter(config)
    elif config.provider == "doubleword":
        adapter = DoublewordAdapter(config)
    else:
        print(f"Provider {config.provider} not supported in this runner.")
        return

    exp_name = f"{fs_mode}_{dataset_name}"
    if suffix:
        exp_name += f"_{suffix}"

    if mode == "batched":
        # Fire and forget (or wait manually if I wanted, but user asked for run_experiment to use batch manager)
        # The default behavior of adapter.predict_batch with use_batch_api=True is BLOCKING in my previous code
        # (check OpenAIAdapter.predict_batch).
        # Wait, I implemented blocking there for "Pipeline Compatibility".
        # But `run_experiment` originally was async.
        # Let's check logic:
        # OpenAIAdapter.predict_batch calls create_batch_file -> batch_manager.submit_batch -> loop wait.

        # If I want async here, I should call create_batch_file + batch_manager.submit locally,
        # OR update adapter to have a `wait=False` flag.
        # Since I can't easily change adapter method signature without breaking others potentially (though I just overwrote it),
        # I'll just use the raw steps here for flexibility.

        timestamp = int(time.time())
        batch_filename = f"batch_{model_key}_{exp_name}_{timestamp}.jsonl"
        batch_path = os.path.join("outputs", "batches", batch_filename)

        print(f"Generating batch file: {batch_path}")
        adapter.create_batch_file(full_data, schema, batch_path, examples)

        print(f"Submitting to {config.provider} Batch API...")
        batch_id = adapter.batch_manager.submit_batch(adapter.client, batch_path, model_key, exp_name)
        print(f"Launched: {batch_id}")

    elif mode == "direct":
        # Run synchronous inference
        predictions = adapter.predict_direct(full_data, schema, examples)

        # Post-process immediately
        predictions = [inject_schema_details(p, schema) for p in predictions]

        # Save results same format as batch results for consistency?
        # Or just save generic result file.
        res_dir = os.path.join("outputs", "local_runs")
        os.makedirs(res_dir, exist_ok=True)
        out_file = os.path.join(res_dir, f"results_{model_key}_{exp_name}.jsonl")

        # We need to format the output similar to what parser expects if we want check_experiments to work?
        # No, direct run is "Direct". We should evaluate here.

        # Evaluate if ground truth exists (observations in input)
        ground_truth = []
        has_gt = True
        for item in full_data:
            if 'observations' in item:
                ground_truth.append(item)
            else:
                has_gt = False
                break

        if has_gt:
            print("Evaluating against ground truth...")
            # align
            f1 = evaluate_predictions(predictions, ground_truth)
            print(f"F1 Score: {f1:.4f}")

            # Save metrics
            with open(os.path.join(res_dir, "metrics.txt"), "a") as f:
                f.write(f"{model_key} | {exp_name} | F1: {f1:.4f}\n")

        # Save Preds
        sub_dir = os.path.join("outputs", "submission", f"{model_key}_{exp_name}_DIRECT")
        zip_path = prepare_submission(predictions, full_data, output_dir=sub_dir)
        print(f"Saved local results to {zip_path}")


def main():
    parser = argparse.ArgumentParser(description="MEDIQA-SYNUR Experiment Runner")
    parser.add_argument("--stage", type=str, default="0-shot", choices=["0-shot", "5-shot-fixed", "5-shot-dynamic"], help="Experiment stage")
    parser.add_argument("--dataset", type=str, default="test", choices=["train", "dev", "test"], help="Dataset to run on")
    parser.add_argument("--model", type=str, help="Specific model key (e.g. gpt-5-nano). If not set, runs default set.")
    parser.add_argument("--mode", type=str, default="batched", choices=["batched", "direct"], help="Execution mode")
    parser.add_argument("--exp-suffix", type=str, default="", help="Optional experiment name suffix (e.g. v2_prompt)")
    parser.add_argument("--rag", action="store_true", help="Enable RAG (Dynamic 5-shot)")
    parser.add_argument("--re-run-rag", action="store_true", help="Force re-retrieval of RAG examples")
    parser.add_argument("--ensemble", action="store_true", help="Enable Ensemble Mode (Experts-as-Hints)")
    parser.add_argument("--ensemble-config", type=str, default="ensemble_config.json", help="Path to ensemble config")

    args = parser.parse_args()

    full_data = load_dataset_by_name(args.dataset)

    # RAG Pre-processing
    if args.rag:
        print(f"RAG Enabled for {args.dataset}. Checking cache...")
        rag_cache_dir = os.path.join("outputs", "rag")
        os.makedirs(rag_cache_dir, exist_ok=True)
        rag_cache_path = os.path.join(rag_cache_dir, f"{args.dataset}_rag_examples.jsonl")

        cached_examples = {}
        if os.path.exists(rag_cache_path) and not args.re_run_rag:
            print(f"Loading RAG examples from cache: {rag_cache_path}")
            with open(rag_cache_path, 'r') as f:
                for line in f:
                    entry = json.loads(line)
                    cached_examples[str(entry['id'])] = entry['examples']

        missing_ids = [str(item['id']) for item in full_data if str(item['id']) not in cached_examples]

        if missing_ids or args.re_run_rag:
            print(f"Retrieving dynamic examples for {len(missing_ids)} items (missing/forced)...")
            from src.retrieval import Retriever
            try:
                retriever = Retriever()
                from tqdm import tqdm

                new_entries = []
                for item in tqdm(full_data):
                    item_id = str(item['id'])
                    if item_id in cached_examples and not args.re_run_rag:
                        item['_dynamic_examples'] = cached_examples[item_id]
                        continue

                    # Retrieve 5 examples
                    examples = retriever.get_similar_examples(item['transcript'], k=5)
                    item['_dynamic_examples'] = examples
                    new_entries.append({"id": item_id, "examples": examples})

                # Update/Save cache
                if new_entries:
                    mode = 'w' if args.re_run_rag else 'a'
                    with open(rag_cache_path, mode) as f:
                        for entry in new_entries:
                            f.write(json.dumps(entry) + "\n")
                    print(f"RAG examples cached to {rag_cache_path}")

            except Exception as e:
                print(f"Failed to run RAG: {e}")
                return
        else:
            print("All items found in RAG cache.")
            for item in full_data:
                item['_dynamic_examples'] = cached_examples[str(item['id'])]

        args.exp_suffix += "_RAG" if not args.exp_suffix else "_RAG"

    # Ensemble Pre-processing
    if args.ensemble:
        print(f"Ensemble Mode Enabled. Loading experts from {args.ensemble_config}...")
        with open(args.ensemble_config, 'r') as f:
            e_cfg = json.load(f)

        for expert in e_cfg['experts']:
            print(f"Loading expert: {expert['name']} (F1: {expert['f1']}) from {expert['path']}")
            if not os.path.exists(expert['path']):
                print(f"Warning: Expert file not found at {expert['path']}")
                continue

            # Load expert predictions
            expert_preds = {str(e['id']): e['observations'] for e in load_jsonl(expert['path'])}

            # Inject into full_data
            for item in full_data:
                item_id = str(item['id'])
                if item_id in expert_preds:
                    if '_expert_hints' not in item:
                        item['_expert_hints'] = []
                    item['_expert_hints'].append({
                        "name": expert['name'],
                        "f1": expert['f1'],
                        "observations": expert_preds[item_id]
                    })

        args.exp_suffix += "_ENSEMBLE" if not args.exp_suffix else "_ENSEMBLE"

    # Target Models
    if args.model:
        targets = [args.model]
    else:
        # Default set from original script
        targets = ["gpt-5-nano", "gpt-4o-mini", "dw_qwen_30b", "dw_qwen_235b"]

    for model_key in targets:
        run_pipeline(model_key, args.dataset, args.stage, args.mode, full_data, args.exp_suffix)

    if args.mode == "batched":
        print("\n--- All Batches Submitted ---")
        print("Use `python check_experiments.py` to monitor status.")

if __name__ == "__main__":
    main()
