import argparse
import os
import sys
import json
import time
from typing import List, Dict

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import MODELS
from src.utils import load_schema, inject_schema_details, load_dataset_by_name
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.doubleword_adapter import DoublewordAdapter
from src.batch_manager import BatchManager
from src.repair import identify_failures, create_repair_batch_file
from src.submission import prepare_submission

# Config
SCHEMA_PATH = "synur_schema.json"
# DATA_PATH removed, loaded dynamically per experiment

# parse_results_generic removed, unnecessary as adapters handle parsing now (or we use adapter logic)
# Actually, we call adapter._parse_batch_results which expects a file.
# But check_experiments didn't use adapter methods before? It implemented parse_results_generic.
# Adapters methods were updated to return Dict. Let's use adapter methods if possible,
# or update parse_results_generic to match the new ID-based logic if we want to keep it independent.
# Better to use adapter methods for consistency.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reprocess", action="store_true", help="Force re-processing of completed batches")
    args = parser.parse_args()

    print("--- Checking Experiments ---")
    if args.reprocess:
        print("!! REPROCESS MODE: Will re-process all completed batches !!")

    manager = BatchManager()
    manager = BatchManager()
    schema = load_schema(SCHEMA_PATH)

    # Pre-load datasets if needed or load on fly
    # We will load on fly inside loop


    clients = {}
    for key, cfg in MODELS.items():
        if cfg.provider == "openai":
            # Just instantiate Adapter to use its helper methods if needed?
            # Or just use client.
            # We want to use adapter._parse_batch_results actually.
            clients[key] = OpenAIAdapter(cfg)
        elif cfg.provider == "doubleword":
            clients[key] = DoublewordAdapter(cfg)

    updated = False

    # FIX: Iterate over list(items) to allow modification of dictionary during loop
    for batch_id, info in list(manager.tracker.items()):
        if info['status'] in ['completed', 'failed', 'expired', 'cancelled']:
            if not args.reprocess:
                continue
            # Only reprocess actually completed ones, not failures
            if info['status'] != 'completed':
                continue

        model_key = info['model']
        if model_key not in clients:
            print(f"Warning: No client found for {model_key}")
            continue

        adapter = clients[model_key]
        client = adapter.client

        try:
            batch = client.batches.retrieve(batch_id)
            new_status = batch.status
            if new_status != info['status']:
                print(f"Batch {batch_id} ({info['experiment']}) updated: {info['status']} -> {new_status}")
                info['status'] = new_status
                updated = True

            if new_status == 'completed':
                info['output_file_id'] = batch.output_file_id
                manager._download_results(client, batch_id, info)

                exp_name = info['experiment']
                result_filename = f"results_{model_key}_{exp_name}.jsonl"
                result_path = os.path.join("outputs", result_filename)

                if os.path.exists(result_path):
                    print(f"Processing results for {batch_id} ({exp_name})...")

                    # Identify Dataset
                    if "_dev" in exp_name: dataset_name = "dev"
                    elif "_train" in exp_name: dataset_name = "train"
                    elif "_test" in exp_name: dataset_name = "test"
                    else:
                        print(f"Cannot infer dataset from {exp_name}. Skipping validation.")
                        continue

                    try:
                        full_data = load_dataset_by_name(dataset_name)
                    except Exception as e:
                        print(f"Failed to load dataset {dataset_name}: {e}")
                        continue

                    transcripts = [d['transcript'] for d in full_data]

                    # Use Adapter Parsing (Returns Dict[id, obs])
                    current_results_map = adapter._parse_batch_results(result_path)

                    # Logic Branch: Is this a Repair job or Original?
                    is_repair = "_repair" in exp_name

                    if is_repair:
                        # REPAIR MERGE LOGIC
                        parent_exp = exp_name.replace("_repair", "")
                        print(f"Repair detected. Merging with parent experiment: {parent_exp}")

                        # Load Parent Results
                        parent_result_path = os.path.join("outputs", f"results_{model_key}_{parent_exp}.jsonl")
                        if os.path.exists(parent_result_path):
                            parent_map = adapter._parse_batch_results(parent_result_path)

                            # Update Parent with Repair
                            # Repair map contains only fixed indices
                            for idx, obs in current_results_map.items():
                                parent_map[idx] = obs

                            final_map = parent_map
                        else:
                            print(f"Error: Parent result file not found at {parent_result_path}")
                            final_map = current_results_map # Fallback (incomplete)

                    else:
                        final_map = current_results_map

                    # Convert map to list aligned with full_data
                    predictions = []
                    for idx, item in enumerate(full_data):
                        # Try ID match first (new format)
                        key = str(item['id'])
                        if key in final_map:
                            predictions.append(final_map[key])
                        else:
                            # Fallback to legacy req-Index
                            legacy_key = f"req-{idx}"
                            legacy_key_repair = f"repair-{idx}" # Check repair naming too if needed, though map logic should have resolved it

                            if legacy_key in final_map:
                                predictions.append(final_map[legacy_key])
                            elif legacy_key_repair in final_map:
                                predictions.append(final_map[legacy_key_repair])
                            else:
                                if is_repair and (legacy_key in final_map or key in final_map):
                                     pass # Should have found it
                                predictions.append([])

                    # Inject Schema Details
                    predictions = [inject_schema_details(p, schema) for p in predictions]

                    # Check for Failures
                    failed_indices, repair_requests = identify_failures(predictions, transcripts, schema)

                    # SAVE SUBMISSION (Robustness Step 1)
                    # Determine folder name based on state
                    if is_repair:
                        sub_suffix = "FINAL_REPAIRED" if not failed_indices else "REPAIRED_PARTIAL"
                    else:
                        sub_suffix = "INITIAL_BROKEN" if failed_indices else "INITIAL_CLEAN"

                    sub_dir_name = f"{model_key}_{exp_name.replace('_repair', '')}_{sub_suffix}"
                    sub_dir = os.path.join("outputs", "submission", sub_dir_name)

                    print(f"Saving submission state to {sub_dir}...")
                    zip_path = prepare_submission(predictions, full_data, output_dir=sub_dir)
                    print(f"Archive created: {zip_path}")

                    # TRIGGER REPAIR (Robustness Step 2)
                    if failed_indices and not is_repair:
                        print(f"Found {len(failed_indices)} failures. Launching Repair...")

                        repair_exp_name = f"{exp_name}_repair"
                        timestamp = int(time.time())
                        repair_filename = f"batch_{model_key}_{repair_exp_name}_{timestamp}.jsonl"
                        repair_path = os.path.join("outputs", "batches", repair_filename)

                        create_repair_batch_file(repair_requests, schema, MODELS[model_key], repair_path)

                        # Submit
                        rep_batch_id = manager.submit_batch(client, repair_path, model_key, repair_exp_name)
                        print(f"Repair Batch ID: {rep_batch_id}")

                        # Explicitly update tracker with this new job immediately so we don't miss it?
                        # manager.submit_batch ALREADY adds to tracker.
                        updated = True # Ensure we save the new tracker state

                    elif failed_indices and is_repair:
                        print(f"Repair finished but {len(failed_indices)} errors remain. Manual review required.")

        except Exception as e:
            print(f"Error check {batch_id}: {e}")
            import traceback
            traceback.print_exc()

    if updated:
        manager._save_status()

if __name__ == "__main__":
    main()
