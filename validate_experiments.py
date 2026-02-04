import os
import sys
import argparse
import json
import glob
from typing import List, Dict
from collections import defaultdict

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils import load_schema, load_jsonl, validate_observation
from src.metrics import evaluate_predictions

# Constants
SCHEMA_PATH = "synur_schema.json"
SUBMISSION_DIR = "outputs/submission"

# Dataset Map (Duplicate from run_experiment for now, common pattern)
# Ideally move to config if shared more
DATASETS = {
    "test": "SYNUR_testset_input.jsonl",
    "train": "train.jsonl",
    "dev": "dev.jsonl"
}

def analyze_errors(predictions: List[List[Dict]], schema: List[Dict]) -> Dict:
    """
    Returns text analysis of errors:
    - total_observations
    - invalid_count
    - invalid_types (counts by reason)
    """
    stats = {
        "total_obs": 0,
        "valid_obs": 0,
        "invalid_obs": 0,
        "errors_by_type": defaultdict(int),
        "missing_ids": 0
    }

    schema_map = {item['id']: item for item in schema}

    for pred_list in predictions:
        for obs in pred_list:
            stats["total_obs"] += 1

            # Re-implement strict validation with logging
            if not isinstance(obs, dict):
                 stats["invalid_obs"] += 1
                 stats["errors_by_type"]["Invalid Format (Not a Dict)"] += 1
                 continue

            obs_id = obs.get('id')
            if obs_id not in schema_map:
                stats["invalid_obs"] += 1
                stats["missing_ids"] += 1
                stats["errors_by_type"]["Unknown ID"] += 1
                continue

            item = schema_map[obs_id]
            val_type = item['value_type']
            val = obs.get('value')

            is_valid = True
            failure_reason = ""

            if val_type == 'NUMERIC':
                try:
                    float(val)
                except:
                    is_valid = False
                    failure_reason = "Non-Numeric"

            elif val_type == 'SINGLE_SELECT':
                enums = item.get('value_enum', [])
                if val not in enums:
                    is_valid = False
                    failure_reason = "Invalid Enum"

            elif val_type == 'MULTI_SELECT':
                if not isinstance(val, list):
                    is_valid = False
                    failure_reason = "Not a List"
                else:
                    enums = set(item.get('value_enum', []))
                    if not all(v in enums for v in val):
                        is_valid = False
                        failure_reason = "Invalid Enum in List"

            if is_valid:
                stats["valid_obs"] += 1
            else:
                stats["invalid_obs"] += 1
                stats["errors_by_type"][f"{val_type} Error: {failure_reason}"] += 1

    return stats

def process_experiment_folder(folder_path: str, schema: List[Dict]):
    folder_name = os.path.basename(folder_path)
    pred_path = os.path.join(folder_path, "pred.jsonl")

    if not os.path.exists(pred_path):
        return

    print(f"\nAnalyzing: {folder_name}")

    # 1. Identify Dataset & Load GT
    dataset_path = None
    if "dev" in folder_name:
        dataset_path = DATASETS["dev"]
    elif "train" in folder_name:
        dataset_path = DATASETS["train"]
    # Test set has no GT usually available for local metric calc

    predictions_raw = load_jsonl(pred_path) # [{'id':..., 'observations': []}]

    # Extract just observation lists
    predictions = [p['observations'] for p in predictions_raw]

    # 2. Schema Analysis (Always possible)
    stats = analyze_errors(predictions, schema)
    print(f"  Observations: {stats['total_obs']}")
    print(f"  Valid: {stats['valid_obs']} | Invalid: {stats['invalid_obs']}")
    if stats['invalid_obs'] > 0:
        print(f"  Errors: {dict(stats['errors_by_type'])}")

    # 3. Metrics (If GT available)
    if dataset_path and os.path.exists(dataset_path):
        gt_data = load_jsonl(dataset_path)

        # Align (predictions_raw has IDs, gt_data has IDs)
        # Create map
        gt_map = {item['id']: item for item in gt_data}

        aligned_preds = []
        aligned_gt = []

        for p_item in predictions_raw:
            pid = p_item['id']
            if pid in gt_map:
                # We need to inject schema details for metric script to work if missing
                # But metrics.py wrapper handles raw dicts?
                # Let's check: metrics.py calls classify_observations which expects 'value_type'.
                # The prediction file might not have value_type if direct output.
                # So we must inject here to be safe.
                from src.utils import inject_schema_details

                obs_enriched = inject_schema_details(p_item['observations'], schema)

                aligned_preds.append({"id": pid, "observations": obs_enriched})
                aligned_gt.append(gt_map[pid])

        if aligned_preds:
            # We need to unwrap for evaluate_predictions?
            # evaluate_predictions takes List[List[Dict]] (preds) and List[Dict] (gt items)
            # wait, evaluate_predictions signature: (predictions: List[List[Dict]], ground_truth: List[Dict])
            # aligned_preds currently is List[Dict wrapper].

            p_lists = [x['observations'] for x in aligned_preds]

            try:
                f1 = evaluate_predictions(p_lists, aligned_gt)
                print(f"  F1 Score: {f1:.4f}")
            except Exception as e:
                print(f"  Metric Calculation Failed: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="Specific experiment folder name or pattern")
    args = parser.parse_args()

    schema = load_schema(SCHEMA_PATH)

    if args.exp:
        target_path = os.path.join(SUBMISSION_DIR, args.exp)
        if os.path.exists(target_path):
            process_experiment_folder(target_path, schema)
        else:
            # Try glob
            pattern = os.path.join(SUBMISSION_DIR, f"*{args.exp}*")
            matches = glob.glob(pattern)
            for m in sorted(matches):
                 if os.path.isdir(m):
                     process_experiment_folder(m, schema)
    else:
        # Run all
        print(f"Scanning {SUBMISSION_DIR}...")
        subdirs = sorted([f.path for f in os.scandir(SUBMISSION_DIR) if f.is_dir()])
        for d in subdirs:
            process_experiment_folder(d, schema)

if __name__ == "__main__":
    main()
