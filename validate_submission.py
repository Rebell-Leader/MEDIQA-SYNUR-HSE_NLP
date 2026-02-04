import json
import sys
import os
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils import load_jsonl, load_schema, validate_observation, inject_schema_details
from src.metrics import evaluate_predictions

PRED_FILE = "outputs/debug_submission_batched/pred.jsonl"
GT_FILE = "dev.jsonl"
SCHEMA_FILE = "synur_schema.json"

def main():
    print(f"Validating {PRED_FILE}...")

    # 1. Load Files
    try:
        preds = load_jsonl(PRED_FILE)
        gt_full = load_jsonl(GT_FILE)
        schema = load_schema(SCHEMA_FILE)
    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return

    # Map GT by ID for easy lookup
    gt_map = {item['id']: item for item in gt_full}

    # 2. Schema Validation
    print("\n--- Schema Validation ---")
    valid_count = 0
    invalid_count = 0

    # Iterate through predictions
    # Predictions structure: [{"id": "...", "observations": [...]}, ...]

    aligned_preds = []
    aligned_gt = []

    for item in preds:
        pred_id = item['id']
        obs_list = item['observations']

        # Check against Schema
        item_valid = True
        for obs in obs_list:
            if not validate_observation(obs, schema):
                print(f"[INVALID] Record {pred_id}, Obs {obs.get('id')}: Value '{obs.get('value')}' invalid")
                item_valid = False

        if item_valid:
            valid_count += 1
        else:
            invalid_count += 1

        # Align for Metrics
        if pred_id in gt_map:
            # Inject schema details for evaluation metrics
            enriched_obs = inject_schema_details(obs_list, schema)
            aligned_preds.append(enriched_obs)
            aligned_gt.append(gt_map[pred_id])
        else:
            print(f"Warning: Prediction ID {pred_id} not found in Ground Truth")

    print(f"Schema Validation Results: {valid_count} Valid, {invalid_count} Invalid records.")

    # 3. Metric Calculation
    print("\n--- Metric Calculation (F1) ---")
    if aligned_preds:
        f1 = evaluate_predictions(aligned_preds, aligned_gt)
        print(f"F1 Score (on {len(aligned_preds)} matching records): {f1:.4f}")
    else:
        print("No matching records found for evaluation.")

if __name__ == "__main__":
    main()
