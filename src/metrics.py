import sys
import os
import json
from typing import List, Dict, Any

# Ensure we can import the official script from project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import from the official script in the root directory
# Assuming mediqa_synur_eval_script.py is in /home/obi/competitions/mediqa/MEDIQA-SYNUR-2026/
try:
    from mediqa_synur_eval_script import ClassifiedObs, ClassificationStats, classify_observations
except ImportError:
    # Fallback if specific path setup is needed or file missing in dev env
    print("Warning: Could not import mediqa_synur_eval_script. Using local fallback?")
    raise

def evaluate_predictions(predictions: List[List[Dict]], ground_truth: List[Dict]) -> float:
    """
    Computes F1 score using the official MEDIQA-SYNUR evaluation logic.
    Wrapper around ClassifiedObs and ClassificationStats.
    """
    classified_obs = ClassifiedObs()

    # Map by ID for the official script's expected format
    # The official script expects: classify_observations(classified_obs, predicted_dict_with_id, reference_dict_with_id)
    # where dicts are {"id": "...", "observations": [...]}

    for pred_list, gt_item in zip(predictions, ground_truth):
        # Construct dictionaries as expected by 'classify_observations'
        # pred_list is just [obs1, obs2...], need to wrap it

        # Note: gt_item['observations'] might be a string or list
        gt_obs = gt_item.get('observations', [])
        if isinstance(gt_obs, str):
            gt_obs = json.loads(gt_obs)

        # Re-wrap just for the function call format
        pred_wrapper = {
            "id": gt_item["id"],
            "observations": pred_list
        }
        ref_wrapper = {
            "id": gt_item["id"],
            "observations": gt_obs
        }

        # Accumulate stats
        try:
            classified_obs = classify_observations(classified_obs, pred_wrapper, ref_wrapper)
        except Exception as e:
            print(f"Error evaluating sample {gt_item['id']}: {e}")

    # Calculate final metrics
    final_results = ClassificationStats()
    final_results.calc(classified_obs)

    return final_results.f1
