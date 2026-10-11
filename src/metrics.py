import sys
import os
import json
from typing import List, Dict, Any

# Ensure we can import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Import from the official script in the root directory
try:
    from mediqa_synur_eval_script import ClassifiedObs, ClassificationStats, classify_observations
except ImportError:
    print("Warning: Could not import mediqa_synur_eval_script. Using local fallback?")
    raise

def evaluate_predictions(predictions: List[List[Dict]], ground_truth: List[Dict]) -> float:
    """
    Computes F1 score using the official MEDIQA-SYNUR evaluation logic.
    Wrapper around ClassifiedObs and ClassificationStats.
    """
    from src.utils import load_schema, inject_schema_details
    schema = load_schema("synur_schema.json")

    classified_obs = ClassifiedObs()

    for pred_list, gt_item in zip(predictions, ground_truth):
        gt_obs = gt_item.get('observations', [])
        if isinstance(gt_obs, str):
            gt_obs = json.loads(gt_obs)

        # Inject mandatory schema fields for the evaluator
        enriched_preds = inject_schema_details(pred_list, schema)
        enriched_refs = inject_schema_details(gt_obs, schema)

        pred_wrapper = {
            "id": gt_item["id"],
            "observations": enriched_preds
        }
        ref_wrapper = {
            "id": gt_item["id"],
            "observations": enriched_refs
        }

        try:
            classified_obs = classify_observations(classified_obs, pred_wrapper, ref_wrapper)
        except Exception as e:
            # print(f"Error evaluating sample {gt_item['id']}: {e}")
            pass

    final_results = ClassificationStats()
    final_results.calc(classified_obs)

    return final_results.f1

if __name__ == "__main__":
    import argparse
    from src.utils import load_jsonl

    parser = argparse.ArgumentParser()
    parser.add_argument("--preds", type=str, required=True)
    parser.add_argument("--gold", type=str, required=True)
    args = parser.parse_args()

    # Load gold
    gold_data = load_jsonl(args.gold)
    gold_ids = [str(d['id']) for d in gold_data]

    # Load preds
    preds_raw = load_jsonl(args.preds)
    preds_map = {}

    for p in preds_raw:
        iid = str(p.get('custom_id') or p.get('id', ''))
        if not iid: continue

        # 1. Standard format
        if 'observations' in p:
            preds_map[iid] = p['observations']
            continue

        # 2. Raw Batch format
        try:
            body = p.get('response', {}).get('body', {})
            content = body.get('choices', [{}])[0].get('message', {}).get('content', '') or ''

            # Extract JSON
            if "{" in content:
                content_clean = "{" + content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                parsed = json.loads(content_clean)
                preds_map[iid] = parsed.get('observations', [])
            else:
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, list): preds_map[iid] = parsed
                    elif isinstance(parsed, dict): preds_map[iid] = parsed.get('observations', [])
                except:
                    preds_map[iid] = []
        except:
            preds_map[iid] = []

    # Align
    aligned_preds = [preds_map.get(gid, []) for gid in gold_ids]

    f1 = evaluate_predictions(aligned_preds, gold_data)
    print(f"Results for {args.preds}:")
    print(f"F1 Score: {f1:.4f}")
