import json
import os
import sys
from typing import List, Dict, Tuple
from dataclasses import dataclass

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mediqa_synur_eval_script import classify_observations, ClassifiedObs, unroll_observations
from src.utils import load_schema, load_dataset_by_name

@dataclass
class ErrorCategory:
    id_hallucination: int = 0
    schema_violation: int = 0
    rag_leakage: int = 0
    pure_hallucination: int = 0
    value_mismatch: int = 0  # Same ID, different value
    missed_concept: int = 0

def analyze_sample_errors(
    prediction: List[Dict],
    reference: List[Dict],
    transcript: str,
    schema: List[Dict],
    rag_examples: List[Dict] = None
) -> Dict:
    """
    Decomposes errors for a single sample.
    """
    schema_map = {item['id']: item for item in schema}

    # Use official classifier to get TP/FP/FN
    obs = ClassifiedObs()
    obs = classify_observations(obs, {"observations": prediction}, {"observations": reference})

    # FP Analysis
    fp_details = []
    for fp in obs.fp_obs:
        fp_id = fp.get('id')
        fp_val = fp.get('value')

        category = "unknown"
        if fp_id not in schema_map:
            category = "id_hallucination"
        else:
            # Check schema violation
            item = schema_map[fp_id]
            # (Basic check, run_pipeline already does some sanitation but anyway)

            # RAG Leakage Check
            leakage = False
            if rag_examples:
                for ex in rag_examples:
                    # Check if this exact [id, value] pair exists in any few-shot example
                    for ex_obs in ex.get('observations', []):
                        if ex_obs['id'] == fp_id and ex_obs['value'] == fp_val:
                            leakage = True
                            break
                    if leakage: break

            if leakage:
                category = "rag_leakage"
            else:
                # Check if value is in transcript
                val_str = str(fp_val).lower()
                if val_str not in transcript.lower():
                    category = "pure_hallucination"
                else:
                    category = "over_extraction" # In transcript but perhaps wrong context or missed by ref

        fp_details.append({"obs": fp, "category": category})

    # FN Analysis
    fn_details = []
    for fn in obs.fn_obs:
        fn_id = fn.get('id')
        fn_val = fn.get('value')

        # Check if model predicted this ID but with a different value
        matching_fp = [fp for fp in obs.fp_obs if fp['id'] == fn_id]
        if matching_fp:
            category = "value_mismatch"
        else:
            category = "missed_concept"

        fn_details.append({"obs": fn, "category": category})

    return {
        "tp_count": len(obs.tp_obs),
        "fp_count": len(obs.fp_obs),
        "fn_count": len(obs.fn_obs),
        "fp_details": fp_details,
        "fn_details": fn_details
    }

def run_analysis_on_results(results_path: str, dataset_name: str = "dev"):
    print(f"Analyzing {results_path}...")
    schema = load_schema("synur_schema.json")
    full_data = load_dataset_by_name(dataset_name)

    with open(results_path, 'r') as f:
        predictions = [json.loads(line) for line in f]

    # Align predictions with data (by ID)
    pred_map = {str(p['id']): p['observations'] for p in predictions}

    aggregate_stats = {
        "total_tp": 0,
        "total_fp": 0,
        "total_fn": 0,
        "categories": {}
    }

    sample_reports = []

    for item in full_data:
        item_id = str(item['id'])
        if item_id not in pred_map:
            continue

        pred = pred_map.get(item_id, [])
        ref = item['observations']
        transcript = item['transcript']

        # Optional: Load RAG examples if it was a RAG run
        # Implementation depends on how we stored them.
        # For now we assume no RAG info unless we re-run or have logs.

        report = analyze_sample_errors(pred, ref, transcript, schema)
        report['id'] = item_id
        sample_reports.append(report)

        aggregate_stats["total_tp"] += report["tp_count"]
        aggregate_stats["total_fp"] += report["fp_count"]
        aggregate_stats["total_fn"] += report["fn_count"]

        for fp in report["fp_details"]:
            cat = fp["category"]
            aggregate_stats["categories"][cat] = aggregate_stats["categories"].get(cat, 0) + 1
        for fn in report["fn_details"]:
            cat = fn["category"]
            aggregate_stats["categories"][cat] = aggregate_stats["categories"].get(cat, 0) + 1

    # Print Summary
    print("\n--- ERROR ANALYSIS SUMMARY ---")
    print(f"Total TP: {aggregate_stats['total_tp']}")
    print(f"Total FP: {aggregate_stats['total_fp']}")
    print(f"Total FN: {aggregate_stats['total_fn']}")
    print("Categories:")
    for cat, count in aggregate_stats["categories"].items():
        print(f"  - {cat}: {count}")

    # Save detailed report
    report_path = results_path.replace(".jsonl", "_analysis.json")
    with open(report_path, 'w') as f:
        json.dump({"stats": aggregate_stats, "samples": sample_reports}, f, indent=2)
    print(f"Full report saved to {report_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("results_path", type=str)
    parser.add_argument("--dataset", type=str, default="dev")
    args = parser.parse_args()
    run_analysis_on_results(args.results_path, args.dataset)
