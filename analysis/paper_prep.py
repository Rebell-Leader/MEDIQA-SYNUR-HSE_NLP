import os
import json
import pandas as pd
import numpy as np
from typing import List, Dict
import argparse
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: Matplotlib/Seaborn not found. Skipping figure generation.")

def compute_dataset_stats(gold_data: List[Dict]):
    print("\nComputing Dataset Statistics...")

    num_docs = len(gold_data)
    total_tokens = 0
    total_obs = 0
    unique_concepts = set()
    value_types = {}

    for item in gold_data:
        transcript = item.get('transcript', '')
        # Simple whitespace tokenization
        total_tokens += len(transcript.split())

        obs_list = item.get('observations', [])
        total_obs += len(obs_list)

        for obs in obs_list:
            unique_concepts.add(obs.get('id', 'UNKNOWN'))
            vtype = obs.get('value_type', 'UNKNOWN')
            value_types[vtype] = value_types.get(vtype, 0) + 1

    avg_tokens = total_tokens / num_docs if num_docs > 0 else 0
    avg_obs = total_obs / num_docs if num_docs > 0 else 0

    stats = {
        'Metric': ['Examples', 'Avg Tokens/Transcript', 'Avg Obs/Example', 'Unique Concepts'],
        'Value': [num_docs, f"{avg_tokens:.1f}", f"{avg_obs:.1f}", len(unique_concepts)]
    }

    print(pd.DataFrame(stats))
    print("\nValue Type Distribution:")
    print(pd.Series(value_types))

    return stats

def analyze_adjudication_impact(union_preds: List[Dict], adj_preds: List[Dict], gold: List[Dict]):
    print("\nAnalyzing Adjudication Impact...")
    gold_map = get_predictions_map(gold)
    union_map = get_predictions_map(union_preds)
    adj_map = get_predictions_map(adj_preds)
    type_map = build_attribute_type_map(gold)

    removed_count = 0
    removed_fps = 0
    removed_tps = 0

    for pid, g_obs in gold_map.items():
        if pid not in union_map or pid not in adj_map: continue

        u_obs = inject_value_types(union_map[pid], type_map)
        a_obs = inject_value_types(adj_map[pid], type_map)

        # Determine strict set of (id, value) tuples
        u_set = set((o['id'], str(o['value']).lower()) for o in u_obs)
        a_set = set((o['id'], str(o['value']).lower()) for o in a_obs)
        g_set = set((o['id'], str(o['value']).lower()) for o in g_obs)

        # Items in Union but NOT in Adjudication => Removed
        removed = u_set - a_set

        for r_id, r_val in removed:
            removed_count += 1
            # Check if this removed item was actually correct (TP) or False (FP)
            if (r_id, r_val) in g_set:
                removed_tps += 1 # Bad removal (Recall loss)
            else:
                removed_fps += 1 # Good removal (Precision gain)

    print(f"Total Observations Removed by Adjudication: {removed_count}")
    print(f"  - Correctly Removed (FPs): {removed_fps} ({removed_fps/removed_count*100:.1f}%) -> Precision Gain")
    print(f"  - Incorrectly Removed (TPs): {removed_tps} ({removed_tps/removed_count*100:.1f}%) -> Recall Loss")

# Add project root to path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from mediqa_synur_eval_script import classify_observations, unroll_observations, ClassifiedObs, ClassificationStats

def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Auto-parse stringified observations if present
            if 'observations' in item and isinstance(item['observations'], str):
                try:
                    item['observations'] = json.loads(item['observations'])
                except:
                    pass
            data.append(item)
    return data

def build_attribute_type_map(gold_data: List[Dict]) -> Dict[str, str]:
    """Builds a map of observation types from gold data."""
    type_map = {}
    for item in gold_data:
        for obs in item.get('observations', []):
            if 'id' in obs and 'value_type' in obs:
                type_map[str(obs['id'])] = obs['value_type']
    return type_map

def inject_value_types(observations: List[Dict], type_map: Dict[str, str]) -> List[Dict]:
    """Injects missing value_type fields into observations."""
    fixed_obs = []
    for obs in observations:
        new_obs = obs.copy()
        if 'value_type' not in new_obs:
            oid = str(new_obs.get('id', ''))
            if oid in type_map:
                new_obs['value_type'] = type_map[oid]
            else:
                # Fallback implementation for hallucinated attributes
                val = new_obs.get('value')
                if isinstance(val, list):
                    new_obs['value_type'] = "MULTI_SELECT"
                else:
                    new_obs['value_type'] = "SINGLE_SELECT"
        fixed_obs.append(new_obs)
    return fixed_obs

def get_predictions_map(data: List[Dict]) -> Dict[str, List[Dict]]:
    return {str(item['id']): item['observations'] for item in data}

def analyze_errors_granular(predictions: List[Dict], gold: List[Dict], transcripts: Dict[str, str]):
    """
    Detailed error analysis:
    - Categorize by error type (FP, FN, Value Error)
    - Categorize by transcript length
    """
    print("Running Granular Error Analysis...")
    gold_map = get_predictions_map(gold)
    type_map = build_attribute_type_map(gold)

    error_stats = []

    for pred_item in predictions:
        pid = str(pred_item['id'])
        if pid not in gold_map: continue

        pred_obs = inject_value_types(pred_item['observations'], type_map)
        gold_obs = gold_map[pid]

        # Reset for single doc
        doc_cls = ClassifiedObs()
        doc_cls = classify_observations(doc_cls, {'observations': pred_obs}, {'observations': gold_obs})

        transcript_len = len(transcripts.get(pid, ""))

        # Analyzing FPs
        for fp in doc_cls.fp_obs:
            error_stats.append({
                'id': pid,
                'type': 'FP',
                'obs_id': fp['id'],
                'obs_val': fp['value'],
                'transcript_len': transcript_len
            })

        # Analyzing FNs
        for fn in doc_cls.fn_obs:
            # Check if it was a value error (ID matches an FP)
            is_value_error = any(fp['id'] == fn['id'] for fp in doc_cls.fp_obs)
            err_type = 'Value Error' if is_value_error else 'FN'

            error_stats.append({
                'id': pid,
                'type': err_type,
                'obs_id': fn['id'],
                'obs_val': fn['value'],
                'transcript_len': transcript_len
            })

    df = pd.DataFrame(error_stats)

    # Save CSV
    output_dir = Path("paper_artifacts")
    output_dir.mkdir(exist_ok=True)
    df.to_csv(output_dir / "error_analysis_granular.csv", index=False, escapechar='\\')

    # Summary by ID (Entity)
    if not df.empty:
        print("\nTop 10 Most Problematic Entities (FN + Value Error):")
        fns = df[df['type'].isin(['FN', 'Value Error'])]
        print(fns['obs_id'].value_counts().head(10))

        print("\nTop 10 Most Hallucinated Entities (FP):")
        fps = df[df['type'] == 'FP']
        print(fps['obs_id'].value_counts().head(10))

    return df

def run_ablation_study(file_map: Dict[str, str], gold_path: str):
    print("\nRunning Ablation Study...")
    gold_data = load_jsonl(gold_path)
    gold_map = get_predictions_map(gold_data)
    type_map = build_attribute_type_map(gold_data)

    results = []

    for name, path in file_map.items():
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue

        preds = load_jsonl(path)
        pred_map = get_predictions_map(preds)

        cls_obs = ClassifiedObs()
        # Iterate over GOLD IDs to ensure we cover all documents in evaluation,
        # or iterate over all union of IDs. Standard eval usually iterates over valid test set IDs.

        for pid, gold_observations in gold_map.items():
            if pid in pred_map:
                curr_obs = inject_value_types(pred_map[pid], type_map)
                curr_pred = {'observations': curr_obs}
            else:
                curr_pred = {'observations': []} # Missing prediction

            # Note: classify_observations expects dicts with 'observations' key
            # and it accumulates into cls_obs
            cls_obs = classify_observations(cls_obs, curr_pred, {'observations': gold_observations})

        stats = ClassificationStats()
        stats.calc(cls_obs)

        res_entry = {
            'Model': name,
            'Precision': stats.precision,
            'Recall': stats.recall,
            'F1': stats.f1
        }
        results.append(res_entry)
        print(f"{name}: P={stats.precision:.4f}, R={stats.recall:.4f}, F1={stats.f1:.4f}")

    df = pd.DataFrame(results)

    # Save md table
    output_dir = Path("paper_artifacts")
    output_dir.mkdir(exist_ok=True)

    markdown_table = df.to_markdown(index=False)
    with open(output_dir / "ablation_results.md", "w") as f:
        f.write("# Ablation Results\n\n")
        f.write(markdown_table)

    return df

def generate_plots(ablation_df, error_df=None):
    if not PLOTTING_AVAILABLE:
        return

    output_dir = Path("paper_artifacts")
    output_dir.mkdir(exist_ok=True)

    # Figure 2: Precision-Recall Scatter Plot
    plt.figure(figsize=(10, 6))
    if not ablation_df.empty:
        # Create scatter plot
        sns.scatterplot(data=ablation_df, x='Recall', y='Precision', s=100, hue='Model', style='Model')

        # Annotate points
        for i, row in ablation_df.iterrows():
            plt.text(row['Recall']+0.001, row['Precision']+0.001, row['Model'])

        plt.title('Precision vs Recall - Ablation Study')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "figure_2_precision_recall.png", dpi=300)
        plt.close()
        print("Generated Figure 2")

        # Figure 3: F1 Score Bar Chart
        plt.figure(figsize=(10, 6))
        sns.barplot(data=ablation_df, x='Model', y='F1')
        plt.title('F1 Score by Model/Stage')
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / "figure_3_f1_scores.png", dpi=300)
        plt.close()
        print("Generated Figure 3")

    # Figure 4: Error Heatmap (Confusion Matrix of sorts)
    if error_df is not None and not error_df.empty and PLOTTING_AVAILABLE:
        # Create a heatmap of Error Type vs Observation ID (Top 20)
        top_obs = error_df['obs_id'].value_counts().head(20).index
        filtered_df = error_df[error_df['obs_id'].isin(top_obs)]

        pivot_table = pd.crosstab(filtered_df['obs_id'], filtered_df['type'])

        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot_table, annot=True, fmt='d', cmap='YlOrRd')
        plt.title('Top 20 Problematic Entities: Error Type Distribution')
        plt.tight_layout()
        plt.savefig(output_dir / "figure_4_error_heatmap.png", dpi=300)
        plt.close()
        print("Generated Figure 4")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="test_gold.jsonl")
    parser.add_argument("--v3", default="outputs/results_ablations/v3_union_test_gold.jsonl")
    parser.add_argument("--v4", default="outputs/results_ablations/v4_adjudication_test_gold.jsonl")
    parser.add_argument("--v5", default="outputs/results_ablations/v5_repair_test_gold.jsonl")
    parser.add_argument("--v6", default="outputs/results_ablations/v6_ace_full_test_gold.jsonl")
    args = parser.parse_args()

    if not os.path.exists(args.gold):
        print(f"Error: Gold file {args.gold} not found.")
        return

    gold_data = load_jsonl(args.gold)

    # Dataset Stats
    compute_dataset_stats(gold_data)

    transcripts = {str(item['id']): item.get('transcript', '') for item in gold_data}

    # Error Analysis (Using V6 ACE Full as primary analysis target)
    error_df = None
    if os.path.exists(args.v6):
        v6_preds = load_jsonl(args.v6)
        error_df = analyze_errors_granular(v6_preds, gold_data, transcripts)

    # Adjudication Impact (V3 Union vs V4 Adjudication)
    if os.path.exists(args.v3) and os.path.exists(args.v4):
        v3_preds = load_jsonl(args.v3)
        v4_preds = load_jsonl(args.v4)
        analyze_adjudication_impact(v3_preds, v4_preds, gold_data)

    # Ablation Study
    files = {
        "V3 Union": args.v3,
        "V4 Adjudication": args.v4,
        "V5 Repair Only": args.v5,
        "V6 ACE Full": args.v6
    }

    # Check which files exist
    valid_files = {k: v for k, v in files.items() if os.path.exists(v)}

    ablation_df = run_ablation_study(valid_files, args.gold)

    generate_plots(ablation_df, error_df)

if __name__ == "__main__":
    main()
