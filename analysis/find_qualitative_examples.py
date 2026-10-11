import json
import argparse
from pathlib import Path

def load_jsonl(path):
    data = {}
    with open(path, 'r') as f:
        for line in f:
            item = json.loads(line)
            if 'observations' in item and isinstance(item['observations'], str):
                try:
                    item['observations'] = json.loads(item['observations'])
                except:
                    pass
            data[str(item['id'])] = item
    return data

def find_adjudication_success(gold, v3_union, v4_adj, limit=5):
    """Finds examples where Union had a FP that Adjudication removed."""
    examples = []

    for pid, g_item in gold.items():
        if pid not in v3_union or pid not in v4_adj:
            continue

        g_obs = {(o['id'], str(o['value']).lower()) for o in g_item['observations']}
        v3_obs = {(o['id'], str(o['value']).lower()): o for o in v3_union[pid]['observations']}
        v4_obs = {(o['id'], str(o['value']).lower()) for o in v4_adj[pid]['observations']}

        # Find FPs in V3 (in V3 but not in Gold)
        v3_fps = [key for key in v3_obs if key not in g_obs]

        for fp_key in v3_fps:
            # Check if this FP was removed in V4
            if fp_key not in v4_obs:
                examples.append({
                    "id": pid,
                    "type": "Adjudication Success (FP Removed)",
                    "transcript": g_item['transcript'],
                    "removed_observation": v3_obs[fp_key],
                    "context": "V3 (Union) predicted this, but V4 (Adjudication) correctly removed it as it was not in Gold."
                })
                if len(examples) >= limit:
                    return examples
    return examples

def find_repair_success(gold, v4_adj, v6_ace, limit=5):
    """Finds examples where V6 corrected a value from V4 to match Gold."""
    examples = []

    for pid, g_item in gold.items():
        if pid not in v4_adj or pid not in v6_ace:
            continue

        g_map = {o['id']: str(o['value']) for o in g_item['observations']}
        v4_map = {o['id']: str(o['value']) for o in v4_adj[pid]['observations']}
        v6_map = {o['id']: str(o['value']) for o in v6_ace[pid]['observations']}
        v6_full_obs = {o['id']: o for o in v6_ace[pid]['observations']}

        # Check for items where V4 != V6 AND V6 == Gold
        common_ids = set(v4_map.keys()) & set(v6_map.keys()) & set(g_map.keys())

        for oid in common_ids:
            val_v4 = v4_map[oid]
            val_v6 = v6_map[oid]
            val_gold = g_map[oid]

            if val_v4 != val_v6 and val_v6 == val_gold:
                 examples.append({
                    "id": pid,
                    "type": "Repair Success (Value Corrected)",
                    "transcript": g_item['transcript'],
                    "observation_id": oid,
                    "v4_value": val_v4,
                    "v6_value": val_v6,
                    "gold_value": val_gold,
                    "full_observation": v6_full_obs[oid],
                    "context": f"V4 predicted '{val_v4}', which likely violated schema or was incorrect. V6 repaired it to '{val_v6}', matching Gold."
                })
                 if len(examples) >= limit:
                    return examples
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", default="test_gold.jsonl")
    parser.add_argument("--v3", default="outputs/results_ablations/v3_union_test_gold.jsonl")
    parser.add_argument("--v4", default="outputs/results_ablations/v4_adjudication_test_gold.jsonl")
    parser.add_argument("--v6", default="outputs/results_ablations/v6_ace_full_test_gold.jsonl")
    args = parser.parse_args()

    print("Loading data...")
    gold = load_jsonl(args.gold)
    v3 = load_jsonl(args.v3)
    v4 = load_jsonl(args.v4)
    v6 = load_jsonl(args.v6)

    print("Finding Adjudication Successes...")
    adj_success = find_adjudication_success(gold, v3, v4)

    print("Finding Repair Successes...")
    rep_success = find_repair_success(gold, v4, v6)

    output_path = "paper_artifacts/qualitative_examples.md"
    with open(output_path, "w") as f:
        f.write("# Qualitative Examples for Paper\n\n")

        f.write("## 1. Adjudication Success (Hallucination Removal)\n")
        for i, ex in enumerate(adj_success, 1):
            f.write(f"### Example 1.{i} (ID: {ex['id']})\n")
            f.write(f"**Transcript Snippet**: ...{ex['transcript'][0:300]}...\n\n")
            f.write(f"**Hallucination (V3 Union)**:\n```json\n{json.dumps(ex['removed_observation'], indent=2)}\n```\n")
            f.write(f"**Outcome**: Adjudication correctly **removed** this observation (not present in Gold).\n\n")

        f.write("\n## 2. Repair Success (Value Correction)\n")
        for i, ex in enumerate(rep_success, 1):
            f.write(f"### Example 2.{i} (ID: {ex['id']})\n")
            f.write(f"**Transcript Snippet**: ...{ex['transcript'][0:300]}...\n\n")
            f.write(f"**Concept ID**: {ex['observation_id']}\n")
            f.write(f"- **V4 Value (Before Repair)**: `{ex['v4_value']}`\n")
            f.write(f"- **V6 Value (After Repair)**: `{ex['v6_value']}` (Matches Gold)\n")
            f.write(f"- **Gold Value**: `{ex['gold_value']}`\n\n")

    print(f"Examples written to {output_path}")

if __name__ == "__main__":
    main()
