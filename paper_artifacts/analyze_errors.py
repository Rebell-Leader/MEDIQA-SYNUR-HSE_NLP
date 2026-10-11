import csv
import json
from collections import Counter, defaultdict
import sys

def load_id_map(path):
    id_map = {}
    try:
        with open(path, 'r') as f:
            for line in f:
                item = json.loads(line)
                # Check examples for names
                for ex in item.get('examples', []):
                    for obs in ex.get('observations', []):
                        if 'id' in obs and 'name' in obs:
                            id_map[str(obs['id'])] = obs['name']
    except Exception as e:
        print(f"Error loading ID map: {e}")
    return id_map

def analyze_errors():
    # Load ID mapping
    id_map = load_id_map("outputs/rag/test_rag_examples.jsonl")

    # Read errors
    errors_by_id = defaultdict(lambda: {'FP': 0, 'FN': 0, 'Value': 0, 'total': 0, 'examples': []})

    csv_path = "paper_artifacts/error_analysis_granular.csv"
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            # Handle null bytes generator
            clean_lines = (line.replace('\0', '') for line in f)
            reader = csv.DictReader(clean_lines)

            for row in reader:
                if not row: continue
                oid = row.get('obs_id', 'Unknown')
                etype = row.get('type', 'Unknown')
                val = row.get('obs_val', '')

                # Normalize error type
                if 'Value' in etype: etype = 'Value'

                errors_by_id[oid][etype] += 1
                errors_by_id[oid]['total'] += 1
                if len(errors_by_id[oid]['examples']) < 3:
                    errors_by_id[oid]['examples'].append(f"{etype}: {val}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Sort by total errors
    sorted_errors = sorted(errors_by_id.items(), key=lambda x: x[1]['total'], reverse=True)

    print("## Top 15 Error Concepts")
    print(f"{'ID':<5} {'Name':<35} {'Total':<5} {'FP':<5} {'FN':<5} {'Val':<5} {'Examples'}")
    print("-" * 120)

    for oid, stats in sorted_errors[:15]:
        name = id_map.get(oid, "Unknown")
        # Truncate examples for display
        ex_str = str(stats['examples'])[:60]
        print(f"{oid:<5} {name[:35]:<35} {stats['total']:<5} {stats['FP']:<5} {stats['FN']:<5} {stats['Value']:<5} {ex_str}")

if __name__ == "__main__":
    analyze_errors()
