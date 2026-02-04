import argparse
import json
import subprocess
import sys
import os

def check_alignment(pred_path, ref_path):
    print(f"Checking alignment between {pred_path} and {ref_path}...")

    if not os.path.exists(ref_path):
        print(f"Error: Reference file {ref_path} not found.")
        return False

    with open(ref_path, 'r') as f:
        ref_ids = set()
        for i, line in enumerate(f):
            try:
                ref_ids.add(json.loads(line)['id'])
            except Exception as e:
                print(f"Error parsing reference line {i+1}: {e}")

    if not os.path.exists(pred_path):
        print(f"Error: Prediction file {pred_path} not found.")
        return False

    with open(pred_path, 'r') as f:
        pred_ids = set()
        for i, line in enumerate(f):
            try:
                pred_ids.add(json.loads(line)['id'])
            except Exception as e:
                print(f"Error parsing prediction line {i+1}: {e}")

    missing = ref_ids - pred_ids
    extra = pred_ids - ref_ids

    if missing:
        print(f"WARNING: Missing predictions for {len(missing)} IDs: {list(missing)[:5]}...")
    else:
        print("SUCCESS: All reference IDs found in predictions.")

    if extra:
        print(f"WARNING: {len(extra)} extra IDs found in predictions (will be ignored by evaluator): {list(extra)[:5]}...")

    return len(missing) == 0

def main():
    parser = argparse.ArgumentParser(description="Run official evaluation script 'as is'")
    parser.add_argument("-p", "--predicted", required=True, help="Path to pred.jsonl")
    parser.add_argument("-r", "--reference", default="dev.jsonl", help="Path to reference jsonl")
    args = parser.parse_args()

    if check_alignment(args.predicted, args.reference):
        print("Data alignment verified. Running official evaluator...\n")
    else:
        print("Data alignment issues found. Evaluator will still run but results may be degraded.\n")

    cmd = [sys.executable, "mediqa_synur_eval_script.py", "-p", args.predicted, "-r", args.reference]
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
