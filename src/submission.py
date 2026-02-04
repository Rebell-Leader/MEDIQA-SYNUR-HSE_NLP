import json
import os
import zipfile
from typing import List, Dict

def prepare_submission(predictions: List[List[Dict]], data: List[Dict], output_dir: str = "outputs/submission"):
    """
    Create pred.jsonl and pred.zip for submission.
    predictions: list of observation lists, corresponding to data order
    data: original data list (to get IDs)
    """
    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "pred.jsonl")
    zip_path = os.path.join(output_dir, "pred.zip")

    print(f"Preparing submission files in {output_dir}...")

    # Write JSONL
    # Format: {"id": "sample_id", "observations": [...]}
    with open(jsonl_path, 'w') as f:
        for item, obs_list in zip(data, predictions):
            entry = {
                "id": item["id"],
                "observations": obs_list
            }
            f.write(json.dumps(entry) + '\n')

    print(f"Created {jsonl_path}")

    # Zip it
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(jsonl_path, arcname="pred.jsonl")

    print(f"Created {zip_path}")
    return zip_path
