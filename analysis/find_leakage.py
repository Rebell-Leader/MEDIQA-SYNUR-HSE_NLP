import json
import os
from collections import defaultdict
from typing import List, Set, Dict, Tuple

def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def get_observations(item: Dict) -> Set[Tuple[str, str]]:
    """Returns a set of (id, str(value)) tuples."""
    obs_list = item.get('observations')

    # If not found at top level, try extracting from prediction response structure
    if obs_list is None:
        try:
            # Handle deep nested structure: response -> body -> choices -> [0] -> message -> content
            response = item.get('response', {})
            # Handle case where response might be stringified (seen in some formats)
            if isinstance(response, str):
                 try:
                     response = json.loads(response)
                 except:
                     pass

            body = response.get('body', {})
            choices = body.get('choices', [])
            if choices:
                content = choices[0].get('message', {}).get('content', '')
                if content:
                    # Content is usually a stringified JSON object {"observations": [...]}
                    try:
                        parsed_content = json.loads(content)
                        obs_list = parsed_content.get('observations', [])
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse content JSON for item {item.get('id')}")
        except Exception as e:
            # print(f"Debug: Extraction failed for {item.get('id')}: {e}")
            pass

    # Handle stringified list (as seen in gold data)
    if isinstance(obs_list, str):
        try:
            obs_list = json.loads(obs_list)
        except:
            print(f"Warning: Could not parse observations string for item {item.get('id')}")
            obs_list = []

    if obs_list is None:
        obs_list = []

    obs_set = set()
    for obs in obs_list:
        if not isinstance(obs, dict): continue
        # Handle case where keys might be different or missing
        if 'id' not in obs: continue

        val = str(obs.get('value', '')).strip()
        # Normalize simple types
        if val.lower() == 'true': val = 'Yes'
        if val.lower() == 'false': val = 'No'
        obs_set.add((str(obs.get('id')), val))
    return obs_set

def analyze_leakage():
    # Paths
    RAG_EXAMPLES_PATH = "outputs/rag/test_rag_examples.jsonl"
    PREDICTIONS_PATH = "outputs/results_gpt-5-nano-2025-08-07_v2_gpt5_nano_rag_test_gold.jsonl"

    # Try to find gold file
    GOLD_PATH = "test_gold.jsonl"
    if not os.path.exists(GOLD_PATH):
        GOLD_PATH = "data/test_gold.jsonl"
    if not os.path.exists(GOLD_PATH):
        # Fallback to looking in current directory
        files = [f for f in os.listdir('.') if 'test_gold.jsonl' in f]
        if files:
            GOLD_PATH = files[0]
        else:
            print("CRITICAL: test_gold.jsonl not found.")
            return

    print(f"Loading files...\nGold: {GOLD_PATH}\nPred: {PREDICTIONS_PATH}\nRAG: {RAG_EXAMPLES_PATH}")

    rag_data = {item['id']: item for item in load_jsonl(RAG_EXAMPLES_PATH)}
    pred_data = {str(item.get('custom_id') or item.get('id')): item for item in load_jsonl(PREDICTIONS_PATH)}
    gold_data = {str(item.get('custom_id') or item.get('id')): item for item in load_jsonl(GOLD_PATH)}

    print(f"Loaded {len(rag_data)} RAG items, {len(pred_data)} Pred items, {len(gold_data)} Gold items.")
    sample_key = list(gold_data.keys())[0]
    print(f"Sample Key: {sample_key}")
    print(f"In Rag: {sample_key in rag_data}")
    print(f"In Pred: {sample_key in pred_data}")

    leakage_counts = 0
    strong_leakages = []

    for cid, gold_item in gold_data.items():
        if cid not in pred_data or cid not in rag_data:
            continue

        pred_item = pred_data[cid]
        rag_item = rag_data[cid]

        gold_obs = get_observations(gold_item)
        pred_obs = get_observations(pred_item)

        # Collect all observations from retrieved examples
        retrieved_obs = set()
        for ex in rag_item.get("examples", []):
            for obs in ex.get("observations", []):
                val = str(obs.get("value", "")).strip()
                # Do the same normalization for retrieved items
                if val.lower() == 'true': val = 'Yes'
                if val.lower() == 'false': val = 'No'
                retrieved_obs.add((str(obs.get("id")), val))

        # Hallucinations = Pred - Gold
        hallucinations = pred_obs - gold_obs

        # Leakage = Hallucinations INTERSECT Retrieved
        leakage = hallucinations.intersection(retrieved_obs)

        if len(hallucinations) > 0 and len(leakage) == 0 and len(retrieved_obs) > 0:
             # Debug one case of non-leakage hallucination to see if we missed a match due to formatting
             pass

        if leakage:
            leakage_counts += 1
            # Filter for "interesting" leakage (ignore common defaults like "No")
            for (lid, lval) in leakage:
                # Heuristic: Numeric values or multi-word strings are stronger evidence than "Yes"/"No"
                is_strong = False
                try:
                    float(lval)
                    is_strong = True # leaked a number!
                except:
                    if len(lval.split()) > 1: # leaked a phrase
                        is_strong = True
                    # specialized strings
                    if lval in ["Present", "Absent", "Normal"]:
                         is_strong = False # Common

                if is_strong:
                    strong_leakages.append({
                        "id": cid,
                        "leakage_id": lid,
                        "leakage_value": lval,
                        "transcript_snippet": gold_item['transcript'][:200] + "...",
                        "retrieved_example_count": len(rag_item.get('examples', []))
                    })

    print(f"Total Cases with Leakage: {leakage_counts}")
    print(f"Strong Leakage Candidates: {len(strong_leakages)}")

    print("\n--- Top 5 Strong Leakage Examples ---")
    for item in strong_leakages[:5]:
        print(f"Case {item['id']}: Leaked ({item['leakage_id']}={item['leakage_value']})")

    # Save details
    with open("analysis/rag_leakage_report.json", "w") as f:
        json.dump(strong_leakages, f, indent=2)

if __name__ == "__main__":
    analyze_leakage()
