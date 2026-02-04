import json
import os
import argparse
from typing import List, Dict, Any
from collections import Counter, defaultdict
from mediqa_synur_eval_script import ClassifiedObs, ClassificationStats, classify_observations, unroll_observations

def load_results_file(path: str, schema_map: Dict[str, Any], ordered_ids: List[str] = None) -> Dict[str, Dict]:
    """Robustly loads results and injects required value_type from schema."""
    results = {}
    if not os.path.exists(path):
        return {}

    with open(path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                item = json.loads(line)
            except:
                continue

            custom_id_raw = item.get('custom_id', '')
            if custom_id_raw.startswith('req-'):
                try:
                    idx = int(custom_id_raw.replace('req-', ''))
                    if ordered_ids and idx < len(ordered_ids):
                        custom_id = ordered_ids[idx]
                    else:
                        custom_id = str(idx) # Fallback to index string
                except:
                    custom_id = str(custom_id_raw)
            else:
                custom_id = str(custom_id_raw)

            if not custom_id: continue

            try:
                resp_body = item.get('response', {}).get('body', {})
                content = ""
                if 'choices' in resp_body:
                    content = resp_body['choices'][0]['message']['content']

                # Unwrap markdown
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                elif "```" in content:
                    parts = content.split("```")
                    content = parts[1].strip() if len(parts) >= 3 else parts[0].strip()

                if content:
                    try:
                        obs_data = json.loads(content)
                        raw_observations = obs_data.get("observations", [])
                    except:
                        if "{" in content:
                            try:
                                content_clean = "{" + content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                                raw_observations = json.loads(content_clean).get("observations", [])
                            except: raw_observations = []
                        else: raw_observations = []

                    fixed_observations = []
                    for obs in raw_observations:
                        oid = str(obs.get('id'))
                        val = obs.get('value')

                        # Injected value_type if missing (required by eval script)
                        vtype = obs.get('value_type')
                        if not vtype and oid in schema_map:
                            vtype = schema_map[oid]['value_type']

                        if oid in schema_map:
                            # Robust handle MULTI_SELECT if stringified list
                            if vtype == "MULTI_SELECT" and isinstance(val, str) and val.startswith('['):
                                try:
                                    val = json.loads(val)
                                except: pass

                            fixed_observations.append({
                                "id": oid,
                                "value": val,
                                "value_type": vtype,
                                "name": obs.get('name', schema_map[oid]['name'])
                            })

                    results[custom_id] = {"id": custom_id, "observations": fixed_observations}
            except Exception:
                results[custom_id] = {"id": custom_id, "observations": []}
    return results

def analyze_model_errors(prediction_path: str, reference_path: str, schema_path: str):
    # 1. Load Schema first
    with open(schema_path, 'r') as f:
        schema = json.load(f)
        schema_map = {str(item['id']): item for item in schema}

    # 2. Load Reference
    with open(reference_path, 'r') as f:
        ref_lines = [json.loads(line) for line in f]
        reference = {str(d["id"]): d for d in ref_lines}
        ordered_ids = [str(d["id"]) for d in ref_lines]

    # 3. Load Candidates with injection
    candidates = load_results_file(prediction_path, schema_map, ordered_ids)

    if not candidates:
        print(f"No valid predictions found in {prediction_path}")
        return

    # 4. Granular Classification
    overall_stats = ClassifiedObs()
    per_id_stats = defaultdict(ClassifiedObs)
    found_docs = 0

    for doc_id, ref_doc in reference.items():
        doc_stats = ClassifiedObs()
        if doc_id in candidates:
            hyp_doc = candidates[doc_id]
            found_docs += 1
        else:
            hyp_doc = {"observations": []}

        # Ensure ref_doc observations also have value_type (already do in dev.jsonl)
        classify_observations(doc_stats, hyp_doc, ref_doc)
        overall_stats += doc_stats

        for obs in doc_stats.tp_obs: per_id_stats[obs['id']].tp_obs.append(obs)
        for obs in doc_stats.fp_obs: per_id_stats[obs['id']].fp_obs.append(obs)
        for obs in doc_stats.fn_obs: per_id_stats[obs['id']].fn_obs.append(obs)

    # 5. Report Generation
    print("="*60)
    print(f"ERROR ANALYSIS FOR: {os.path.basename(prediction_path)}")
    print(f"Matched {found_docs}/{len(reference)} documents")
    print("="*60)

    final_stats = ClassificationStats()
    final_stats.calc(overall_stats)
    print(f"OVERALL: F1: {final_stats.f1:.4f} | Prec: {final_stats.precision:.4f} | Rec: {final_stats.recall:.4f}")
    print(f"Total Reference Obs: {len(overall_stats.fn_obs) + len(overall_stats.tp_obs)}")
    print(f"Total Predicted Obs: {len(overall_stats.fp_obs) + len(overall_stats.tp_obs)}")
    print("-" * 30)

    print("\nTOP MISSING ENTITIES (FN):")
    fn_counts = Counter({id: len(stats.fn_obs) for id, stats in per_id_stats.items()})
    for id, count in fn_counts.most_common(10):
        name = schema_map.get(id, {}).get('name', 'Unknown')
        print(f" - [{id:3}] {name:30}: {count} missed")

    print("\nTOP HALLUCINATIONS/OVER-GENERATION (FP):")
    fp_counts = Counter({id: len(stats.fp_obs) for id, stats in per_id_stats.items()})
    for id, count in fp_counts.most_common(10):
        name = schema_map.get(id, {}).get('name', 'Unknown')
        print(f" - [{id:3}] {name:30}: {count} extra")

    print("\nF1 BY VALUE TYPE:")
    type_stats = defaultdict(ClassifiedObs)
    for id, stats in per_id_stats.items():
        vtype = schema_map.get(id, {}).get('value_type', 'UNKNOWN')
        type_stats[vtype] += stats

    for vtype, stats in type_stats.items():
        if len(stats.tp_obs) + len(stats.fp_obs) + len(stats.fn_obs) == 0: continue
        s = ClassificationStats()
        s.calc(stats)
        print(f" - {vtype:15}: F1: {s.f1:.4f} (Rec: {s.recall:.4f}, Prec: {s.precision:.4f})")

    print("\nFP CAUSE ANALYSIS:")
    wrong_value_count = 0; wrong_id_count = 0
    id_mismatch_details = defaultdict(int)

    for doc_id, ref_doc in reference.items():
        if doc_id not in candidates: continue
        hyp_obs_map = unroll_observations(candidates[doc_id]['observations'])
        ref_obs_map = unroll_observations(ref_doc['observations'])
        for id_hyp, obs_list in hyp_obs_map.items():
            if id_hyp in ref_obs_map:
                hyp_vals = [str(o['value']) for o in obs_list]
                ref_vals = [str(o['value']) for o in ref_obs_map[id_hyp]]
                for v in hyp_vals:
                    if v not in ref_vals:
                        wrong_value_count += 1
                        id_mismatch_details[id_hyp] += 1
            else:
                wrong_id_count += len(obs_list)

    print(f" - Wrong Value (Known ID): {wrong_value_count}")
    print(f" - Wrong ID (Hallucination): {wrong_id_count}")

    if id_mismatch_details:
        print("\nTOP IDs WITH VALUE ERRORS:")
        for id, count in Counter(id_mismatch_details).most_common(5):
             name = schema_map.get(id, {}).get('name', 'Unknown')
             print(f" - [{id:3}] {name:30}: {count} errors")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prediction", required=True)
    parser.add_argument("-r", "--reference", default="dev.jsonl")
    parser.add_argument("-s", "--schema", default="synur_schema.json")
    args = parser.parse_args()
    analyze_model_errors(args.prediction, args.reference, args.schema)
