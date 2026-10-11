import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import time
import argparse
from typing import List, Dict, Any, Optional

from src.config import MODELS
from src.utils import load_jsonl, save_jsonl, load_schema, validate_observation

def get_adapter(model_key: str):
    config = MODELS.get(model_key)
    if not config:
        raise ValueError(f"Unknown model key: {model_key}")

    if config.provider == "openai":
        from src.adapters.openai_adapter import OpenAIAdapter
        return OpenAIAdapter(config)
    elif config.provider == "doubleword":
        from src.adapters.doubleword_adapter import DoublewordAdapter
        return DoublewordAdapter(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

def check_and_proceed(dataset_name: str):
    """
    Check status of submitted jobs and perform next steps in rotation.
    Phase 2: V3, V4, V5, V6
    """
    schema = load_schema("synur_schema.json")
    full_data = load_dataset_by_name(dataset_name)

    # Initialize adapters for checking status and parsing
    gpt5_nano = get_adapter("gpt-5-nano")
    qwen = get_adapter("dw_qwen_235b")
    gpt4o = get_adapter("gpt-4o")

    manager = gpt5_nano.batch_manager
    print("Updating statuses for all providers...")
    manager.update_statuses(gpt5_nano.client)
    manager.update_statuses(qwen.client)

    # Map from Experiment Name to results mapping (ID -> observations)
    results = {}

    expected_extractions = [
        ("gpt-5-nano", f"v1_gpt5_nano_{dataset_name}"),
        ("gpt-5-nano", f"v2_gpt5_nano_rag_{dataset_name}"),
        ("dw_qwen_235b", f"v1_dw_qwen_235b_{dataset_name}"),
        ("gpt-4o", f"v1_gpt4o_{dataset_name}")
    ]

    all_ready = True
    for model_key, exp in expected_extractions:
        found = False
        adapter = get_adapter(model_key)
        for bid, info in manager.tracker.items():
            if info['experiment'] == exp and info['status'] == 'completed':
                safe_model = info['model'].replace("/", "_")
                res_path = os.path.join("outputs", f"results_{safe_model}_{exp}.jsonl")
                if os.path.exists(res_path):
                    print(f"Parsing results from {res_path}...")
                    res_map = adapter._parse_batch_results(res_path)
                    results[exp] = res_map
                    found = True
                    break
        if not found:
            print(f"Extraction job {exp} is not yet ready.")
            all_ready = False

    if not all_ready:
        print("Waiting for base extractions to complete before proceeding to Ensemble/Repair steps.")
        return

    print("--- Base Extractions Ready ---")

    # 1. V3: Union Ensemble (Local)
    v3_path = f"outputs/results_ablations/v3_union_{dataset_name}.jsonl"
    if not os.path.exists(v3_path):
        print("Assembling V3 (Union of Nano, Qwen, GPT-4o)...")
        v1_nano_res = results[f"v1_gpt5_nano_{dataset_name}"]
        v1_qwen_res = results[f"v1_dw_qwen_235b_{dataset_name}"]
        v1_4o_res = results[f"v1_gpt4o_{dataset_name}"]

        v3_results = []
        for item in full_data:
            iid = str(item['id'])
            r1 = v1_nano_res.get(iid, [])
            r2 = v1_qwen_res.get(iid, [])
            r3 = v1_4o_res.get(iid, [])

            # Deduplicated union by ID
            merged = {obs['id']: obs for obs in (r1 + r2 + r3)}
            v3_results.append({"id": item['id'], "observations": list(merged.values())})

        os.makedirs("outputs/results_ablations", exist_ok=True)
        save_jsonl(v3_results, v3_path)
        print(f"V3 Union results saved to {v3_path}")
    else:
        print(f"V3 Union results already exist at {v3_path}")

    # 2. V4: Adjudication (Submit/Check New Batch)
    v4_exp_name = f"v4_adjudication_{dataset_name}"
    v4_submitted = False
    v4_ready = False
    for bid, info in manager.tracker.items():
        if info['experiment'] == v4_exp_name:
            if info['status'] == 'completed':
                 print(f"V4 Adjudication batch COMPLETED (ID: {bid})")
                 v4_ready = True
            else:
                 print(f"V4 Adjudication batch active (ID: {bid}, Status: {info['status']})")
            v4_submitted = True
            break

    if not v4_submitted:
        # Prepare Expert Hints Map for Adjudication
        v1_nano_res = results[f"v1_gpt5_nano_{dataset_name}"]
        v1_qwen_res = results[f"v1_dw_qwen_235b_{dataset_name}"]
        v1_4o_res = results[f"v1_gpt4o_{dataset_name}"]

        expert_hints_map = {}
        for item in full_data:
            iid = str(item['id'])
            expert_hints_map[iid] = [
                {"name": "gpt-5-nano", "f1": 0.82, "observations": v1_nano_res.get(iid, [])},
                {"name": "dw_qwen_235b", "f1": 0.81, "observations": v1_qwen_res.get(iid, [])},
                {"name": "gpt-4o", "f1": 0.85, "observations": v1_4o_res.get(iid, [])}
            ]

        print(f"Submitting V4 Adjudication batch...")
        gpt4o.predict_batch(
            full_data,
            schema,
            use_batch_api=True,
            experiment_name=v4_exp_name,
            wait=False,
            expert_hints_map=expert_hints_map
        )
        return

    # 3. V5: Repair (Identify failures in V4 and Repair)
    v4_results_map = {}
    if v4_ready:
        v4_res_path = os.path.join("outputs", f"results_gpt-4o_{v4_exp_name}.jsonl")
        print("--- Processing V4 Adjudication Results ---")
        v4_results_map = gpt4o._parse_batch_results(v4_res_path)

        v5_exp_name = f"v5_repair_{dataset_name}"
        v5_submitted = False
        v5_ready = False
        for bid, info in manager.tracker.items():
            if info['experiment'] == v5_exp_name:
                 if info['status'] == 'completed':
                     print(f"V5 Repair batch COMPLETED (ID: {bid})")
                     v5_ready = True
                 else:
                     print(f"V5 Repair batch active (ID: {bid}, Status: {info['status']})")
                 v5_submitted = True
                 break

        if not v5_submitted:
             print("Identifying failures in V4 for V5 Repair...")
             from src.repair import identify_failures, build_repair_user_prompt, REPAIR_SYSTEM_PROMPT
             from src.utils import build_system_prompt, filter_schema_by_ids

             transcripts = [d['transcript'] for d in full_data]
             aligned_v4 = [v4_results_map.get(str(d['id']), []) for d in full_data]

             failed_indices, repair_requests = identify_failures(aligned_v4, transcripts, schema)

             if not failed_indices:
                 print("No repairs needed for V5. V5 is same as V4.")
                 # (Implementation for final V6)
             else:
                 print(f"Found {len(failed_indices)} items needing repair. Submitting V5...")
                 custom_msgs = {}
                 for req in repair_requests:
                     iid = str(full_data[req['index']]['id'])
                     failed_ids = list(set(str(obs.get('id')) for obs in req['invalid_obs'] if isinstance(obs, dict)))
                     subset_schema = filter_schema_by_ids(schema, failed_ids)

                     sys_p = build_system_prompt(subset_schema, REPAIR_SYSTEM_PROMPT)
                     user_p = build_repair_user_prompt(req['transcript'], req['invalid_obs'], req.get('errors', []))

                     custom_msgs[iid] = [
                         {"role": "system", "content": sys_p},
                         {"role": "user", "content": user_p}
                     ]

                 gpt5_nano.predict_batch(
                     full_data,
                     schema,
                     use_batch_api=True,
                     experiment_name=v5_exp_name,
                     wait=False,
                     custom_messages_map=custom_msgs
                 )
                 return

        # 4. V6: ACE Full (Merge V4 Adjudicated + V5 Repaired)
        if v5_ready:
            v5_res_path = os.path.join("outputs", f"results_gpt-5-nano-2025-08-07_{v5_exp_name}.jsonl")
            print("--- Processing V5 Repair Results for V6 Assembly ---")
            v5_results_map = gpt5_nano._parse_batch_results(v5_res_path)

            v6_results = []
            for item in full_data:
                iid = str(item['id'])
                v4_obs = v4_results_map.get(iid, [])

                # Check if it was repaired
                if iid in v5_results_map:
                    # Merge logic: Filter V4 for valid ones, add all V5 (repairs)
                    valid_v4 = [o for o in v4_obs if validate_observation(o, schema)]
                    merged = valid_v4 + v5_results_map[iid]
                    # Dedup by ID just in case
                    deduped = {obs['id']: obs for obs in merged}
                    v6_results.append({"id": item['id'], "observations": list(deduped.values())})
                else:
                    v6_results.append({"id": item['id'], "observations": v4_obs})

            v6_path = f"outputs/results_ablations/v6_ace_full_{dataset_name}.jsonl"
            save_jsonl(v6_results, v6_path)
            print(f"V6 ACE Full results saved to {v6_path}")
            print("\nDONE: All ablation phases completed!")

def load_dataset_by_name(name: str):
    if name == "test_gold":
        return load_jsonl("test_gold.jsonl")
    elif name == "valid":
        return load_jsonl("valid.jsonl")
    else:
        raise ValueError(f"Unknown dataset {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", choices=["submit_v1", "check"], default="check")
    parser.add_argument("--dataset", choices=["test_gold", "valid"], default="test_gold")
    args = parser.parse_args()

    if args.action == "submit_v1":
        # (Simplified Phase 1 submission for dev reference)
        pass
    else:
        check_and_proceed(args.dataset)

if __name__ == "__main__":
    main()
