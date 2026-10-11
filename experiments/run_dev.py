import argparse
import os
import sys
import json
import time
from typing import List, Dict, Any

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import MODELS
from src.utils import load_schema, load_dataset_by_name, inject_schema_details, save_jsonl, load_jsonl
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.doubleword_adapter import DoublewordAdapter
from src.metrics import evaluate_predictions
from src.repair import identify_failures
from experiments.variants import VARIANTS, VariantSpec

def get_adapter(model_key: str):
    config = MODELS.get(model_key)
    if not config:
        raise ValueError(f"Unknown model: {model_key}")
    if config.provider == "openai":
        return OpenAIAdapter(config)
    elif config.provider == "doubleword":
        return DoublewordAdapter(config)
    else:
        raise ValueError(f"Unsupported provider: {config.provider}")

def run_variant(variant: VariantSpec, dataset_name: str, mode: str, subset: int = None):
    print(f"--- Running Variant: {variant.name} ---")
    print(f"Description: {variant.description}")

    schema = load_schema("synur_schema.json")
    full_data = load_dataset_by_name(dataset_name)
    if subset:
        full_data = full_data[:subset]

    # Initialize Adapters
    adapters = {m: get_adapter(m) for m in variant.models}

    # Results directory
    output_dir = "outputs/results_ablations"
    os.makedirs(output_dir, exist_ok=True)

    # Handle RAG / Few-Shot
    examples = []
    if variant.fs_mode == "5-shot-fixed":
        examples = load_jsonl("5_shot_examples.jsonl")
    elif variant.fs_mode == "5-shot-dynamic" and variant.use_rag:
        print("Initializing Retriever...")
        from src.retrieval import Retriever
        retriever = Retriever()
        # For direct mode, we can enrich items with examples
        for item in full_data:
            item['_dynamic_examples'] = retriever.get_similar_examples(item['transcript'], k=5)

    if mode == "direct":
        results = run_direct(variant, full_data, schema, adapters, examples)

        # Step 3: Repair Loop
        if variant.use_repair:
            print("Starting Repair Loop...")
            transcripts = [d['transcript'] for d in full_data]
            failed_indices, repair_requests = identify_failures(results, transcripts, schema)

            if failed_indices:
                print(f"Found {len(failed_indices)} items needing repair.")
                primary_adapter = adapters[variant.models[0]]
                for req in repair_requests:
                    idx = req['index']
                    print(f"Repairing item {idx}...")
                    from src.repair import REPAIR_SYSTEM_PROMPT, build_repair_user_prompt
                    from src.utils import filter_schema_by_ids, build_system_prompt

                    failed_ids = list(set(str(obs.get('id')) for obs in req['invalid_obs'] if isinstance(obs, dict)))
                    subset_schema = filter_schema_by_ids(schema, failed_ids)

                    repair_messages = [
                        {"role": "system", "content": build_system_prompt(subset_schema) + "\n" + REPAIR_SYSTEM_PROMPT},
                        {"role": "user", "content": build_repair_user_prompt(req['transcript'], req['invalid_obs'], req['errors'])}
                    ]

                    try:
                        kwargs = primary_adapter._prepare_request_body(repair_messages, None)
                        res = primary_adapter.client.chat.completions.create(**kwargs)
                        repaired_obs = primary_adapter._parse_json_content(res.choices[0].message.content)

                        # Fix: Merge repaired into original valid ones
                        orig_valid = []
                        failed_ids = [o.get('id') for o in req['invalid_obs']]
                        for o in results[idx]:
                            if o.get('id') not in failed_ids:
                                orig_valid.append(o)

                        results[idx] = orig_valid + (repaired_obs if isinstance(repaired_obs, list) else [])
                    except Exception as e:
                        print(f"Repair failed for item {idx}: {e}")
            else:
                print("No repairs needed.")
        # Final formatting and metrics
        final_preds = [inject_schema_details(p, schema) for p in results]
        f1 = evaluate_predictions(final_preds, full_data)
        print(f"Variant {variant.name} F1 Score: {f1:.4f}")

        # Save results
        out_path = os.path.join(output_dir, f"{variant.name}_{dataset_name}.jsonl")
        # Align with competition format
        save_items = []
        for item, pred in zip(full_data, final_preds):
            save_items.append({
                "id": item["id"],
                "observations": pred
            })
        save_jsonl(save_items, out_path)
        print(f"Results saved to {out_path}")

    elif mode == "batched":
        # Simplified batch submission for dev set
        # For ensemble/adjudication, this is complex as it requires sequential batches or multi-threading
        # For now, we only support single model batches or manual ensemble runs
        if variant.ensemble_mode:
             print("Error: Batched mode for ensemble/adjudication is not yet implemented in run_dev.py.")
             print("Please use --mode direct for ensemble variants or run extractors individually.")
             return

        model_key = variant.models[0]
        adapter = adapters[model_key]
        adapter.predict_batch(
            full_data,
            schema,
            use_batch_api=True,
            experiment_name=f"{variant.name}_{dataset_name}",
            # examples=... (Handle RAG if used)
        )
        print("Batch submitted. Please check status with check_experiments.py")

def run_direct(variant: VariantSpec, data: List[Dict], schema: List[Dict], adapters: Dict, examples: List = None) -> List[List[Dict]]:
    # Step 1: Extractions
    all_extracted = {}
    for model_key, adapter in adapters.items():
        print(f"Extracting with {model_key}...")
        # items in data might have _dynamic_examples already
        preds = adapter.predict_direct(data, schema, examples)
        all_extracted[model_key] = preds

    # Step 2: Combination
    final_preds = []
    if not variant.ensemble_mode:
        # Single model
        return all_extracted[variant.models[0]]

    print(f"Combining results using {variant.ensemble_mode}...")
    for i in range(len(data)):
        item_preds = []
        for model_key in variant.models:
            item_preds.append(all_extracted[model_key][i])

        if variant.ensemble_mode == "union":
            # Simple deduplicated union by ID
            merged = {}
            for obs_list in item_preds:
                for obs in obs_list:
                    merged[obs['id']] = obs
            final_preds.append(list(merged.values()))

        elif variant.ensemble_mode == "adjudicate":
            # This requires a judge call per item (slow in direct mode)
            # For paper evidence, we might prefer offline adjudication
            # But let's implement the logic using the judge model
            judge_adapter = get_adapter(variant.adjudicator_model)

            # Formulate adjudication prompt
            # For simplicity, we compare the first two models if more than one
            # Ideally we adjudicate across all, but the current prompt is A vs B
            if len(item_preds) < 2:
                final_preds.append(item_preds[0])
                continue

            # Adjudication Logic (A vs B)
            # If 3 models, we could do A+B then (A+B)+C
            current_obs = item_preds[0]
            for j in range(1, len(item_preds)):
                # Adjudicate current_obs vs item_preds[j]
                # We need bridge code to call the judge adapter
                # For now, let's use a simple Union as fallback if judge fails
                # Actually, let's try to call the judge
                from src.utils import build_adjudicator_messages
                msg = build_adjudicator_messages(
                    schema,
                    data[i]['transcript'],
                    {"name": variant.models[0], "f1": 0.8, "observations": current_obs},
                    {"name": variant.models[j], "f1": 0.8, "observations": item_preds[j]}
                )

                # Manual direct call
                try:
                    res = judge_adapter.client.chat.completions.create(
                        model=judge_adapter.config.model_id,
                        messages=msg,
                        response_format={"type": "json_object"}
                    )
                    parsed = judge_adapter._parse_json_content(res.choices[0].message.content)
                    current_obs = parsed
                except Exception as e:
                    print(f"Adjudication failed for item {i}: {e}")
                    # Fallback to union
                    merged = {o['id']: o for o in current_obs}
                    for o in item_preds[j]: merged[o['id']] = o
                    current_obs = list(merged.values())

            final_preds.append(current_obs)

    return final_preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True, choices=list(VARIANTS.keys()))
    parser.add_argument("--dataset", type=str, default="dev")
    parser.add_argument("--mode", type=str, choices=["direct", "batched"], default="direct")
    parser.add_argument("--subset", type=int, help="Run on small subset")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    run_variant(variant, args.dataset, args.mode, args.subset)

if __name__ == "__main__":
    main()
