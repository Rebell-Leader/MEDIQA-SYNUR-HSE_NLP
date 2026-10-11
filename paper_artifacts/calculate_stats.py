import json
import os
import tiktoken

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    # Schema is a list of dicts: [{"id": "0", "name": "...", ...}, ...]
    # Build map: ID (str) -> Name
    # Build map: ID (str) -> Name
    id_map = {str(item['id']): item['name'] for item in schema}
    type_map = {str(item['id']): item['value_type'] for item in schema}
    return id_map, schema, type_map

def calculate_stats(file_path, schema_map, type_map):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]

    enc = tiktoken.get_encoding("cl100k_base")

    num_examples = len(data)
    total_tokens = 0
    total_obs = 0
    unique_concepts = set()
    value_type_counts = {"SINGLE_SELECT": 0, "MULTI_SELECT": 0, "STRING": 0, "NUMERIC": 0}

    for entry in data:
        # Transcript length
        transcript = entry.get('transcript') or entry.get('text') or ''
        total_tokens += len(enc.encode(transcript))

        # Observations
        obs = entry.get('observations') or entry.get('labels') or entry.get('gold') or []
        # Handle stringified JSON if necessary (some datasets might do this)
        if isinstance(obs, str):
            try:
                obs = json.loads(obs)
            except:
                obs = []

        total_obs += len(obs)

        # Value Type Stats
        for o in obs:
            # key might be 'concept_id' or 'id' or 'observation_id'
            cid = str(o.get('concept_id') or o.get('id') or o.get('observation_id'))

            # Normalize ID
            if cid not in type_map:
                cid_stripped = cid.lstrip('0')
                if cid_stripped == '': cid_stripped = '0'
                if cid_stripped in type_map:
                    cid = cid_stripped

            unique_concepts.add(cid)

            vtype = type_map.get(cid)
            if vtype and vtype in value_type_counts:
                value_type_counts[vtype] += 1

    avg_tokens = total_tokens / num_examples if num_examples > 0 else 0
    avg_obs = total_obs / num_examples if num_examples > 0 else 0

    return {
        "num_examples": num_examples,
        "avg_tokens": avg_tokens,
        "avg_obs": avg_obs,
        "total_obs": total_obs,
        "unique_concepts": len(unique_concepts),
        "value_type_counts": value_type_counts
    }

if __name__ == "__main__":
    # Repository root (one level above paper_artifacts/)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    schema_path = os.path.join(base_dir, "synur_schema.json")

    id_map, schema, type_map = load_schema(schema_path)

    datasets = ["train.jsonl", "dev.jsonl", "test_gold.jsonl"]

    print(f"| Split | Examples | Avg Tokens | Avg Obs | Unique Concepts | % SINGLE | % MULTI | % NUMERIC | % STRING |")
    print(f"|---|---|---|---|---|---|---|---|---|")

    for ds in datasets:
        path = os.path.join(base_dir, ds)
        stats = calculate_stats(path, id_map, type_map)
        if stats:
            total = max(stats['total_obs'], 1)
            pcts = {k: (v/total)*100 for k,v in stats['value_type_counts'].items()}

            print(f"| {ds.replace('.jsonl','').title()} | {stats['num_examples']} | {stats['avg_tokens']:.1f} | {stats['avg_obs']:.1f} | {stats['unique_concepts']} | {pcts['SINGLE_SELECT']:.1f}% | {pcts['MULTI_SELECT']:.1f}% | {pcts['NUMERIC']:.1f}% | {pcts['STRING']:.1f}% |")

    print("\nTop 10 Error ID Mapping:")
    # IDs from log: 148, 177, 03, 139, 149, 84, 165, 13, 00, 130
    error_ids = ["148", "177", "3", "139", "149", "84", "165", "13", "0", "130"]

    for eid in error_ids:
        name = id_map.get(eid)
        if not name:
            # Try stripping leading zeros or adding them?
            # Schema IDs seem to be "0", "1", ... so "03" -> "3".
            name = id_map.get(str(int(eid)))
        print(f"ID {eid}: {name}")
