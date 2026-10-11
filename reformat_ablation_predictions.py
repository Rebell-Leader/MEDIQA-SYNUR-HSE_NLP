import json
import sys
import os

def load_schema(schema_path):
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    # Map id to value_type. Normalize keys by stripping leading zeros.
    return {item['id'].lstrip('0') or '0': item['value_type'] for item in schema}

def convert(input_path, output_path, schema_map):
    print(f"Converting {input_path} -> {output_path}")
    count = 0
    with open(input_path, 'r') as fin, open(output_path, 'w') as fout:
        for line in fin:
            if not line.strip(): continue
            try:
                data = json.loads(line)
                # Handle Batch API format
                if 'custom_id' in data and 'response' in data:
                    p_id = data['custom_id']

                    if data.get('response') is None:
                        print(f"Skipping failed item {p_id}")
                        continue

                    content_str = data['response']['body']['choices'][0]['message']['content']

                    # Cleanup markdown
                    if content_str.startswith('```json'):
                        content_str = content_str.replace('```json', '', 1)
                        if content_str.endswith('```'):
                            content_str = content_str[:-3]
                    elif content_str.startswith('```'):
                        content_str = content_str.replace('```', '', 1)
                        if content_str.endswith('```'):
                            content_str = content_str[:-3]

                    try:
                        content = json.loads(content_str.strip())
                    except json.JSONDecodeError:
                        print(f"JSON decode error for {p_id}")
                        continue

                    observations = content.get('observations', [])

                    # Inject value_type
                    for obs in observations:
                        obs_id = str(obs.get('id')).lstrip('0') or '0'
                        # Handle case where schema might use '03' or '3'.
                        # Actually schema uses '3'. Input might be '03'.
                        # Let's normalize both schema map keys and input ids to stripped strings.

                        # Wait, schema map keys are already loaded. I should normalize them in load_schema too.
                        # But simpler is to check both `obs_id` (original) and `obs_id_stripped`.

                        if obs_id in schema_map:
                            if 'value_type' not in obs:
                                obs['value_type'] = schema_map[obs_id]
                        else:
                            # Try with original just in case schema has leading zeros
                            raw_id = str(obs.get('id'))
                            if raw_id in schema_map:
                                if 'value_type' not in obs:
                                    obs['value_type'] = schema_map[raw_id]

                    out_item = {
                        'id': p_id.replace('req-', ''),
                        'observations': observations
                    }
                    fout.write(json.dumps(out_item) + '\n')
                    count += 1
                else:
                    # Maybe it's already in the right format but missing value_type?
                    # Check if it has 'id' and 'observations'
                    if 'id' in data and 'observations' in data:
                        observations = data['observations']
                        for obs in observations:
                            obs_id = str(obs.get('id')).lstrip('0') or '0'
                            raw_id = str(obs.get('id'))

                            found_type = schema_map.get(obs_id) or schema_map.get(raw_id)

                            if found_type and 'value_type' not in obs:
                                obs['value_type'] = found_type

                        fout.write(json.dumps(data) + '\n')
                        count += 1
                    else:
                        print(f"Skipping unknown format line")
            except Exception as e:
                print(f"Error processing line: {e}")

    print(f"Converted {count} items.")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python reformat_ablation_predictions.py schema.json input_file output_file")
        sys.exit(1)

    schema_map = load_schema(sys.argv[1])
    convert(sys.argv[2], sys.argv[3], schema_map)
