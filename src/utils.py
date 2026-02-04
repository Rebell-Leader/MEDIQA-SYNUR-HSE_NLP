import json
import os
from typing import List, Dict, Any

def load_jsonl(path: str) -> List[Dict]:
    """Load JSONL file"""
    with open(path, 'r') as f:
        return [json.loads(line) for line in f]

def load_schema(path: str) -> List[Dict]:
    """Load schema JSON"""
    with open(path, 'r') as f:
        return json.load(f)

def save_jsonl(data: List[Dict], path: str):
    """Save list of dicts to JSONL"""
    with open(path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def build_system_prompt(schema: List[Dict]) -> str:
    # Convert schema to string for prompt inclusion
    schema_str = json.dumps(schema, indent=2)

    prompt = f"""You are a clinical documentation expert extracting structured observations from nurse dictations.

TASK: Extract all clinical observations mentioned in the transcript and map them to the given ontology.

OUTPUT FORMAT: Return valid JSON object with an "observations" array. Each observation must include:
- id: unique observation ID from ontology (string)
- value_type: one of SINGLE_SELECT, MULTI_SELECT, STRING, NUMERIC
- name: observation name from ontology
- value: extracted value (matching value_type)

RULES:
1. Extract ONLY observations explicitly mentioned in the transcript.
2. Match observation names and IDs to the ontology exactly.
3. For SINGLE_SELECT: Value MUST be an exact string match to one of the options in 'value_enum'.
4. For MULTI_SELECT: Value MUST be a JSON List of strings, even if only one option is selected. Example: ["Symptom"], NOT "Symptom". All items must match 'value_enum'.
5. For NUMERIC: extract numbers only; NO units in value field.
6. For STRING: extract the full phrase exactly as spoken in transcript.
7. Return empty array if no observations found: {{"observations": []}}
8. Do NOT hallucinate observations or values not mentioned in transcript.
9. Ensure valid JSON output.

ONTOLOGY:
{schema_str}
"""
    return prompt

def build_user_prompt(transcript: str) -> str:
    return f"Transcript:\n{transcript}\n\nExtract observations."

def validate_observation(obs: Dict, schema: List[Dict]) -> bool:
    """
    Basic local validation against schema.
    Returns True if valid, False otherwise.
    """
    if not isinstance(obs, dict):
        return False

    schema_map = {item['id']: item for item in schema}

    obs_id = obs.get('id')
    if obs_id not in schema_map:
        return False

    schema_item = schema_map[obs_id]
    val_type = schema_item['value_type']
    val = obs.get('value')

    if val_type == 'NUMERIC':
        if isinstance(val, (int, float)): return True
        try:
            float(val)
            return True
        except:
            return False

    elif val_type == 'SINGLE_SELECT':
        enums = schema_item.get('value_enum', [])
        return val in enums

    elif val_type == 'MULTI_SELECT':
        if not isinstance(val, list): return False
        enums = set(schema_item.get('value_enum', []))
        return all(v in enums for v in val)

    return True

def sanitize_observation(obs: Dict, schema_map: Dict) -> Dict:
    """
    Cleans up common model formatting errors:
    1. Stringified lists (e.g. "['foo', 'bar']") for MULTI_SELECT.
    2. Lists for SINGLE_SELECT (e.g. ["foo"] -> "foo").
    3. Stringified lists for SINGLE_SELECT.
    """
    if not isinstance(obs, dict): return obs

    obs_id = obs.get('id')
    if obs_id not in schema_map: return obs

    schema_item = schema_map[obs_id]
    val_type = schema_item['value_type']
    val = obs.get('value')

    # Handle Stringified JSON first
    if isinstance(val, str) and val.strip().startswith('[') and val.strip().endswith(']'):
        try:
            parsed = json.loads(val)
            val = parsed
        except:
            pass # Not valid JSON, keep as is

    # Now handle type mismatches
    if val_type == 'MULTI_SELECT':
        if isinstance(val, str):
            # If still string (and not parsed above or was plain string), wrap in list?
            # Or if it was "Symptom", make it ["Symptom"]
            val = [val]
        elif isinstance(val, (int, float)):
             val = [str(val)]

    elif val_type == 'SINGLE_SELECT':
        if isinstance(val, list):
            if len(val) == 1:
                val = val[0]
            elif len(val) == 0:
                val = None # Invalid
            else:
                 # Taking first? or joining? Schema implies single.
                 # Let's take first for now or let validation fail.
                 val = val[0]

    new_obs = obs.copy()

    # --- SPECIFIC FIXES ---
    # ID 179: Temperature unit (Schema has encoding artifacts)
    if obs_id == "179" and isinstance(val, str):
        if val == "°C" or val == "C":
            val = "\u00c2\u00b0C"
        elif val == "°F" or val == "F":
            val = "\u00c2\u00b0F"

    # ID 37: Motor strength - Map missing numeric scales to nearest if safe?
    # Schema has: '2 out of 5', '5 out of 5'. Missing 3, 4.
    # If model says '3 out of 5', arguably 'Weak'? But let's leave for now.

    new_obs['value'] = val
    return new_obs

def inject_schema_details(observations: List[Dict], schema: List[Dict]) -> List[Dict]:
    """
    Injects 'value_type' and 'name' from schema into observations based on 'id'.
    Also Sanitizes the values.
    """
    schema_map = {item['id']: item for item in schema}
    enriched = []

    for obs in observations:
        if not isinstance(obs, dict):
            # Keep as is, will fail validation but prevents crash
            enriched.append(obs)
            continue

        obs_id = obs.get('id')
        if obs_id in schema_map:
            # Create new dict to avoid mutating original if needed, or just update
            # SANITIZE FIRST
            new_obs = sanitize_observation(obs, schema_map)

            # Enforce types from schema
            new_obs['value_type'] = schema_map[obs_id]['value_type']
            new_obs['name'] = schema_map[obs_id]['name']
            enriched.append(new_obs)
        else:
            # If ID invalid, DROP IT.
            # Keeping it harms Precision/F1 if it's a hallucination (ID 210+).
            # The official evaluator strictly penalizes false positives.
            pass

    return enriched

def build_messages(schema: List[Dict], transcript: str, examples: List[Dict] = None, expert_hints: List[Dict] = None) -> List[Dict]:
    """
    Constructs the full list of messages for Chat Completion.
    Supports optional few-shot examples and ensemble expert hints.
    """
    messages = [
        {"role": "system", "content": build_system_prompt(schema)}
    ]

    if examples:
        for ex in examples:
            messages.append({"role": "user", "content": build_user_prompt(ex['transcript'])})
            mock_output = {"observations": ex.get('observations', [])}
            messages.append({"role": "assistant", "content": json.dumps(mock_output)})

    # Target Input
    user_content = build_user_prompt(transcript)

    if expert_hints:
        hint_str = "\n\n### EXPERT HINTS (Draft Extractions from other systems):\n"
        for hint in expert_hints:
            hint_str += f"- System [{hint['name']}] (Estimated F1: {hint['f1']}):\n"
            hint_str += f"  Extracted: {json.dumps(hint['observations'])}\n"

        user_content += hint_str
        user_content += "\n\nINSTRUCTION: Consider the expert hints above to resolve conflicts and ensure maximum accuracy. Extract the final structured observations."

    messages.append({"role": "user", "content": user_content})

    return messages

def build_adjudicator_messages(schema: List[Dict], transcript: str, expert_a: Dict, expert_b: Dict) -> List[Dict]:
    """
    Builds a prompt for a 'Master Resolver' model to adjudicate between two candidate extractions.
    """
    system_prompt = "You are a senior medical consultant and data extraction expert. Your task is to produce the most accurate and complete extraction of medical observations from a patient-clinician transcript."
    messages = [{"role": "system", "content": system_prompt}]

    user_content = f"TRANSCRIPT:\n{transcript}\n\n"
    user_content += "Below are two candidate extractions from high-performing models:\n"
    user_content += f"DRAFT A (Model: {expert_a['name']}, Est. F1: {expert_a['f1']}):\n{json.dumps(expert_a['observations'])}\n\n"
    user_content += f"DRAFT B (Model: {expert_b['name']}, Est. F1: {expert_b['f1']}):\n{json.dumps(expert_b['observations'])}\n\n"

    user_content += """YOUR TASK:
1. Synthesize these lists into a single, high-fidelity extraction.
2. If the models agree on an ID but differ in the extracted value, consult the TRANSCRIPT carefully to decide which value is more accurate and complete.
3. If an observation appears in only one list, verify its presence in the transcript. If it is supported, include it. If it is a hallucination not found in the text, remove it.
4. Remove all duplicates.
5. Strictly adhere to the output JSON schema: {"observations": [{"id": "...", "value": "..."}, ...]}.
6. Ensure all IDs map correctly to the standard schema IDs.

Output ONLY the final JSON object."""

    messages.append({"role": "user", "content": user_content})
    return messages

DATASETS = {
    "test": "SYNUR_testset_input.jsonl",
    "train": "train.jsonl",
    "dev": "dev.jsonl"
}

def load_dataset_by_name(name: str) -> List[Dict]:
    """
    Loads dataset by name (train/dev/test).
    Handles fallback for 'dev' -> train[:200] if dev.jsonl missing.
    """
    path = DATASETS.get(name)
    if not path:
        raise ValueError(f"Unknown dataset name: {name}")

    if name == 'dev' and not os.path.exists(path):
        # Fallback: check train
        train_path = DATASETS['train']
        if os.path.exists(train_path):
            print(f"Dataset '{name}' not found at {path}. Loading subset from {train_path}...")
            full = load_jsonl(train_path)
            return full[:200]

    if not os.path.exists(path):
         raise ValueError(f"Dataset file not found: {path}")

    print(f"Loading data from {path}...")
    return load_jsonl(path)

def filter_schema_by_ids(schema: List[Dict], target_ids: List[str]) -> List[Dict]:
    """
    Returns a subset of the schema containing only definitions for target_ids.
    Used for Targeted Repair to reduce context size.
    """
    target_set = set(target_ids)
    return [item for item in schema if item['id'] in target_set]
