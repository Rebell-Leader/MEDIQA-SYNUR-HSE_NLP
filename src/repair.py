import json
from typing import List, Dict, Any, Tuple
from .utils import validate_observation, load_schema, build_system_prompt

REPAIR_SYSTEM_PROMPT = """You are a clinical data quality specialist.
Your task is to fix malformatted or invalid observations extracted from a transcript.

INPUT:
- Transcript
- Invalid Observations (with error details)
- Ontology (Schema)

OUTPUT:
Return strict JSON with the corrected "observations" list.
"""

def build_repair_user_prompt(transcript: str, invalid_obs: List[Dict], errors: List[str]) -> str:
    feedbacks = []
    for obs, err in zip(invalid_obs, errors):
        feedbacks.append(f"Observation ID '{obs.get('id')}': {err}. Value was: {obs.get('value')}")

    feedback_str = "\n".join(feedbacks)

    return f"""TRANSCRIPT:
{transcript}

PREVIOUS (INVALID) OBSERVATIONS:
{json.dumps(invalid_obs, indent=2)}

ERRORS FOUND:
{feedback_str}

Please correct these observations based on the transcript and ontology. Return the full list of valid observations."""

def identify_failures(predictions: List[List[Dict]], transcripts: List[str], schema: List[Dict]) -> Tuple[List[int], List[Dict]]:
    """
    Returns:
        failed_indices: list of indices in the original prediction list that failed validation.
        repair_requests: list of dicts {index, transcript, invalid_obs, errors}
    """
    failed_indices = []
    repair_requests = []

    for idx, (preds, transcript) in enumerate(zip(predictions, transcripts)):
        invalid_obs = []
        errors = []

        for obs in preds:
            if not isinstance(obs, dict):
                 # Invalid format entirely
                 errors.append(f"Observation is not a dictionary. Value was: {str(obs)}")
                 invalid_obs.append({"error": "Invalid Format (Not a Dict)", "data": str(obs)})
                 continue

            if not validate_observation(obs, schema):
                # Try to determine why (simple re-check)
                val = obs.get('value')
                item = next((i for i in schema if i['id'] == obs.get('id')), None)
                if not item:
                    errors.append(f"ID {obs.get('id')} not found in ontology")
                else:
                    errors.append(f"Value '{val}' does not match type {item['value_type']} or enum constraints")

                invalid_obs.append(obs)

        if invalid_obs:
            failed_indices.append(idx)
            repair_requests.append({
                "index": idx,
                "transcript": transcript,
                "invalid_obs": invalid_obs,
                "errors": errors
            })

    return failed_indices, repair_requests

from .config import ModelConfig

def create_repair_batch_file(repair_requests: List[Dict], schema: List[Dict], config: ModelConfig, output_path: str):
    """
    Creates a batch file specifically for fixing the identified failures.
    Uses ModelConfig to determine model_id and response_format.

    OPTIMIZATION (Targeted Repair):
    Filters schema to only include definitions for the specific IDs that failed validation.
    This drastically reduces prompt size vs sending full ontology.
    """
    from .utils import filter_schema_by_ids

    with open(output_path, 'w') as f:
        for req in repair_requests:
            custom_id = f"repair-{req['index']}"

            # Extract IDs from invalid_obs to filter schema
            # We want schema for the IDs that are present in the invalid set.
            # If the ID is missing from ontology (hallucination), we can't provide unknown schema,
            # but usually the ID exists and the Value is wrong.
            failed_ids = []
            for obs in req['invalid_obs']:
                if isinstance(obs, dict) and 'id' in obs:
                    failed_ids.append(str(obs['id']))

            # Remove duplicates
            failed_ids = list(set(failed_ids))

            if failed_ids:
                subset_schema = filter_schema_by_ids(schema, failed_ids)
            else:
                # Fallback: If no IDs found (e.g. format error entire dict),
                # we might have to send full schema or cannot repair specific ID?
                # If the error is "Invalid Format", maybe we send full schema?
                # For safety, if list empty, let's look at the errors.
                # Ideally we want to be targeted. If we don't know the ID, we can't help much.
                # Let's send full schema ONLY if we really can't narrow it down?
                # actually sending full schema causes crash.
                # BETTER: Send NO schema (empty list) if we didn't find any IDs
                # (meaning pure hallucination/garbage), or maybe just the first 5??
                # Let's try sending full schema as a risky fallback
                # OR just log warning and send subset (empty).
                # Given the overflow crash, empty/small is better than crash.
                subset_schema = []
                # OR: if we have NO valid IDs, maybe we should just not repair?
                # But let's assume if there are invalid_obs, there's something.

            # If subset_schema is empty, the model might struggle.
            # But the 'observations' in system prompt will be empty.
            # The model will see the "PREVIOUS (INVALID)" and "ERRORS".
            # If the error is "ID not found", we can't provide schema for it anyway.

            messages = [
                {"role": "system", "content": build_system_prompt(subset_schema) + "\n" + REPAIR_SYSTEM_PROMPT},
                {"role": "user", "content": build_repair_user_prompt(req['transcript'], req['invalid_obs'], req['errors'])}
            ]

            body = {
                "model": config.model_id,
                "messages": messages,
            }
            if "gpt-5" not in config.model_id:
                body["temperature"] = 0.0

            # Provider-specific logic
            if config.provider == "doubleword" and config.supports_structured_output:
                body["response_format"] = {"type": "json_object"}
            elif config.provider == "openai":
                # GPT-4o supports strict schema, but for repair strict might be too rigid or just right.
                # Let's use json_object for repair to be safe and flexible, or strict if we match Adapter.
                # Use JSON Schema Strict if supported for consistency with main pipeline
                if config.supports_structured_output:
                     body["response_format"] = {
                        "type": "json_schema",
                        "json_schema": {
                             "name": "observations_extraction_repair",
                             "strict": True,
                             "schema": {
                                "type": "object",
                                "properties": {
                                    "observations": {
                                        "type": "array",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "id": {"type": "string"},
                                                "value": {"type": ["string", "number", "boolean", "null"]}
                                            },
                                            "required": ["id", "value"],
                                            "additionalProperties": False
                                        }
                                    }
                                },
                                "required": ["observations"],
                                "additionalProperties": False
                            }
                        }
                    }

            request = {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body
            }
            f.write(json.dumps(request) + '\n')

    return output_path
