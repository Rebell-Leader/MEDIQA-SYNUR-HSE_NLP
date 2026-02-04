import json
import time
import os
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from openai import OpenAI
from ..batch_manager import BatchManager
from ..config import ModelConfig, get_api_key
from ..utils import build_system_prompt, build_user_prompt, build_messages

class OpenAIAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = get_api_key(config.api_key_env)
        self.client = OpenAI(api_key=self.api_key)
        self.batch_manager = BatchManager()

    def _prepare_request_body(self, messages: List[Dict], schema_json: Optional[Dict]) -> Dict:
        """Helper to construct request body with model-specific constraints."""
        body = {
            "model": self.config.model_id,
            "messages": messages,
        }

        # Temperature is only supported for GPT-4o family; removed for GPT-5
        if "gpt-5" not in self.config.model_id:
            body["temperature"] = 0.0

        if self.config.supports_structured_output and schema_json:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": schema_json
            }
        return body

    def create_batch_file(self, items: List[Dict], schema: List[Dict], output_path: str, examples: List[Dict] = None, expert_hints_map: Dict[str, List[Dict]] = None) -> str:
        """
        Creates a JSONL file formatted for OpenAI/Doubleword Batch API.
        """
        schema_json = None
        if self.config.supports_structured_output:
             # Define Schema once (same for all requests)
             schema_json = {
                 "name": "observations_extraction",
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
                                    "value": {
                                        # Strict schema limitation
                                        "type": ["string", "number", "boolean", "null"]
                                    }
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

        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for item in items:
                custom_id = str(item['id'])
                transcript = item['transcript']

                # Check for RAG examples (dynamic) or Fallback to static
                ex = item.get('_dynamic_examples') or examples

                # Expert Hints (Ensemble or Adjudication)
                hints = item.get('_expert_hints') or (expert_hints_map.get(custom_id) if expert_hints_map else None)

                if hints and len(hints) >= 2:
                    from ..utils import build_adjudicator_messages
                    messages = build_adjudicator_messages(schema, transcript, hints[0], hints[1])
                else:
                    messages = build_messages(schema, transcript, ex, hints)

                # Use helper to build body
                body = self._prepare_request_body(messages, schema_json)

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(request) + '\n')

        return output_path

    def _parse_batch_results(self, output_file: str) -> Dict[str, List[Dict]]:
        """Parses the downloaded batch result file."""
        print(f"Parsing results from {output_file}...")
        with open(output_file, 'r') as f:
            content = f.read()

        results_map = {}
        for line in content.split('\n'):
            if not line.strip(): continue
            item = json.loads(line)
            custom_id = item['custom_id']
            # Remove "req-" prefix if present (legacy) or just use as is if we switched to IDs
            # But wait, now I'm sending item['id'] as custom_id. item['id'] is string "152".
            # Batch API returns it exactly.

    def _parse_json_content(self, content: str) -> List[Dict]:
        """Robustly extracts JSON list from potentially markdown-wrapped model output."""
        if not content:
            return []

        # 1. Try direct parse
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed.get("observations", [])
            elif isinstance(parsed, list):
                return parsed
        except:
            pass

        # 2. Extract from markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            content = parts[1].strip() if len(parts) >= 3 else parts[0].strip()

        # 3. Clean and try again
        try:
            parsed = json.loads(content)
            if isinstance(parsed, dict):
                return parsed.get("observations", [])
            elif isinstance(parsed, list):
                return parsed
        except:
            # Last resort: try to find the first '{' and last '}'
            try:
                if "{" in content:
                    content_clean = "{" + content.split("{", 1)[1].rsplit("}", 1)[0] + "}"
                    parsed = json.loads(content_clean)
                    return parsed.get("observations", [])
            except:
                pass

        return []

    def _parse_batch_results(self, output_file: str) -> Dict[str, List[Dict]]:
        """Parses the downloaded batch result file."""
        print(f"Parsing results from {output_file}...")
        results_map = {}
        with open(output_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    custom_id = item['custom_id']
                    resp_body = item['response']['body']
                    msg_content = resp_body['choices'][0]['message']['content']
                    results_map[custom_id] = self._parse_json_content(msg_content)
                except Exception as e:
                    print(f"Error parsing line in {output_file}: {e}")
                    continue

        return results_map

    def predict_direct(self, items: List[Dict], schema: List[Dict], examples: List[Dict] = None) -> List[List[Dict]]:
        """
        Direct synchronous inference (no batch API) for debugging.
        """
        print(f"Starting DIRECT inference with {self.config.model_id}...")
        predictions = []

        schema_json = None
        if self.config.supports_structured_output:
             schema_json = {
                 "name": "observations_extraction",
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

        for item in tqdm(items):
            transcript = item['transcript']
            messages = build_messages(schema, transcript, examples)

            kwargs = self._prepare_request_body(messages, schema_json)

            try:
                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content
                predictions.append(self._parse_json_content(content))
            except Exception as e:
                print(f"Error: {e}")
                predictions.append([])

        return predictions

    def predict_batch(self, items: List[Dict], schema: List[Dict], use_batch_api: bool = False, experiment_name: str = "debug_run", examples: List[Dict] = None) -> List[List[Dict]]:
        """
        Facade for inference.
        If use_batch_api=True, submits via BatchManager and waits (blocking).
        If use_batch_api=False, uses predict_direct.
        """
        if use_batch_api and self.config.supports_batch_api:
            # 1. Prepare Batch File in strict location
            batch_filename = f"batch_input_{self.config.model_id}_{experiment_name}_{int(time.time())}.jsonl"
            batch_path = os.path.join("outputs", "batches", batch_filename)
            self.create_batch_file(items, schema, batch_path, examples)

            # 2. Submit via Manager
            batch_id = self.batch_manager.submit_batch(self.client, batch_path, self.config.model_id, experiment_name)

            # 3. Wait for Completion (Blocking Logic for Pipeline Compatibility)
            print(f"Waiting for batch {batch_id} to complete...")
            while True:
                # Update statuses (encapsulated in Manager)
                self.batch_manager.update_statuses(self.client)

                # Check our specific batch status from the manager's state
                info = self.batch_manager.tracker.get(batch_id)
                if not info:
                    raise RuntimeError("Batch tracking lost.")

                status = info['status']
                if status == 'completed':
                    break
                elif status in ['failed', 'expired', 'cancelled']:
                    raise RuntimeError(f"Batch failed: {status}")

                print(f"Batch {batch_id} status: {status}... waiting 30s")
                time.sleep(30)

            # 4. Retrieve content
            result_filename = f"results_{self.config.model_id}_{experiment_name}.jsonl"
            result_path = os.path.join(self.batch_manager.output_dir, result_filename)

            res_map = self._parse_batch_results(result_path)
            # Re-align to input list order
            return [res_map.get(item['id'], []) for item in items]

        else:
            return self.predict_direct(items, schema, examples)
