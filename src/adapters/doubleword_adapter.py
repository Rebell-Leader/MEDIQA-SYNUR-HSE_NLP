import json
import time
import os
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from ..batch_manager import BatchManager
from ..config import ModelConfig, get_api_key
from ..utils import build_system_prompt, build_user_prompt, build_messages

class DoublewordAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = get_api_key(config.api_key_env)
        if not self.config.base_url:
             raise ValueError("Doubleword requires base_url")

        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.api_key
        )
        self.batch_manager = BatchManager()

    def create_batch_file(self, items: List[Dict], schema: List[Dict], output_path: str, examples: List[Dict] = None) -> str:
        """
        Doubleword compatible batch file.
        Uses json_object instead of strict json_schema.
        """
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for item in items:
                custom_id = item['id']
                transcript = item['transcript']

                # Check for RAG examples (dynamic) or Fallback to static
                ex = item.get('_dynamic_examples') or examples

                # Expert Hints (Adjudication or Ensemble)
                hints = item.get('_expert_hints')
                if hints and len(hints) >= 2:
                    # Use Adjudicator prompt if we have 2 experts
                    from ..utils import build_adjudicator_messages
                    messages = build_adjudicator_messages(schema, transcript, hints[0], hints[1])
                else:
                    messages = build_messages(schema, transcript, ex, hints)

                body = {
                    "model": self.config.model_id,
                    "messages": messages,
                    "temperature": 0.0,
                }

                if self.config.supports_structured_output:
                     body["response_format"] = {"type": "json_object"}

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": body
                }
                f.write(json.dumps(request) + '\n')
        return output_path

    def create_embedding_batch_file(self, items: List[Dict], output_path: str) -> str:
        """
        Creates a batch file for embeddings.
        Input items must have 'id' and 'transcript'.
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as f:
            for item in items:
                custom_id = str(item['id'])
                transcript = item['transcript']

                body = {
                    "model": "Qwen/Qwen3-Embedding-8B", # hardcoded or from config? Use config if possible but this method might be generic.
                    "input": transcript,
                    "encoding_format": "float"
                }

                request = {
                    "custom_id": custom_id,
                    "method": "POST",
                    "url": "/v1/embeddings",
                    "body": body
                }
                f.write(json.dumps(request) + '\n')
        return output_path

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
        """Parses the downloaded batch result file (Doubleword specific parsing)."""
        print(f"Parsing results from {output_file}...")
        results_map = {}
        with open(output_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    item = json.loads(line)
                    custom_id = item['custom_id']

                    resp_body = item['response']['body']

                    # CHECK FOR EMBEDDING RESPONSE
                    if 'data' in resp_body and isinstance(resp_body['data'], list) and 'embedding' in resp_body['data'][0]:
                        embedding = resp_body['data'][0]['embedding']
                        results_map[custom_id] = embedding
                        continue

                    # Standard Chat Completion Parsing
                    msg_content = resp_body['choices'][0]['message']['content']
                    results_map[custom_id] = self._parse_json_content(msg_content)
                except Exception as e:
                    print(f"Error parsing line in {output_file}: {e}")
                    continue

        return results_map

    def predict_direct(self, items: List[Dict], schema: List[Dict], examples: List[Dict] = None) -> List[List[Dict]]:
        """Direct synchronous inference."""
        print(f"Starting DIRECT inference with {self.config.model_id}...")
        predictions = []

        for item in tqdm(items):
            transcript = item['transcript']
            messages = build_messages(schema, transcript, examples)

            # Doubleword: json_object format
            extra_body = {}
            if self.config.supports_structured_output:
                extra_body["response_format"] = {"type": "json_object"}

            try:
                response = self.client.chat.completions.create(
                    model=self.config.model_id,
                    messages=messages,
                    temperature=0.0,
                    **extra_body
                )

                content = response.choices[0].message.content
                predictions.append(self._parse_json_content(content))
            except Exception as e:
                print(f"Error: {e}")
                predictions.append([])

        return predictions

    def predict_batch(self, items: List[Dict], schema: List[Dict], use_batch_api: bool = False, experiment_name: str = "debug_run_dw", examples: List[Dict] = None) -> List[List[Dict]]:
        """
        Facade for inference using BatchManager.
        """
        if use_batch_api and self.config.supports_batch_api:
            # 1. Prepare Batch File
            batch_filename = f"batch_input_{self.config.model_id}_{experiment_name}_{int(time.time())}.jsonl"
            batch_path = os.path.join("outputs", "batches", batch_filename)
            self.create_batch_file(items, schema, batch_path, examples)

            # 2. Submit via Manager
            batch_id = self.batch_manager.submit_batch(self.client, batch_path, self.config.model_id, experiment_name)

            # 3. Wait (Blocking) for compatibility
            print(f"Waiting for batch {batch_id} to complete...")
            while True:
                self.batch_manager.update_statuses(self.client)
                info = self.batch_manager.tracker.get(batch_id)
                if not info: raise RuntimeError("Batch tracking lost.")

                status = info['status']
                if status == 'completed': break
                elif status in ['failed', 'expired', 'cancelled']: raise RuntimeError(f"Batch failed: {status}")

                print(f"Batch {batch_id} status: {status}... waiting 30s")
                time.sleep(30)

            # 4. Retrieve
            result_filename = f"results_{self.config.model_id}_{experiment_name}.jsonl"
            result_path = os.path.join(self.batch_manager.output_dir, result_filename)
            res_map = self._parse_batch_results(result_path)
            # Align
            return [res_map.get(item['id'], []) for item in items]
        else:
            return self.predict_direct(items, schema, examples)
