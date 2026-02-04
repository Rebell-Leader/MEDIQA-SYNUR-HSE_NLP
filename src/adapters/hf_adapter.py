import json
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from ..config import ModelConfig, get_api_key
from ..utils import build_system_prompt, build_user_prompt

class HFAdapter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = get_api_key(config.api_key_env)
        if not self.config.base_url:
             raise ValueError("HF config requires base_url")

        self.client = OpenAI(
            base_url=self.config.base_url,
            api_key=self.api_key
        )

    def predict_batch(self, transcripts: List[str], schema: List[Dict]) -> List[List[Dict]]:
        predictions = []
        print(f"Starting inference with {self.config.model_id} (HF Router)...")

        for transcript in tqdm(transcripts):
            try:
                messages = [
                    {"role": "system", "content": build_system_prompt(schema)},
                    {"role": "user", "content": build_user_prompt(transcript)}
                ]

                kwargs = {
                    "model": self.config.model_id,
                    "messages": messages,
                    "temperature": 0.0,
                }

                # HF models might not support response_format at all
                # We rely on system prompt to get JSON

                response = self.client.chat.completions.create(**kwargs)
                content = response.choices[0].message.content

                if content:
                    clean_content = content
                    if "```json" in clean_content:
                        clean_content = clean_content.split("```json")[1].split("```")[0].strip()
                    elif "```" in clean_content:
                        clean_content = clean_content.split("```")[1].split("```")[0].strip()

                    # Try to find JSON start/end if extra text exists
                    start = clean_content.find('{')
                    end = clean_content.rfind('}') + 1
                    if start >= 0 and end > start:
                        clean_content = clean_content[start:end]

                    parsed = json.loads(clean_content)
                    observations = parsed.get("observations", [])
                    # If model returns list directly (rare but possible), handle it
                    if isinstance(parsed, list):
                        observations = parsed

                    predictions.append(observations)
                else:
                    predictions.append([])

            except Exception as e:
                print(f"Error processing transcript: {e}")
                predictions.append([])

        return predictions
