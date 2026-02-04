import os
import sys
import json
from typing import List, Dict, Tuple
from qdrant_client import QdrantClient
from openai import OpenAI

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import MODELS, get_api_key

# Config
COLLECTION_NAME = "synur_train"
EMBEDDING_MODEL_KEY = "dw_qwen_embedding"
QDRANT_URL_ENV = "QDRANT_URL"
QDRANT_API_KEY_ENV = "QDRANT_API_KEY"

class Retriever:
    def __init__(self):
        # 1. Connect to Qdrant
        url = os.getenv(QDRANT_URL_ENV)
        key = os.getenv(QDRANT_API_KEY_ENV)
        if not url or not key:
            raise ValueError("QDRANT_URL or QDRANT_API_KEY not set")
        self.q_client = QdrantClient(url=url, api_key=key)

        # 2. Connect to Doubleword (Synchronous for Query)
        cfg = MODELS[EMBEDDING_MODEL_KEY]
        self.emb_model_id = cfg.model_id
        api_key = get_api_key(cfg.api_key_env)

        self.openai_client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key
        )

    def embed_query(self, text: str) -> List[float]:
        try:
            resp = self.openai_client.embeddings.create(
                model=self.emb_model_id,
                input=text,
                encoding_format="float"
            )
            return resp.data[0].embedding
        except Exception as e:
            print(f"Error embedding query: {e}")
            return []

    def get_similar_examples(self, transcript: str, k: int = 5) -> List[Dict]:
        """
        Returns k similar examples as a list of dicts: {'transcript': ..., 'observations': ...}
        """
        # 1. Embed
        vector = self.embed_query(transcript)
        if not vector:
            return []

        # 2. Search
        try:
            results = self.q_client.search(
                collection_name=COLLECTION_NAME,
                query_vector=vector,
                limit=k,
                with_payload=True
            )
        except Exception as e:
            print(f"Error querying Qdrant: {e}")
            return []

        # 3. Format
        examples = []
        for point in results:
            payload = point.payload
            ex_transcript = payload.get('transcript')
            ex_obs_json = payload.get('observations_json', '[]')

            try:
                ex_obs = json.loads(ex_obs_json)
                if isinstance(ex_obs, str):
                    ex_obs = json.loads(ex_obs)
            except:
                ex_obs = []

            examples.append({
                "transcript": ex_transcript,
                "observations": ex_obs
            })

        return examples
