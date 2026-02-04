import os
import sys
import json
import time
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Add project root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.config import MODELS, get_api_key
from src.adapters.doubleword_adapter import DoublewordAdapter
from src.utils import load_jsonl

# Qdrant Config
QDRANT_URL_ENV = "QDRANT_URL"
QDRANT_API_KEY_ENV = "QDRANT_API_KEY"
COLLECTION_NAME = "synur_train"
VECTOR_SIZE = 4096  # Qwen/Qwen3-Embedding-8B dimension

def get_qdrant_client():
    url = os.getenv(QDRANT_URL_ENV)
    key = os.getenv(QDRANT_API_KEY_ENV)
    if not url or not key:
        raise ValueError("QDRANT_URL or QDRANT_API_KEY not set in .env")
    return QdrantClient(url=url, api_key=key)

def process_embeddings(adapter: DoublewordAdapter, train_data: List[Dict]):
    # 1. Check for existing embeddings
    output_dir = "outputs/embeddings"
    os.makedirs(output_dir, exist_ok=True)
    embeddings_file = os.path.join(output_dir, "train_embeddings.jsonl")
    result_path = os.path.join("outputs", f"results_dw_qwen_embedding_train_embed.jsonl")

    if os.path.exists(embeddings_file) or os.path.exists(result_path):
        if os.path.exists(result_path):
            print(f"Found existing results at {result_path}. Skipping submission.")
        else:
            print(f"Loading existing embeddings from {embeddings_file}...")
        pass
    else:
        print("Generating embeddings via Batch API...")
        # Create Batch
        batch_filename = f"batch_embed_train_{int(time.time())}.jsonl"
        batch_path = os.path.join("outputs", "batches", batch_filename)
        adapter.create_embedding_batch_file(train_data, batch_path)

        # Submit
        batch_id = adapter.batch_manager.submit_batch(adapter.client, batch_path, "dw_qwen_embedding", "train_embed")
        print(f"Batch submitted: {batch_id}. Waiting for completion...")

        # Wait loop
        while True:
            adapter.batch_manager.update_statuses(adapter.client)
            info = adapter.batch_manager.tracker.get(batch_id)
            if not info: raise RuntimeError("Batch tracking lost")

            if info['status'] == 'completed':
                print("Batch completed. Downloading...")
                adapter.batch_manager._download_results(adapter.client, batch_id, info)
                # Results are in outputs/results_...jsonl
                # We need to move/rename or just use that file
                break
            elif info['status'] in ['failed', 'expired', 'cancelled']:
                raise RuntimeError(f"Batch failed: {info['status']}")

            print(f"Status: {info['status']}... waiting 30s")
            time.sleep(30)

    # 2. Parse Results
    # Find the result file from tracker or predictable name
    # We can infer it: results_dw_qwen_embedding_train_embed.jsonl.
    # But wait, experiment name "train_embed" is constant so multiple runs might conflict if using same name?
    # BatchManager handles tracking.
    # Let's grab the latest completed batch for this experiment or explicit path.
    # We can just re-parse the predictable path:
    result_path = os.path.join("outputs", f"results_dw_qwen_embedding_train_embed.jsonl")

    if not os.path.exists(result_path):
        # Maybe it has a different name? Adapter save logic: results_{model}_{experiment}.jsonl
        pass

    embeddings_map = adapter._parse_batch_results(result_path)
    print(f"Loaded {len(embeddings_map)} embeddings.")
    return embeddings_map

def upsert_to_qdrant(client: QdrantClient, train_data: List[Dict], embeddings_map: Dict):
    print(f"Checking collection {COLLECTION_NAME}...")
    try:
        client.get_collection(COLLECTION_NAME)
        exists = True
    except Exception:
        exists = False

    if not exists:
        print(f"Creating collection {COLLECTION_NAME} (size={VECTOR_SIZE})...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )
    else:
        print(f"Collection {COLLECTION_NAME} exists. Upserting data...")

    points = []
    skipped = 0
    for item in train_data:
        str_id = str(item['id'])
        if str_id not in embeddings_map:
            skipped += 1
            continue

        vector = embeddings_map[str_id]

        # Payload: Transcript + maybe the ground truth observations?
        # RAG needs to show examples: Transcript -> Observations
        payload = {
            "transcript": item['transcript'],
            "observations_json": json.dumps(item.get('observations', []))
        }

        # Qdrant ID: try to use integer ID if possible, else uuid.
        # Our IDs are likely integers or strings. Qdrant supports int or uuid.
        # If item['id'] is int-like, use it.
        try:
            p_id = int(str_id)
        except:
            # Hash it or use uuid
            import uuid
            p_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str_id))

        points.append(PointStruct(id=p_id, vector=vector, payload=payload))

    if points:
        print(f"Upserting {len(points)} points...")
        # Batch upsert
        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print("Upsert complete.")
    else:
        print("No points to upsert.")

    if skipped:
        print(f"Warning: Skipped {skipped} items due to missing embeddings.")

def main():
    # Load Train Data
    train_data = load_jsonl("train.jsonl")
    print(f"Loaded {len(train_data)} training examples.")

    # Init Adapter
    cfg = MODELS["dw_qwen_embedding"]
    adapter = DoublewordAdapter(cfg)

    # 1. Get Embeddings (Batch)
    embeddings_map = process_embeddings(adapter, train_data)

    # 2. Upsert to Qdrant
    if embeddings_map:
        q_client = get_qdrant_client()
        upsert_to_qdrant(q_client, train_data, embeddings_map)

if __name__ == "__main__":
    main()
