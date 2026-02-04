import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

print("--- OpenAI Batches ---")
batches = client.batches.list(limit=10)
for b in batches.data:
    print(f"ID: {b.id} | Status: {b.status} | Output: {b.output_file_id} | Errors: {b.errors}")
    if b.status == "completed" and b.output_file_id:
        print(f"  -> Ready for download")
    elif b.status == "failed":
        print(f"  -> FAILED: {b.errors}")
