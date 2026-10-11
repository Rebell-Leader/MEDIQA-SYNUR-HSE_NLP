import os
import sys
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

load_dotenv()

from src.retrieval import Retriever

try:
    print("Initializing Retriever...")
    r = Retriever()
    print("Testing retrieval...")
    res = r.get_similar_examples("Patient has a history of cough and fever.", k=1)
    print(f"Retrieved {len(res)} examples.")
    if res:
        print("Success!")
        print(f"First example transcript snippet: {res[0]['transcript'][:50]}...")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback
    traceback.print_exc()
