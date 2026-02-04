# %%
import sys
import os
import json
from openai import OpenAI
import requests

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import MODELS, get_api_key

# %%
def test_openai():
    print("--- Testing OpenAI ---")
    cfg = MODELS["openai_nano"]
    try:
        api_key = get_api_key(cfg.api_key_env)
        client = OpenAI(api_key=api_key)

        response = client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=10
        )
        print(f"Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed: {e}")

# %%
def test_hf():
    print("\n--- Testing HuggingFace Router ---")
    cfg = MODELS["hf_qwen"]
    try:
        api_key = get_api_key(cfg.api_key_env)
        # HF Router uses OpenAI client structure
        client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key
        )

        response = client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=10
        )
        print(f"Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed: {e}")

# %%
def test_doubleword():
    print("\n--- Testing Doubleword ---")
    cfg = MODELS["dw_qwen"]
    try:
        api_key = get_api_key(cfg.api_key_env)
        # Doubleword also follows OpenAI format usually, but we can verify
        client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key
        )

        response = client.chat.completions.create(
            model=cfg.model_id,
            messages=[{"role": "user", "content": "Say hello!"}],
            max_tokens=10
        )
        print(f"Success: {response.choices[0].message.content}")
    except Exception as e:
        print(f"Failed: {e}")

# %%
if __name__ == "__main__":
    test_openai()
    test_hf()
    test_doubleword()
