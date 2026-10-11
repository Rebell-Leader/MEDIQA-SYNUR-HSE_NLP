import json
import os
from src.config import MODELS
from src.adapters.openai_adapter import OpenAIAdapter
from src.adapters.doubleword_adapter import DoublewordAdapter

def test_connectivity():
    print("--- Testing API Connectivity ---")

    # Test OpenAI
    try:
        oa = OpenAIAdapter(MODELS["gpt-4o-mini"])
        resp = oa.client.chat.completions.create(
            model=MODELS["gpt-4o-mini"].model_id,
            messages=[{"role": "user", "content": "Say Hello"}],
            max_tokens=5
        )
        print(f"OpenAI (gpt-4o-mini) Success: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"OpenAI Connectivity Failed: {e}")

    # Test Doubleword
    try:
        da = DoublewordAdapter(MODELS["dw_qwen_30b"])
        resp = da.client.chat.completions.create(
            model=MODELS["dw_qwen_30b"].model_id,
            messages=[{"role": "user", "content": "Say Hello"}],
            max_tokens=5
        )
        print(f"Doubleword (Qwen 30B) Success: {resp.choices[0].message.content.strip()}")
    except Exception as e:
        print(f"Doubleword Connectivity Failed: {e}")

def test_parsing():
    print("\n--- Testing Robust Parsing Logic ---")
    oa = OpenAIAdapter(MODELS["gpt-4o-mini"])
    da = DoublewordAdapter(MODELS["dw_qwen_30b"])

    test_cases = [
        ('{"observations": [{"id": "1", "value": "A"}]}', 1, "Direct JSON Object"),
        ('[{"id": "2", "value": "B"}]', 1, "Direct JSON List"),
        ('Some talk before ```json\n{"observations": [{"id": "3", "value": "C"}]}\n``` and after', 1, "Markdown Wrapped"),
        ('```\n[{"id": "4", "value": "D"}]\n```', 1, "Raw Markdown Codeblock"),
        ('Messy text { "observations": [{"id": "5", "value": "E"}] } trail', 1, "Messy Wrapper"),
        ('Invalid JSON string', 0, "Empty/Invalid")
    ]

    for content, expected_len, desc in test_cases:
        # Test both as they share logic
        res_oa = oa._parse_json_content(content)
        res_da = da._parse_json_content(content)

        status = "PASS" if len(res_oa) == expected_len and len(res_da) == expected_len else "FAIL"
        print(f"[{status}] {desc:20} -> Found {len(res_oa)} items (Expected {expected_len})")
        if status == "FAIL":
            print(f"      Content: {content}")
            print(f"      Result: {res_oa}")

if __name__ == "__main__":
    test_connectivity()
    test_parsing()
