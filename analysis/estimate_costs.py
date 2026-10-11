import json
import argparse
# import numpy as np

def estimate_tokens(text):
    # Rough estimation: 1.3 tokens per word is standard for English
    if not text:
        return 0
    return int(len(str(text).split()) * 1.3)

def load_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data", default="test_gold.jsonl")
    args = parser.parse_args()

    data = load_jsonl(args.test_data)
    num_docs = len(data)

    # --- Constants (Estimates based on prompts) ---
    SCHEMA_TOKENS = 1800 # Ontology schema
    BASE_INSTR_TOKENS = 500 # Instructions
    RAG_EXAMPLE_TOKENS = 300 # Per example (transcript + obs)
    NUM_RAG_EXAMPLES = 5

    ADJUDICATION_INSTR_TOKENS = 800
    REPAIR_INSTR_TOKENS = 400

    # --- Stage 1: Extraction ---
    # Qwen (RAG)
    qwen_input_tokens = 0
    qwen_output_tokens = 0

    # GPT-5 (0-shot)
    gpt5_input_tokens = 0
    gpt5_output_tokens = 0

    # --- Stage 2: Adjudication ---
    adj_input_tokens = 0
    adj_output_tokens = 0

    # --- Stage 3: Repair ---
    # Assuming 35% repair rate, 1.5 passes on average for those
    repair_input_tokens = 0
    repair_output_tokens = 0

    for item in data:
        transcript_len = estimate_tokens(item.get('transcript', ''))

        # Qwen Input: Schema + Instr + (5 * Ex) + Transcript
        q_in = SCHEMA_TOKENS + BASE_INSTR_TOKENS + (NUM_RAG_EXAMPLES * RAG_EXAMPLE_TOKENS) + transcript_len
        qwen_input_tokens += q_in

        # GPT-5 Input: Schema + Instr + Transcript
        g_in = SCHEMA_TOKENS + BASE_INSTR_TOKENS + transcript_len
        gpt5_input_tokens += g_in

        # Outputs (Estimate based on gold length + overhead)
        obs_len = estimate_tokens(json.dumps(item.get('observations', [])))
        qwen_output_tokens += obs_len
        gpt5_output_tokens += obs_len

        # Adjudication Input: Instr + Transcript + Candidate JSON (approx 1.5x gold size for union)
        union_len = int(obs_len * 1.5)
        adj_in = ADJUDICATION_INSTR_TOKENS + transcript_len + union_len
        adj_input_tokens += adj_in
        adj_output_tokens += obs_len # Output is filtered JSON (approx gold size)

    # Repair (Estimated)
    # 35% of docs trigger repair
    num_repair_docs = int(num_docs * 0.35)
    avg_repair_input = 2000 # "Compact prompt" from paper
    avg_repair_output = 50 # Validated value

    repair_input_tokens = num_repair_docs * avg_repair_input
    repair_output_tokens = num_repair_docs * avg_repair_output

    # --- Pricing (User Provided Batch Rates per 1M tokens) ---
    # Qwen (Doubleword): $0.10 / $0.40
    # GPT-5 (OpenAI Batch): $0.625 / $5.00
    PRICE_QWEN_IN = 0.10
    PRICE_QWEN_OUT = 0.40
    PRICE_GPT5_IN = 0.625
    PRICE_GPT5_OUT = 5.00

    cost_qwen_ext = (qwen_input_tokens/1e6 * PRICE_QWEN_IN) + (qwen_output_tokens/1e6 * PRICE_QWEN_OUT)
    cost_gpt5_ext = (gpt5_input_tokens/1e6 * PRICE_GPT5_IN) + (gpt5_output_tokens/1e6 * PRICE_GPT5_OUT)
    cost_adj = (adj_input_tokens/1e6 * PRICE_QWEN_IN) + (adj_output_tokens/1e6 * PRICE_QWEN_OUT)
    cost_repair = (repair_input_tokens/1e6 * PRICE_QWEN_IN) + (repair_output_tokens/1e6 * PRICE_QWEN_OUT)

    total_cost = cost_qwen_ext + cost_gpt5_ext + cost_adj + cost_repair

    # Generate Markdown Table
    print(f"| Stage | Model | Input Tokens | Output Tokens | Est. Cost ($) |")
    print(f"| :--- | :--- | :--- | :--- | :--- |")
    print(f"| **1. Extraction** | Qwen-235B (RAG) | {qwen_input_tokens:,} | {qwen_output_tokens:,} | ${cost_qwen_ext:.2f} |")
    print(f"| | GPT-5 (0-shot) | {gpt5_input_tokens:,} | {gpt5_output_tokens:,} | ${cost_gpt5_ext:.2f} |")
    print(f"| **2. Adjudication** | Qwen-235B | {adj_input_tokens:,} | {adj_output_tokens:,} | ${cost_adj:.2f} |")
    print(f"| **3. Repair** | Qwen-235B (Targeted) | {repair_input_tokens:,} | {repair_output_tokens:,} | ${cost_repair:.2f} |")
    print(f"| **TOTAL** | | **{qwen_input_tokens+gpt5_input_tokens+adj_input_tokens+repair_input_tokens:,}** | **{qwen_output_tokens+gpt5_output_tokens+adj_output_tokens+repair_output_tokens:,}** | **${total_cost:.2f}** |")

if __name__ == "__main__":
    main()
