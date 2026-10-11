from dataclasses import dataclass, field
from typing import List, Optional, Dict

@dataclass
class VariantSpec:
    name: str
    description: str
    models: List[str]
    fs_mode: str = "0-shot"
    use_rag: bool = False
    ensemble_mode: Optional[str] = None  # None, "union", "adjudicate"
    adjudicator_model: Optional[str] = None
    use_repair: bool = False
    exp_suffix: str = ""

VARIANTS = {
    # V1: Core Single-Model Baselines (0-shot)
    "v1_gpt5_nano": VariantSpec(
        name="v1_gpt5_nano",
        description="GPT-5 Nano baseline (0-shot)",
        models=["gpt-5-nano"],
        fs_mode="0-shot"
    ),
    "v1_dw_qwen_235b": VariantSpec(
        name="v1_dw_qwen_235b",
        description="Doubleword Qwen 235B baseline (0-shot)",
        models=["dw_qwen_235b"],
        fs_mode="0-shot"
    ),
    "v1_gpt4o": VariantSpec(
        name="v1_gpt4o",
        description="GPT-4o baseline (0-shot)",
        models=["gpt-4o"],
        fs_mode="0-shot"
    ),

    # V2: Few-Shot RAG Impact
    "v2_gpt5_nano_rag": VariantSpec(
        name="v2_gpt5_nano_rag",
        description="GPT-5 Nano with 5-shot RAG",
        models=["gpt-5-nano"],
        fs_mode="5-shot-dynamic",
        use_rag=True
    ),
    "v2_dw_qwen_235b_rag": VariantSpec(
        name="v2_dw_qwen_235b_rag",
        description="Doubleword Qwen 235B with 5-shot RAG",
        models=["dw_qwen_235b"],
        fs_mode="5-shot-dynamic",
        use_rag=True
    ),

    # V3: Union-Only Ensemble
    "v3_union": VariantSpec(
        name="v3_union",
        description="Union of GPT-5-Nano, Qwen 235B, and GPT-4o (0-shot)",
        models=["gpt-5-nano", "dw_qwen_235b", "gpt-4o"],
        fs_mode="0-shot",
        ensemble_mode="union"
    ),

    # V4: Adjudication Impact
    "v4_adjudication": VariantSpec(
        name="v4_adjudication",
        description="Adjudicated Ensemble (Judge: GPT-4o)",
        models=["gpt-5-nano", "dw_qwen_235b", "gpt-4o"],
        fs_mode="0-shot",
        ensemble_mode="adjudicate",
        adjudicator_model="gpt-4o"
    ),

    # V5: Targeted Repair Impact
    "v5_repair": VariantSpec(
        name="v5_repair",
        description="Adjudicated Ensemble + Targeted Repair",
        models=["gpt-5-nano", "dw_qwen_235b", "gpt-4o"],
        fs_mode="0-shot",
        ensemble_mode="adjudicate",
        adjudicator_model="gpt-4o",
        use_repair=True
    ),

    # V6: ACE Full (The Paper configuration)
    "v6_ace_full": VariantSpec(
        name="v6_ace_full",
        description="ACE Full System (Optimized)",
        models=["gpt-5-nano", "dw_qwen_235b"], # Based on Agents.md findings
        fs_mode="0-shot",
        ensemble_mode="adjudicate",
        adjudicator_model="gpt-5-nano", # Or whatever was best
        use_repair=True,
        exp_suffix="ACE_v1"
    )
}
