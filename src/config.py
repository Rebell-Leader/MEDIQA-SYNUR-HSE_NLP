import os
from dataclasses import dataclass
from enum import Enum
from dotenv import load_dotenv

# Load env vars from project root
load_dotenv()

class ExperimentStage(Enum):
    CONNECTION_TEST = "connection_test"
    ZERO_SHOT_VALIDATION = "0_shot_validation"
    ZERO_SHOT_TEST = "0_shot_test"
    FEW_SHOT_5_FIXED = "5_shot_fixed"
    FEW_SHOT_5_DYNAMIC = "5_shot_dynamic_rag"

@dataclass
class ModelConfig:
    provider: str
    model_id: str
    base_url: str = None
    api_key_env: str = None
    supports_structured_output: bool = False
    supports_batch_api: bool = False

# Central Registry
MODELS = {
    # OpenAI Models
    "gpt-4o": ModelConfig(
        provider="openai",
        model_id="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        supports_batch_api=True
    ),
    "gpt-4o-mini": ModelConfig(
        provider="openai",
        model_id="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        supports_batch_api=True
    ),
    "gpt-5-nano": ModelConfig(
        provider="openai",
        model_id="gpt-5-nano-2025-08-07",
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        supports_batch_api=True
    ),
    "gpt-5-mini": ModelConfig(
        provider="openai",
        model_id="gpt-5-mini-2025-08-07",
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        supports_batch_api=True
    ),
    "gpt-5": ModelConfig(
        provider="openai",
        model_id="gpt-5-2025-08-07",
        api_key_env="OPENAI_API_KEY",
        supports_structured_output=True,
        supports_batch_api=True
    ),

    # Doubleword Models (Qwen)
    "dw_qwen_30b": ModelConfig(
        provider="doubleword",
        model_id="Qwen/Qwen3-VL-30B-A3B-Instruct-FP8",
        base_url="https://api.doubleword.ai/v1",
        api_key_env="DOUBLEWORD_API_KEY",
        supports_structured_output=True, # Supports json_object
        supports_batch_api=True
    ),
    "dw_qwen_235b": ModelConfig(
        provider="doubleword",
        model_id="Qwen/Qwen3-VL-235B-A22B-Instruct-FP8",
        base_url="https://api.doubleword.ai/v1",
        api_key_env="DOUBLEWORD_API_KEY",
        supports_structured_output=True,
        supports_batch_api=True
    ),
    "dw_qwen_embedding": ModelConfig(
        provider="doubleword",
        model_id="Qwen/Qwen3-Embedding-8B",
        base_url="https://api.doubleword.ai/v1",
        api_key_env="DOUBLEWORD_API_KEY",
        supports_structured_output=False,
        supports_batch_api=True
    ),

    # HuggingFace Models (Novita/Together via Router)
    "hf_kimi": ModelConfig(
        provider="hf",
        model_id="moonshotai/Kimi-K2.5:together",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_API_KEY",
        supports_structured_output=False,
        supports_batch_api=False
    ),
    "hf_glm_flash": ModelConfig(
        provider="hf",
        model_id="zai-org/GLM-4.7-Flash:novita",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_API_KEY",
        supports_structured_output=False,
        supports_batch_api=False
    ),
    "hf_glm": ModelConfig(
        provider="hf",
        model_id="zai-org/GLM-4.7:novita",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_API_KEY",
        supports_structured_output=False,
        supports_batch_api=False
    ),
    "hf_deepseek": ModelConfig(
        provider="hf",
        model_id="deepseek-ai/DeepSeek-V3.2:novita",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_API_KEY",
        supports_structured_output=False,
        supports_batch_api=False
    ),
    "hf_minimax": ModelConfig(
        provider="hf",
        model_id="MiniMaxAI/MiniMax-M2.1:novita",
        base_url="https://router.huggingface.co/v1",
        api_key_env="HF_API_KEY",
        supports_structured_output=False,
        supports_batch_api=False
    )
}

def get_api_key(env_var: str) -> str:
    key = os.getenv(env_var)
    if not key:
        raise ValueError(f"Environment variable {env_var} not set")
    return key
