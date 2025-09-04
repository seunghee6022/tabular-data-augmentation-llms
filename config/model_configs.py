"""
Model configurations for different LLM sizes and QLoRA settings
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    """Configuration for model and QLoRA parameters"""
    model_name: str
    model_size: str
    max_length: int = 2048
    
    # QLoRA specific parameters
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    target_modules: list = None
    
    # Quantization parameters
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_quant_type: str = "nf4"
    
    # Training parameters
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    warmup_steps: int = 100
    max_steps: int = 1000
    learning_rate: float = 2e-4
    fp16: bool = True
    logging_steps: int = 25
    save_steps: int = 500
    eval_steps: int = 500
    
    # Authentication parameters
    requires_auth: bool = False
    is_gated: bool = False
    license_info: str = ""

# Model configurations for different sizes
MODEL_CONFIGS = {
    # === TOP HUGGING FACE LEADERBOARD MODELS (2024) ===
    
    "gemma_3_270m": ModelConfig(
        model_name="google/gemma-3-270m", 
        model_size="270M",
        lora_r=32,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=1500,
        requires_auth=True,
        is_gated=True,
        license_info="Requires Meta license agreement",
    ),
    # 7B Parameter Models - Current Top Performers
    "llama3_1_8b": ModelConfig(
        model_name="meta-llama/Meta-Llama-3.1-8B", 
        model_size="8B",
        lora_r=32,  # Reduced for memory efficiency
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Fewer modules
        per_device_train_batch_size=4,  # Reduced batch size
        gradient_accumulation_steps=8,  # Increased to maintain effective batch size
        max_steps=1500,
        learning_rate=1e-4,  # Slightly lower learning rate
        requires_auth=True,
        is_gated=True,
        license_info="Requires Meta license agreement",
    ),
    
    "qwen3_4b": ModelConfig(
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        model_size="4B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        max_steps=1500,
        learning_rate=1e-4,  # Slightly lower learning rate
    ),
    
    "mistral_nemo_12b": ModelConfig(
        model_name="mistralai/Mistral-Nemo-Base-2407",
        model_size="12B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=6,
        max_steps=1800,
    ),
    
    "gemma2_9b": ModelConfig(
        model_name="google/gemma-2-9b",
        model_size="9B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1600,
    ),
    
    "phi3_medium_14b": ModelConfig(
        model_name="microsoft/Phi-3-medium-4k-instruct",
        model_size="14B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=2000,
    ),
    
    # === ESTABLISHED HIGH-PERFORMING MODELS ===
    
    "llama2_7b": ModelConfig(
        model_name="meta-llama/Llama-2-7b-hf",
        model_size="7B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1500,
        requires_auth=True,
        is_gated=True,
        license_info="Requires Meta license agreement",
    ),
    
    "mistral_7b": ModelConfig(
        model_name="mistralai/Mistral-7B-v0.1",
        model_size="7B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1500,
    ),
    
    "codellama_7b": ModelConfig(
        model_name="codellama/CodeLlama-7b-hf",
        model_size="7B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_steps=1500,
        requires_auth=True,
        is_gated=True,
        license_info="Requires Meta license agreement",
    ),
    
   "qwen3-8B": ModelConfig(
        model_name="Qwen/Qwen3-8B",
        model_size="8B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=2000,
        requires_auth=True,
        is_gated=True,
        license_info="Requires Meta license agreement",
    ),
    
    "vicuna_13b": ModelConfig(
        model_name="lmsys/vicuna-13b-v1.5",
        model_size="13B",
        lora_r=64,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        max_steps=2000,
    )
}

def get_model_config(model_key: str) -> ModelConfig:
    """Get model configuration by key"""
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Model {model_key} not found. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_key]
