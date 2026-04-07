"""
Model loading, quantization, and LoRA configuration.

Handles:
- Loading the base model with 4-bit quantization (QLoRA)
- Applying LoRA adapters
- Loading a trained adapter for inference
"""

import torch
from peft import (
    LoraConfig,
    PeftModel,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.config import (
    ADAPTER_DIR,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_R,
    LORA_TARGET_MODULES,
    MODEL_NAME,
)


def get_bnb_config() -> BitsAndBytesConfig:
    """
    Create 4-bit quantization config (the 'Q' in QLoRA).

    NF4 quantization reduces the model from ~28GB to ~4GB in VRAM.
    Double quantization further compresses the quantization constants.
    """
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_base_model(bnb_config: BitsAndBytesConfig | None = None):
    """
    Load the base model with optional quantization.

    Args:
        bnb_config: Quantization config. If None, creates default 4-bit config.

    Returns:
        Tuple of (model, tokenizer)
    """
    if bnb_config is None:
        bnb_config = get_bnb_config()

    print(f"🧠 Loading base model: {MODEL_NAME}")
    print("   Quantization: 4-bit NF4 with double quantization")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"   ✅ Model loaded on: {model.device}")
    return model, tokenizer


def apply_lora(model) -> None:
    """
    Prepare the model for QLoRA training and attach LoRA adapters.

    Steps:
    1. Freeze base model weights + enable gradient checkpointing
    2. Insert LoRA matrices into target attention layers
    3. Print trainable parameter summary

    Args:
        model: The base model (already quantized).

    Returns:
        Model with LoRA adapters applied.
    """
    print(f"\n🔧 Applying LoRA (r={LORA_R}, alpha={LORA_ALPHA})")
    print(f"   Target modules: {LORA_TARGET_MODULES}")

    # Freeze base weights and enable gradient checkpointing
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    # Print summary
    model.print_trainable_parameters()
    return model


def load_for_inference(adapter_path: str | None = None):
    """
    Load the base model + trained LoRA adapter for inference.

    Args:
        adapter_path: Path to the LoRA adapter directory.
                      Defaults to config.ADAPTER_DIR.

    Returns:
        Tuple of (model, tokenizer) ready for generation.
    """
    if adapter_path is None:
        adapter_path = ADAPTER_DIR

    bnb_config = get_bnb_config()

    print(f"🧠 Loading base model for inference: {MODEL_NAME}")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    print(f"🔌 Loading LoRA adapter from: {adapter_path}")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    print("   ✅ Model ready for inference")
    return model, tokenizer
