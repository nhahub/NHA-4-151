"""
Model Loader — Load Qwen2 base model + merge LoRA adapter.

Uses singleton pattern to avoid reloading on every request.
Supports 4-bit quantization via bitsandbytes for GPU efficiency.
Falls back to CPU float32 if no GPU is available.
"""

import os
import logging
import torch
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ── Configuration ──────────────────────────────────────────
BASE_MODEL = os.getenv("BASE_MODEL", "OsamaHayba/qwen-ats-merged-stage1")
ADAPTER_PATH = os.getenv("ADAPTER_PATH", "OsamaHayba/cv-analysis-final-stage2")
HF_TOKEN = os.getenv("HF_TOKEN", "")
MODEL_DEVICE = os.getenv("MODEL_DEVICE", "auto")

# ── Singleton State ────────────────────────────────────────
_model = None
_tokenizer = None
_device = None
_is_loaded = False


def _detect_device() -> str:
    """Detect the best available device."""
    if MODEL_DEVICE != "auto":
        return MODEL_DEVICE
    if torch.cuda.is_available():
        logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
        return "cuda"
    logger.warning("No CUDA GPU detected. Using CPU (inference will be slow).")
    return "cpu"


def load_model():
    """
    Load the base Qwen2 model + LoRA adapter.

    - On GPU: 4-bit quantization via bitsandbytes
    - On CPU: float32 (no quantization)
    - Merges the adapter into the base model for faster inference.

    Returns:
        tuple: (model, tokenizer, device_str)
    """
    global _model, _tokenizer, _device, _is_loaded

    if _is_loaded:
        logger.info("Model already loaded, returning cached instance.")
        return _model, _tokenizer, _device

    from transformers import AutoModelForCausalLM, AutoTokenizer

    _device = _detect_device()
    logger.info(f"Loading base model: {BASE_MODEL}")
    logger.info(f"Loading adapter: {ADAPTER_PATH}")
    logger.info(f"Target device: {_device}")

    # ── Tokenizer ──────────────────────────────────────────
    _tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        token=HF_TOKEN,
        trust_remote_code=True,
    )

    # Ensure pad token is set
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # ── Model Loading ──────────────────────────────────────
    if _device == "cuda":
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            _model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                quantization_config=bnb_config,
                device_map="auto",
                token=HF_TOKEN,
                trust_remote_code=True,
            )
            logger.info("Base model loaded with 4-bit quantization on GPU.")

        except Exception as e:
            logger.warning(f"4-bit quantization failed: {e}. Falling back to float16.")
            _model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                token=HF_TOKEN,
                trust_remote_code=True,
            )
    else:
        # CPU — load in float32
        _model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            token=HF_TOKEN,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        logger.info("Base model loaded in float32 on CPU.")

    # ── Load & Merge LoRA Adapter ──────────────────────────
    try:
        from peft import PeftModel

        logger.info("Loading LoRA adapter...")
        _model = PeftModel.from_pretrained(
            _model,
            ADAPTER_PATH,
            token=HF_TOKEN,
        )

        logger.info("Merging adapter into base model...")
        _model = _model.merge_and_unload()
        logger.info("LoRA adapter merged successfully.")

    except Exception as e:
        logger.error(f"Failed to load/merge LoRA adapter: {e}")
        logger.warning("Proceeding with base model only.")

    _model.eval()
    _is_loaded = True
    logger.info("Model ready for inference.")

    return _model, _tokenizer, _device


def get_model():
    """Get the loaded model, tokenizer, and device. Loads if not already loaded."""
    return load_model()


def is_loaded() -> bool:
    """Check if the model is loaded."""
    return _is_loaded


def get_model_info() -> dict:
    """Return model metadata."""
    return {
        "base_model": BASE_MODEL,
        "adapter": ADAPTER_PATH,
        "device": _device or "not loaded",
        "is_loaded": _is_loaded,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
    }
