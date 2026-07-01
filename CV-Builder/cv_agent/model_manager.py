"""
cv_agent.model_manager
======================
Singleton owning all HuggingFace pipeline handles.

All inference is routed through GPUQueue for thread safety.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from cv_agent.config import PipelineConfig, logger
from cv_agent.gpu_queue import _gpu_queue


# ==============================================================================
# MODEL MANAGER
# ==============================================================================

class ModelManager:
    """
    Singleton owning all HuggingFace pipeline handles.
    All inference is routed through GPUQueue for thread safety.
    """

    _instance: Optional["ModelManager"] = None
    _singleton_lock: threading.Lock      = threading.Lock()

    def __init__(self) -> None:
        self._writer_pipe: Any = None
        self._ats_pipe:    Any = None
        self._hr_pipe:     Any = None
        self._writer_lock  = threading.Lock()
        self._ats_lock     = threading.Lock()
        self._hr_lock      = threading.Lock()

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------

    @staticmethod
    def _device_map(cfg: PipelineConfig) -> str:
        try:
            import torch
            if cfg.device != "auto":
                return cfg.device
            if torch.cuda.is_available():
                return "auto"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    @staticmethod
    def _build_model_kwargs(cfg: PipelineConfig) -> Dict[str, Any]:
        try:
            import torch
            kw: Dict[str, Any] = {
                "device_map":        ModelManager._device_map(cfg),
                "trust_remote_code": True,
            }
            if cfg.hf_token:
                kw["token"] = cfg.hf_token
            if cfg.load_in_4bit and torch.cuda.is_available():
                try:
                    from transformers import BitsAndBytesConfig
                    kw["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16,
                    )
                except ImportError:
                    logger.warning("bitsandbytes not available — 4-bit quantisation disabled")
                    kw["torch_dtype"] = torch.bfloat16
            else:
                kw["torch_dtype"] = torch.bfloat16
            return kw
        except ImportError:
            return {"trust_remote_code": True}

    @staticmethod
    def _make_pipe(
        model_id: str,
        max_tokens: int,
        temperature: float,
        do_sample: bool,
        cfg: PipelineConfig,
    ) -> Any:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
        except ImportError as e:
            raise ImportError("transformers required: pip install transformers>=4.40") from e

        # Some finetuned models have broken tokenizer_class (e.g. "TokenizersBackend").
        # Try AutoTokenizer first, then fall back to explicit class or use_fast=False.
        tok = None
        for attempt_kwargs in [
            {"trust_remote_code": True},
            {"trust_remote_code": True, "use_fast": False},
            {"trust_remote_code": False},
            {"trust_remote_code": False, "use_fast": False},
        ]:
            try:
                tok = AutoTokenizer.from_pretrained(model_id, token=cfg.hf_token, **attempt_kwargs)
                break
            except (ValueError, KeyError, OSError) as e:
                logger.warning("Tokenizer load attempt failed for %s (%s): %s", model_id, attempt_kwargs, e)
                continue

        if tok is None:
            # Last resort: try LlamaTokenizerFast then LlamaTokenizer (sentencepiece)
            for tokenizer_cls_name in ["LlamaTokenizerFast", "LlamaTokenizer"]:
                try:
                    import transformers
                    cls = getattr(transformers, tokenizer_cls_name)
                    tok = cls.from_pretrained(model_id, token=cfg.hf_token)
                    logger.info("Loaded tokenizer via %s fallback for %s", tokenizer_cls_name, model_id)
                    break
                except Exception:
                    continue

        if tok is None:
            raise RuntimeError(
                f"Could not load tokenizer for '{model_id}'. "
                f"The model's tokenizer_config.json may have an invalid tokenizer_class. "
                f"Ensure sentencepiece is installed: pip install sentencepiece"
            )

        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_id, **ModelManager._build_model_kwargs(cfg))
        return hf_pipeline(
            "text-generation",
            model=model,
            tokenizer=tok,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=do_sample,
            repetition_penalty=1.1 if do_sample else 1.0,
            return_full_text=False,
        )

    def writer_pipe(self, cfg: PipelineConfig) -> Any:
        if self._writer_pipe is None:
            with self._writer_lock:
                if self._writer_pipe is None:
                    logger.info("Loading writer model: %s", cfg.writer_model)
                    self._writer_pipe = self._make_pipe(
                        cfg.writer_model, cfg.writer_max_tokens, 0.7, True, cfg
                    )
        return self._writer_pipe

    def ats_judge_pipe(self, cfg: PipelineConfig) -> Any:
        if self._ats_pipe is None:
            with self._ats_lock:
                if self._ats_pipe is None:
                    logger.info("Loading ATS judge — base: %s  adapter: %s", cfg.judge_base, cfg.judge_adapter)
                    try:
                        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline as hf_pipeline
                        # pyrefly: ignore [missing-import]
                        from peft import PeftModel
                    except ImportError as e:
                        raise ImportError("transformers and peft required: pip install transformers peft") from e

                    tok = None
                    for attempt_kwargs in [
                        {"trust_remote_code": True},
                        {"trust_remote_code": True, "use_fast": False},
                        {"trust_remote_code": False},
                        {"trust_remote_code": False, "use_fast": False},
                    ]:
                        try:
                            tok = AutoTokenizer.from_pretrained(cfg.judge_base, token=cfg.hf_token, **attempt_kwargs)
                            break
                        except (ValueError, KeyError, OSError):
                            continue
                    if tok is None:
                        try:
                            from transformers import LlamaTokenizerFast
                            tok = LlamaTokenizerFast.from_pretrained(cfg.judge_base, token=cfg.hf_token)
                        except Exception as e2:
                            raise RuntimeError(f"Could not load tokenizer for '{cfg.judge_base}': {e2}") from e2
                    tok.pad_token = tok.eos_token
                    base  = AutoModelForCausalLM.from_pretrained(cfg.judge_base, **self._build_model_kwargs(cfg))
                    model = PeftModel.from_pretrained(base, cfg.judge_adapter, token=cfg.hf_token)
                    model.eval()
                    self._ats_pipe = hf_pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tok,
                        max_new_tokens=cfg.judge_max_tokens,
                        temperature=0.1,
                        do_sample=False,
                        return_full_text=False,
                    )
        return self._ats_pipe

    def hr_judge_pipe(self, cfg: PipelineConfig) -> Any:
        if self._hr_pipe is None:
            with self._hr_lock:
                if self._hr_pipe is None:
                    logger.info("Loading HR judge model: %s", cfg.hr_judge_model)
                    self._hr_pipe = self._make_pipe(
                        cfg.hr_judge_model, cfg.judge_max_tokens, 0.2, True, cfg
                    )
        return self._hr_pipe

    def release_all(self) -> None:
        with self._writer_lock:
            self._writer_pipe = None
        with self._ats_lock:
            self._ats_pipe = None
        with self._hr_lock:
            self._hr_pipe = None
        logger.info("ModelManager: all pipeline handles released.")


# ==============================================================================
# CHAT HELPER
# ==============================================================================

def chat(pipe: Any, system: str, user: str, temperature: Optional[float] = None) -> str:
    """Send a system + user message pair through a HuggingFace pipeline."""
    tok = pipe.tokenizer
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    prompt = (
        tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if hasattr(tok, "apply_chat_template")
        else f"System: {system}\n\nUser: {user}\n\nAssistant:"
    )
    kwargs: Dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return pipe(prompt, **kwargs)[0]["generated_text"].strip()
