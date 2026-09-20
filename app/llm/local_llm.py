"""Local/open-source text generation, with two interchangeable backends:

- "transformers" (default): runs a small Hugging Face model in-process.
  Works out of the box, no extra installs — good for an MVP on a laptop CPU.
- "ollama": calls a locally running Ollama server (e.g. llama3.2, phi3, mistral).
  Higher quality, but requires Ollama to be installed and running separately.

Select via the LLM_BACKEND env var; see .env.example.
"""
import logging
from functools import lru_cache

import requests

from app.config.settings import settings

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when the configured LLM backend can't produce a response."""


@lru_cache(maxsize=1)
def _load_transformers_model():
    """Load tokenizer + model directly (rather than via the high-level `pipeline()`
    helper), so this doesn't depend on which task names a given transformers version
    happens to register — only on the stable Auto* model/tokenizer classes."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

    model_name = settings.LLM_MODEL
    logger.info("Loading local transformers model '%s'", model_name)
    try:
        config = AutoConfig.from_pretrained(model_name)
        is_encoder_decoder = bool(getattr(config, "is_encoder_decoder", False))
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_cls = AutoModelForSeq2SeqLM if is_encoder_decoder else AutoModelForCausalLM
        model = model_cls.from_pretrained(model_name)
    except Exception as e:
        raise LLMError(f"Could not load local model '{model_name}': {e}") from e
    return tokenizer, model, is_encoder_decoder


def _generate_transformers(prompt: str, max_new_tokens: int) -> str:
    tokenizer, model, is_encoder_decoder = _load_transformers_model()
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    except Exception as e:
        raise LLMError(f"Local model generation failed: {e}") from e

    if is_encoder_decoder:
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
        text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return text.strip()


def _generate_ollama(prompt: str, max_new_tokens: int) -> str:
    try:
        response = requests.post(
            f"{settings.OLLAMA_HOST}/api/generate",
            json={
                "model": settings.LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": max_new_tokens},
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise LLMError(
            f"Could not reach Ollama at {settings.OLLAMA_HOST}. Make sure `ollama serve` "
            f"is running and the model '{settings.LLM_MODEL}' has been pulled "
            f"(`ollama pull {settings.LLM_MODEL}`). Details: {e}"
        ) from e
    return response.json().get("response", "").strip()


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    backend = settings.LLM_BACKEND
    if backend == "ollama":
        return _generate_ollama(prompt, max_new_tokens)
    if backend == "transformers":
        return _generate_transformers(prompt, max_new_tokens)
    raise LLMError(f"Unknown LLM_BACKEND '{backend}'. Use 'transformers' or 'ollama'.")
