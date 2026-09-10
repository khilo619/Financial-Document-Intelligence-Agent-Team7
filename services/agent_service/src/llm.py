import logging
import os

import langchain_openai.chat_models.base as lc_base
from langchain_openai import ChatOpenAI

logger = logging.getLogger("AgentService.LLM")

# ==============================================================================
# Gemini 3.x Thought Signature Compatibility Patch
# Google Gemini 3.x requires a thought_signature in tool_calls during multi-turn
# reasoning. Standard ChatOpenAI strips extra fields, causing Google to reject
# follow-up turns with "Function call is missing a thought_signature".
# We patch _convert_message_to_dict to inject Google's documented fallback
# sentinel 'skip_thought_signature_validator' into serialized assistant tool calls.
# ==============================================================================
_orig_convert_message_to_dict = lc_base._convert_message_to_dict


def _gemini_compat_convert_message_to_dict(message, *args, **kwargs):
    d = _orig_convert_message_to_dict(message, *args, **kwargs)
    if "tool_calls" in d and d["tool_calls"]:
        for tc in d["tool_calls"]:
            if isinstance(tc, dict) and "extra_content" not in tc:
                tc["extra_content"] = {"google": {"thought_signature": "skip_thought_signature_validator"}}
    return d


lc_base._convert_message_to_dict = _gemini_compat_convert_message_to_dict


def get_llm() -> ChatOpenAI:
    """
    Returns an initialized ChatOpenAI client supporting:
    1. Google Gemini via OpenAI-compatible endpoint (GEMINI_API_KEY)
    2. Local Ollama via OLLAMA_BASE_URL (http://localhost:11434/v1)
    3. OpenAI (OPENAI_API_KEY)
    """
    model = os.getenv("LLM_MODEL_NAME", "gemini-3.6-flash")
    # Migrate legacy / deprecated model names to active Google API model
    if model.lower() in {"gemini-2.5-flash", "gemini-1.5-flash"}:
        model = "gemini-3.6-flash"

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # Sanitize keys: strip whitespace, newlines, and accidental wrapping quotes
    if gemini_key:
        gemini_key = gemini_key.strip().strip("'\"")
    if openai_key:
        openai_key = openai_key.strip().strip("'\"")

    if not gemini_key and not openai_key and provider != "ollama":
        logger.error(
            "CRITICAL: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in container environment! "
            "Ensure the .env file exists in the repository root and has GEMINI_API_KEY=AQ... without quotes."
        )

    # If Gemini is selected or GEMINI_API_KEY is present, route to Google AI Studio
    if gemini_key or provider == "gemini":
        target_model = model if "gemini" in model.lower() else "gemini-3.6-flash"
        masked = f"{gemini_key[:4]}...{gemini_key[-4:]}" if gemini_key and len(gemini_key) > 8 else "NONE"
        logger.info("Initializing Google Gemini client (model=%s, key_preview=%s)", target_model, masked)
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
            api_key=gemini_key or openai_key or "missing-gemini-key",
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        )
    elif provider == "ollama":
        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key="ollama",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        )

    # Standard OpenAI
    return ChatOpenAI(
        model=model,
        temperature=temperature,
        api_key=openai_key or "missing-openai-key",
    )
