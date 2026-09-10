import logging
import os
import threading
from typing import Any

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


# ==============================================================================
# Multi-Key Rotation Pool & Quota Auto-Failover
# ==============================================================================
def _parse_gemini_keys() -> list[str]:
    """Extracts and sanitizes all configured Gemini API keys."""
    raw_keys: list[str] = []

    # Support multiple comma-separated keys: GEMINI_API_KEYS=key1,key2,key3
    multi = os.getenv("GEMINI_API_KEYS", "")
    if multi:
        raw_keys.extend(multi.split(","))

    # Support single key: GEMINI_API_KEY=key1
    single = os.getenv("GEMINI_API_KEY", "")
    if single:
        raw_keys.append(single)

    cleaned_keys: list[str] = []
    for k in raw_keys:
        k = k.strip().strip("'\"")
        if k and k not in cleaned_keys and not k.startswith("your_"):
            cleaned_keys.append(k)

    return cleaned_keys


class GeminiKeyPool:
    """Thread-safe round-robin API key pool with automatic quota exhaustion failover."""

    def __init__(self):
        self.keys = _parse_gemini_keys()
        self.index = 0
        self._lock = threading.Lock()

    def refresh(self) -> None:
        with self._lock:
            self.keys = _parse_gemini_keys()

    def get_current_key(self) -> str:
        with self._lock:
            if not self.keys:
                return os.getenv("GEMINI_API_KEY", "").strip().strip("'\"") or "missing-gemini-key"
            return self.keys[self.index % len(self.keys)]

    def rotate(self) -> str:
        with self._lock:
            if not self.keys:
                return "missing-gemini-key"
            self.index = (self.index + 1) % len(self.keys)
            new_key = self.keys[self.index]
            masked = f"{new_key[:4]}...{new_key[-4:]}" if len(new_key) > 8 else "NONE"
            logger.info("Rotated to Gemini API key #%d of %d (%s)", self.index + 1, len(self.keys), masked)
            return new_key


KEY_POOL = GeminiKeyPool()


def _find_chat_models(obj: Any) -> list[ChatOpenAI]:
    found: list[ChatOpenAI] = []
    visited: set[int] = set()

    def _traverse(o: Any) -> None:
        if o is None or id(o) in visited:
            return
        visited.add(id(o))
        if isinstance(o, ChatOpenAI):
            found.append(o)
        if hasattr(o, "bound"):
            _traverse(o.bound)
        if hasattr(o, "first"):
            _traverse(o.first)
        if hasattr(o, "steps"):
            for s in o.steps:
                _traverse(s)

    _traverse(obj)
    return found


def invoke_with_retry(runnable: Any, *args: Any, **kwargs: Any) -> Any:
    """
    Invokes a LangChain runnable. If an HTTP 429 quota exhaustion occurs and multiple
    Gemini keys are configured in the pool, automatically rotates to the next key
    and retries the request seamlessly.
    """
    total_keys = max(1, len(KEY_POOL.keys))
    for attempt in range(total_keys):
        try:
            return runnable.invoke(*args, **kwargs)
        except Exception as exc:
            err_str = str(exc)
            is_quota_error = "429" in err_str or "quota" in err_str.lower() or "resourceexhausted" in err_str.lower()
            if is_quota_error and len(KEY_POOL.keys) > 1 and attempt < total_keys - 1:
                next_key = KEY_POOL.rotate()
                for model in _find_chat_models(runnable):
                    model.openai_api_key = next_key
                    if hasattr(model, "root_client") and model.root_client is not None:
                        model.root_client.api_key = next_key
                logger.warning(
                    "Hit Gemini quota limit (429). Automatically rotated to key #%d/%d and retrying...",
                    attempt + 2,
                    total_keys,
                )
                continue
            raise


def get_llm() -> ChatOpenAI:
    """
    Returns an initialized ChatOpenAI client supporting:
    1. Google Gemini via OpenAI-compatible endpoint (GEMINI_API_KEY / GEMINI_API_KEYS)
    2. Local Ollama via OLLAMA_BASE_URL (http://localhost:11434/v1)
    3. OpenAI (OPENAI_API_KEY)
    """
    model = os.getenv("LLM_MODEL_NAME", "gemini-3.6-flash")
    # Migrate legacy / deprecated model names to active Google API model
    if model.lower() in {"gemini-2.5-flash", "gemini-1.5-flash"}:
        model = "gemini-3.6-flash"

    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        openai_key = openai_key.strip().strip("'\"")

    KEY_POOL.refresh()
    active_key = KEY_POOL.get_current_key()

    if not active_key and not openai_key and provider != "ollama":
        logger.error(
            "CRITICAL: Neither GEMINI_API_KEY nor OPENAI_API_KEY is set in container environment! "
            "Ensure the .env file exists in the repository root and has GEMINI_API_KEY=AQ... without quotes."
        )

    # If Gemini is selected or Gemini keys are present, route to Google AI Studio
    if active_key or provider == "gemini":
        target_model = model if "gemini" in model.lower() else "gemini-3.6-flash"
        masked = f"{active_key[:4]}...{active_key[-4:]}" if active_key and len(active_key) > 8 else "NONE"
        logger.info(
            "Initializing Google Gemini client (model=%s, key_preview=%s, pool_size=%d)",
            target_model,
            masked,
            len(KEY_POOL.keys),
        )
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
            api_key=active_key or openai_key or "missing-gemini-key",
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
