import os

from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    """
    Returns an initialized ChatOpenAI client supporting:
    1. Google Gemini via OpenAI-compatible endpoint (GEMINI_API_KEY)
    2. Local Ollama via OLLAMA_BASE_URL (http://localhost:11434/v1)
    3. OpenAI (OPENAI_API_KEY)
    """
    model = os.getenv("LLM_MODEL_NAME", "gemini-2.5-flash")
    temperature = float(os.getenv("LLM_TEMPERATURE", "0.0"))
    provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    gemini_key = os.getenv("GEMINI_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    # If Gemini is selected or GEMINI_API_KEY is present, route to Google AI Studio
    if gemini_key or provider == "gemini":
        target_model = model if "gemini" in model.lower() else "gemini-2.5-flash"
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
