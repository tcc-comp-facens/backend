"""
Cliente LLM centralizado com rate limiting e retry.

Suporta Groq (prioridade) e Google Gemini (fallback).
Serializa chamadas para evitar estouro de cota,
com retry automático em caso de 429 (RESOURCE_EXHAUSTED / rate limit).
"""

import logging
import os
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Lock global — garante que só uma chamada ao LLM acontece por vez
_lock = threading.Lock()

# Timestamp da última chamada
_last_call_time = 0.0
_MIN_INTERVAL = 2.0  # Groq free tier é mais generoso: 30 RPM

MAX_RETRIES = 3
RETRY_BASE_DELAY = 10.0  # segundos


def _get_provider() -> str:
    """Detecta qual provider usar baseado nas variáveis de ambiente."""
    if os.environ.get("GROQ_API_KEY", "").strip():
        return "groq"
    if os.environ.get("GEMINI_API_KEY", "").strip():
        return "gemini"
    return "none"


def _generate_groq(prompt: str, model: str) -> Optional[str]:
    """Chama a API do Groq."""
    from groq import Groq

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=4096,
    )
    return response.choices[0].message.content or ""


def _generate_gemini(prompt: str, model: str) -> Optional[str]:
    """Chama a API do Google Gemini."""
    from google import genai

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
    )
    return response.text or ""


def generate(prompt: str, model: Optional[str] = None) -> Optional[str]:
    """Chama o LLM com rate limiting e retry.

    Detecta automaticamente o provider (Groq ou Gemini) baseado
    nas variáveis de ambiente.

    Returns:
        Texto gerado, ou None se falhar após retries.
    """
    global _last_call_time

    provider = _get_provider()
    if provider == "none":
        logger.warning("LLM: nenhuma API key configurada (GROQ_API_KEY ou GEMINI_API_KEY)")
        return None

    # Modelo padrão por provider
    if model is None:
        if provider == "groq":
            model = "llama-3.3-70b-versatile"
        else:
            model = "gemini-2.0-flash"

    with _lock:
        for attempt in range(MAX_RETRIES):
            # Rate limit: espera intervalo mínimo entre chamadas
            elapsed = time.time() - _last_call_time
            if elapsed < _MIN_INTERVAL:
                wait = _MIN_INTERVAL - elapsed
                logger.info("LLM rate limit: aguardando %.1fs", wait)
                time.sleep(wait)

            try:
                _last_call_time = time.time()

                if provider == "groq":
                    text = _generate_groq(prompt, model)
                else:
                    text = _generate_gemini(prompt, model)

                if text:
                    logger.info("LLM (%s): resposta gerada com sucesso", provider)
                    return text

                logger.warning("LLM (%s): resposta vazia", provider)
                return None

            except Exception as exc:
                exc_str = str(exc)
                if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "rate_limit" in exc_str.lower():
                    delay = RETRY_BASE_DELAY * (attempt + 1)
                    logger.warning(
                        "LLM (%s): cota excedida (tentativa %d/%d), aguardando %.0fs",
                        provider, attempt + 1, MAX_RETRIES, delay,
                    )
                    time.sleep(delay)
                else:
                    logger.error("LLM (%s): erro inesperado — %s", provider, exc)
                    return None

    logger.error("LLM (%s): falhou apos %d tentativas", provider, MAX_RETRIES)
    return None
