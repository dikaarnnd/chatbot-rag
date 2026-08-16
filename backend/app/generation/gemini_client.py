from __future__ import annotations

import logging
import os
from typing import Iterator

from dotenv import load_dotenv
from google import genai
from google.genai import errors

from app.generation.prompt import SYSTEM_PROMPT

load_dotenv()

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-3.6-flash"
DEFAULT_MAX_OUTPUT_TOKENS = 4096


class GenerationError(Exception):
    """Raised kalau panggilan Gemini API gagal (auth, network, rate limit, dll)."""


def get_client() -> genai.Client:
    """Buat client Gemini. Baca API key dari env var GEMINI_API_KEY.

    Raises:
        GenerationError: kalau GEMINI_API_KEY tidak ter-set.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise GenerationError(
            "GEMINI_API_KEY tidak ditemukan. Set di backend/.env "
            "(lihat .env.example) atau $env:GEMINI_API_KEY di PowerShell."
        )
    return genai.Client(api_key=api_key)


def generate_answer(
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> str:
    """Panggil Gemini API (non-streaming) dan return jawaban lengkap.

    Args:
        user_message: Hasil dari generation/prompt.py build_user_message().
        model: Model Gemini yang dipakai. Default gemini-3.6-flash
        max_output_tokens: Batas token output.
        system_prompt: Instruksi sistem (default dari generation/prompt.py).

    Returns:
        Teks jawaban lengkap dari Gemini.

    Raises:
        GenerationError: kalau API key tidak ada, atau panggilan API gagal
            (network/auth/rate-limit/dll).
    """
    client = get_client()

    try:
        interaction = client.interactions.create(
            model=model,
            system_instruction=system_prompt,
            input=user_message,
            generation_config={"max_output_tokens": max_output_tokens},
        )
    except errors.APIError as exc:
        logger.exception("Gemini API call gagal (model=%s)", model)
        raise GenerationError(f"Gagal memanggil Gemini API: {exc.message}") from exc

    answer = interaction.output_text

    logger.info(
        "Gemini response diterima (model=%s, panjang jawaban=%d karakter)",
        model, len(answer),
    )

    return answer

def generate_answer_stream(
    user_message: str,
    model: str = DEFAULT_MODEL,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    system_prompt: str = SYSTEM_PROMPT,
) -> Iterator[str]:
    """Panggil Gemini API (streaming) dan yield potongan teks jawaban.
 
    Args: sama seperti generate_answer().
 
    Yields:
        Potongan teks jawaban (str) berurutan -- gabungkan untuk dapat jawaban lengkap.
 
    Raises:
        GenerationError: kalau API key tidak ada, atau panggilan API gagal
            (baik saat mulai maupun di tengah streaming).
    """
    client = get_client()
 
    try:
        stream = client.interactions.create(
            model=model,
            system_instruction=system_prompt,
            input=user_message,
            generation_config={"max_output_tokens": max_output_tokens},
            stream=True,
        )
        for event in stream:
            if event.event_type == "step.delta" and event.delta.type == "text":
                yield event.delta.text
    except errors.APIError as exc:
        logger.exception("Gemini API streaming call gagal (model=%s)", model)
        raise GenerationError(f"Gagal memanggil Gemini API: {exc.message}") from exc