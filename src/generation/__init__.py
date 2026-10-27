"""Generación de respuestas con el modelo de lenguaje local de CelsiaBot."""

from src.generation.answer import (
    NO_CONTEXT_ANSWER,
    Answer,
    build_prompt,
    generate_answer,
)
from src.generation.context import build_context
from src.generation.llm import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    create_chat_model,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TEMPERATURE",
    "NO_CONTEXT_ANSWER",
    "Answer",
    "build_context",
    "build_prompt",
    "create_chat_model",
    "generate_answer",
]
