"""Generación de respuestas a partir del contexto recuperado."""

from collections.abc import Sequence
from dataclasses import dataclass, field

from src.generation.context import build_context
from src.generation.llm import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    create_chat_model,
)
from src.retrieval.retriever import RetrievalResult

NO_CONTEXT_ANSWER = (
    "No encontré información suficiente en las fuentes oficiales de Celsia para "
    "responder esta pregunta. No tengo datos verificados sobre el tema."
)


@dataclass
class Answer:
    """Respuesta generada junto con el contexto usado y las fuentes."""

    text: str
    sources: list[RetrievalResult] = field(default_factory=list)
    context: str = ""


def build_prompt(question: str, context: str) -> str:
    """Construye el prompt que se entrega al modelo junto con el contexto."""
    return (
        "Eres CelsiaBot, el asistente virtual de información corporativa de Celsia. "
        "Responde en español, de forma clara y útil.\n\n"
        f"Pregunta del usuario: {question}\n\n"
        f"Contexto recuperado:\n{context}\n\n"
        "Instrucciones:\n"
        "- Responde únicamente con la información del contexto.\n"
        "- Si el contexto no contiene la información suficiente, indica explícitamente "
        "que no la tienes y no inventes datos.\n"
        "- No respondas con afirmaciones que no estén respaldadas por el contexto.\n"
        "- Incluye la URL de la fuente cuando uses información del contexto.\n"
        "- No uses información que no esté en el contexto."
    )


def _response_text(response) -> str:
    """Extrae el texto de contenido de la respuesta del modelo."""
    content = response.content if hasattr(response, "content") else response

    if isinstance(content, str):
        return content.strip()

    return str(content).strip()


def _format_sources(sources: Sequence[RetrievalResult]) -> str:
    """Formatea la sección de fuentes con las URLs utilizadas."""
    lines = ["Fuentes utilizadas:"]

    for source in sources:
        if source.url:
            lines.append(f"- {source.url}")

    if len(lines) == 1:
        lines.append("- (sin URL disponible)")

    return "\n".join(lines)


def _assemble_answer(generated_text: str, sources: Sequence[RetrievalResult]) -> str:
    """Une la respuesta generada con la sección de fuentes."""
    answer_text = generated_text.strip()
    sources_text = _format_sources(sources)

    if not answer_text:
        return sources_text

    return f"{answer_text}\n\n{sources_text}"


def generate_answer(
    question: str,
    results: Sequence[RetrievalResult],
    llm=None,
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Answer:
    """Genera una respuesta fundamentada en los fragmentos recuperados.

    Args:
        question: Pregunta del usuario.
        results: Fragmentos recuperados ordenados por similitud.
        llm: Chat model de LangChain a usar; por defecto se crea el de Ollama.
        model_name: Modelo local usado cuando no se inyecta un ``llm``.
        temperature: Temperatura del modelo cuando no se inyecta un ``llm``.

    Returns:
        La respuesta generada junto con el contexto usado y las fuentes.
    """
    context = build_context(results)

    if not results:
        return Answer(text=NO_CONTEXT_ANSWER, sources=[], context=context)

    chat_model = llm or create_chat_model(model_name=model_name, temperature=temperature)
    response = chat_model.invoke(build_prompt(question, context))

    sources = list(results)

    return Answer(
        text=_assemble_answer(_response_text(response), sources),
        sources=sources,
        context=context,
    )
