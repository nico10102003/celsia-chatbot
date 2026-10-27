"""Construcción del contexto a partir de los fragmentos recuperados."""

from collections.abc import Sequence

from src.retrieval.retriever import RetrievalResult


def _format_fragment(position: int, result: RetrievalResult) -> str:
    """Formatea un fragmento con su posición, metadatos y texto."""
    lines = [f"Fragmento {position}:"]

    if result.url:
        lines.append(f"URL: {result.url}")

    title = result.metadata.get("title")
    if title:
        lines.append(f"Título: {title}")

    category = result.metadata.get("category")
    if category:
        lines.append(f"Categoría: {category}")

    lines.append(f"Texto: {result.text}")
    return "\n".join(lines)


def build_context(results: Sequence[RetrievalResult]) -> str:
    """Serializa los fragmentos recuperados en un contexto de texto.

    Args:
        results: Fragmentos ordenados por similitud.

    Returns:
        Texto con los fragmentos encontrados, o una cadena vacía si no hay resultados.
    """
    if not results:
        return ""

    fragments = [
        _format_fragment(position, result)
        for position, result in enumerate(results, start=1)
    ]

    return "\n\n".join(fragments)
