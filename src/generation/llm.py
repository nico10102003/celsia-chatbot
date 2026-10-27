"""Conexión con el modelo de lenguaje local usado por CelsiaBot."""

from langchain_ollama import ChatOllama

DEFAULT_MODEL_NAME = "deepseek-r1:1.5b"
DEFAULT_TEMPERATURE = 0.0


def create_chat_model(
    model_name: str = DEFAULT_MODEL_NAME,
    temperature: float = DEFAULT_TEMPERATURE,
    base_url: str | None = None,
) -> ChatOllama:
    """Crea el ``ChatOllama`` que genera las respuestas de CelsiaBot.

    Args:
        model_name: Modelo local servido por Ollama.
        temperature: Temperatura del modelo (0.0 para respuestas más deterministas).
        base_url: URL del servidor Ollama (por defecto ``None``).

    Returns:
        Una instancia lista para invocar con LangChain.
    """
    return ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
    )
