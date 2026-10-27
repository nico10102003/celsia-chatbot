"""Embeddings para la recuperación semántica de CelsiaBot.

El modelo se carga de forma perezosa: construir un ``SentenceTransformerEmbedder``
no descarga ni inicializa nada. Esto permite inyectar un embedder falso en las
pruebas sin depender de la red.
"""

from collections.abc import Sequence
from typing import Protocol

DEFAULT_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_BATCH_SIZE = 32


class Embedder(Protocol):
    """Interfaz mínima que necesita el índice vectorial."""

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Genera un vector por cada texto."""

    def embed_query(self, text: str) -> list[float]:
        """Genera el vector de una consulta."""


def _create_sentence_transformer(model_name, device):
    """Importa e inicializa SentenceTransformers solo cuando se necesita."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, device=device)


class SentenceTransformerEmbedder:
    """Embedder basado en un modelo de SentenceTransformers.

    Args:
        model_name: Identificador del modelo en Hugging Face.
        device: Dispositivo de ejecución (por ejemplo ``"cpu"`` o ``"cuda"``).
        batch_size: Tamaño de lote usado al vectorizar documentos.
        model_loader: Fábrica opcional del modelo, útil para pruebas.
    """

    def __init__(
        self,
        model_name=DEFAULT_MODEL_NAME,
        device=None,
        batch_size=DEFAULT_BATCH_SIZE,
        model_loader=None,
    ):
        if not model_name:
            raise ValueError("model_name no puede estar vacío.")

        if batch_size <= 0:
            raise ValueError("batch_size debe ser mayor que cero.")

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self._model_loader = model_loader or _create_sentence_transformer
        self._model = None

    @property
    def model(self):
        """Carga el modelo en el primer uso."""
        if self._model is None:
            self._model = self._model_loader(self.model_name, self.device)

        return self._model

    def embed_documents(self, texts):
        """Vectoriza una secuencia de textos, normalizando cada vector."""
        texts = list(texts)

        if not texts:
            return []

        encoded = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()

        return [[float(value) for value in vector] for vector in encoded]

    def embed_query(self, text):
        """Vectoriza una consulta con el mismo modelo y normalización."""
        if not text or not text.strip():
            raise ValueError("La consulta no puede estar vacía.")

        return self.embed_documents([text])[0]
