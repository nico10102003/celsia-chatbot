"""Consulta del índice vectorial y recuperación de fragmentos relevantes."""

from dataclasses import dataclass, field, replace
from pathlib import Path

from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import DEFAULT_MODEL_NAME, SentenceTransformerEmbedder
from src.retrieval.index import (
    COLLECTION_NAME,
    DEFAULT_INDEX_DIR,
    load_index_manifest,
    open_collection,
)

DEFAULT_TOP_K = 5
DEFAULT_STRATEGY = "hybrid"
CANDIDATE_MULTIPLIER = 3
RRF_K = 60
STRATEGIES = ("vector", "hybrid")


@dataclass(frozen=True)
class RetrievalResult:
    """Fragmento recuperado junto con su puntuación y metadatos."""

    id: str
    text: str
    score: float
    distance: float
    metadata: dict = field(default_factory=dict)

    @property
    def url(self):
        """URL de la fuente original, si está disponible."""
        return self.metadata.get("url")

    def as_dict(self):
        """Representación serializable del resultado."""
        return {
            "id": self.id,
            "score": self.score,
            "distance": self.distance,
            "url": self.url,
            "metadata": dict(self.metadata),
            "text": self.text,
        }


class Retriever:
    """Recupera los fragmentos más cercanos a una consulta.

    Args:
        index_dir: Directorio que contiene el índice persistente.
        embedder: Embedder opcional. Si se omite se usa el modelo registrado en
            el manifiesto del índice.
        collection_name: Nombre de la colección de Chroma.
        model_name: Nombre del modelo a usar cuando no se inyecta un embedder.
        device: Dispositivo para SentenceTransformers.
        strategy: Estrategia de recuperación: ``"vector"`` (solo similitud) o
            ``"hybrid"`` (similitud + BM25 con fusión por Reciprocal Rank Fusion).
    """

    def __init__(
        self,
        index_dir=DEFAULT_INDEX_DIR,
        embedder=None,
        collection_name=COLLECTION_NAME,
        model_name=None,
        device=None,
        strategy=DEFAULT_STRATEGY,
    ):
        if strategy not in STRATEGIES:
            raise ValueError(f"La estrategia debe ser una de: {', '.join(STRATEGIES)}.")

        self.index_dir = Path(index_dir)
        self.collection_name = collection_name
        self.strategy = strategy
        self._collection = None

        if embedder is None:
            manifest = load_index_manifest(self.index_dir) or {}
            model_name = (
                model_name or manifest.get("embedding_model") or DEFAULT_MODEL_NAME
            )
            embedder = SentenceTransformerEmbedder(model_name, device=device)

        self.embedder = embedder

    @property
    def collection(self):
        """Abre la colección en el primer uso."""
        if self._collection is None:
            self._collection = open_collection(
                self.index_dir,
                collection_name=self.collection_name,
            )

        return self._collection

    def retrieve(self, query, k=DEFAULT_TOP_K, where=None):
        """Devuelve hasta ``k`` fragmentos ordenados por similitud.

        Con la estrategia ``"vector"`` se devuelven los ``k`` mejores resultados
        de la búsqueda vectorial de Chroma.

        Con la estrategia ``"hybrid"`` (por defecto) se consulta un pool amplio
        de candidatos y se reordena combinando la similitud vectorial con una
        puntuación BM25 mediante Reciprocal Rank Fusion.

        Args:
            query: Texto de la consulta.
            k: Cantidad máxima de resultados.
            where: Filtro de metadatos compatible con Chroma.

        Returns:
            Una lista de ``RetrievalResult`` ordenada de mayor a menor puntuación.
            Con la estrategia ``"vector"`` la puntuación es la similitud (1 - distancia).
            Con la estrategia ``"hybrid"`` la puntuación es la fusión por Reciprocal
            Rank Fusion (RRF) de los candidatos.
        """
        if not isinstance(query, str) or not query.strip():
            raise ValueError("La consulta no puede estar vacía.")

        if k <= 0:
            raise ValueError("k debe ser mayor que cero.")

        collection = self.collection
        total = collection.count()

        if total == 0:
            return []

        pool_size = min(k * CANDIDATE_MULTIPLIER, total) if self.strategy == "hybrid" else min(k, total)

        response = collection.query(
            query_embeddings=[self.embedder.embed_query(query.strip())],
            n_results=pool_size,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        ids = (response.get("ids") or [[]])[0]
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        results = []

        for chunk_id, document, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            distance = float(distance)
            results.append(
                RetrievalResult(
                    id=chunk_id,
                    text=document or "",
                    score=1.0 - distance,
                    distance=distance,
                    metadata=dict(metadata or {}),
                )
            )

        if self.strategy == "hybrid" and len(results) > 1:
            results = _reciprocal_rank_fusion(results, query)

        return results[:k]


def _reciprocal_rank_fusion(results, query, rrf_k=RRF_K):
    """Fusiona candidatos vectoriales y por términos con Reciprocal Rank Fusion."""
    bm25_scores = BM25Index([result.text for result in results]).scores(query)
    ranked_indices = sorted(range(len(results)), key=lambda index: bm25_scores[index], reverse=True)
    bm25_ranks = {index: rank + 1 for rank, index in enumerate(ranked_indices)}

    fused = []
    for index, result in enumerate(results):
        vector_rank = index + 1
        bm25_rank = bm25_ranks.get(index, len(results) + 1)
        fused_score = 1.0 / (rrf_k + vector_rank) + 1.0 / (rrf_k + bm25_rank)
        fused.append((fused_score, bm25_scores[index], result))

    fused.sort(key=lambda pair: (pair[0], pair[1]), reverse=True)
    return [replace(result, score=fused_score) for fused_score, _, result in fused]


def retrieve(query, k=DEFAULT_TOP_K, index_dir=DEFAULT_INDEX_DIR, where=None):
    """Atajo para recuperar fragmentos con la configuración por defecto."""
    return Retriever(index_dir=index_dir).retrieve(query, k=k, where=where)
