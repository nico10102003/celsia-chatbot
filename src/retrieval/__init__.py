"""Recuperación semántica de fragmentos para CelsiaBot."""

from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import DEFAULT_MODEL_NAME, SentenceTransformerEmbedder
from src.retrieval.index import (
    COLLECTION_NAME,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_DIR,
    IndexNotFoundError,
    build_index,
    load_chunks,
    load_index_manifest,
)
from src.retrieval.retriever import DEFAULT_TOP_K, RetrievalResult, Retriever

__all__ = [
    "COLLECTION_NAME",
    "DEFAULT_CHUNKS_PATH",
    "DEFAULT_INDEX_DIR",
    "DEFAULT_MODEL_NAME",
    "DEFAULT_TOP_K",
    "BM25Index",
    "IndexNotFoundError",
    "RetrievalResult",
    "Retriever",
    "SentenceTransformerEmbedder",
    "build_index",
    "load_chunks",
    "load_index_manifest",
]
