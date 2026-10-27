"""Pruebas de la recuperación semántica sin descargar el modelo real."""

import json
from typing import ClassVar

import pytest

from src.retrieval.bm25 import BM25Index
from src.retrieval.embeddings import SentenceTransformerEmbedder
from src.retrieval.index import (
    MANIFEST_FILENAME,
    IndexNotFoundError,
    build_index,
    load_chunks,
    load_index_manifest,
    open_collection,
)
from src.retrieval.retriever import RRF_K, Retriever

CHUNKS = [
    {
        "id": "source_01-chunk-000",
        "source_id": "source_01",
        "title": "Granjas solares",
        "url": "https://www.celsia.com/es/granjas-solares/",
        "category": "energia_solar",
        "extracted_at": "2026-01-01T00:00:00+00:00",
        "chunk_index": 0,
        "characters": 54,
        "text": "Celsia construye granjas solares en el Valle del Cauca.",
    },
    {
        "id": "source_02-chunk-000",
        "source_id": "source_02",
        "title": "Oportunidades laborales",
        "url": "https://www.celsia.com/es/se-parte-de-nuestro-equipo/",
        "category": "empleo",
        "extracted_at": "2026-01-01T00:00:00+00:00",
        "chunk_index": 0,
        "characters": 52,
        "text": "Consulta las vacantes de empleo y trabajo en Celsia.",
    },
    {
        "id": "source_03-chunk-000",
        "source_id": "source_03",
        "title": "Canales de contacto",
        "url": "https://www.celsia.com/es/encuentranos/",
        "category": "atencion",
        "extracted_at": "2026-01-01T00:00:00+00:00",
        "chunk_index": 0,
        "characters": 50,
        "text": "Estos son los canales de contacto y atención al cliente.",
    },
]


class KeywordEmbedder:
    """Embedder determinista para pruebas; nunca toca la red."""

    model_name = "fake-keyword-embedder"

    def __init__(self):
        self.document_calls = []
        self.query_calls = []

    @staticmethod
    def _vector(text):
        text = text.lower()

        if "solar" in text:
            return [1.0, 0.0, 0.0, 0.0]

        if "empleo" in text or "trabajo" in text:
            return [0.0, 1.0, 0.0, 0.0]

        if "contacto" in text or "atención" in text:
            return [0.0, 0.0, 1.0, 0.0]

        return [0.0, 0.0, 0.0, 1.0]

    def embed_documents(self, texts):
        self.document_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    def embed_query(self, text):
        self.query_calls.append(text)
        return self._vector(text)


class ArbitraryEmbedder:
    """Embedder con ranking vectorial degenerado para forzar el uso de BM25."""

    model_name = "fake-arbitrary-embedder"

    _VECTORS: ClassVar[list[list[float]]] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ]

    def embed_documents(self, texts):
        return [self._VECTORS[index % len(self._VECTORS)] for index, _ in enumerate(texts)]

    def embed_query(self, text):
        return self._VECTORS[0]


def write_chunks(path, chunks=CHUNKS):
    """Escribe un archivo JSONL de chunks para las pruebas."""
    path.write_text(
        "".join(json.dumps(chunk, ensure_ascii=False) + "\n" for chunk in chunks),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def chunks_path(tmp_path):
    return write_chunks(tmp_path / "chunks.jsonl")


@pytest.fixture
def built_index(tmp_path, chunks_path):
    index_dir = tmp_path / "index"
    embedder = KeywordEmbedder()
    manifest = build_index(
        chunks_path=chunks_path,
        index_dir=index_dir,
        embedder=embedder,
        model_name=embedder.model_name,
        batch_size=2,
    )
    return index_dir, manifest, embedder


# --- Carga y validación de chunks -----------------------------------------


def test_load_chunks_reads_jsonl_and_skips_blank_lines(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        "\n".join(json.dumps(chunk, ensure_ascii=False) for chunk in CHUNKS[:2])
        + "\n\n",
        encoding="utf-8",
    )

    chunks = load_chunks(path)

    assert [chunk["id"] for chunk in chunks] == [
        "source_01-chunk-000",
        "source_02-chunk-000",
    ]


def test_load_chunks_rejects_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_chunks(tmp_path / "no-existe.jsonl")


def test_load_chunks_rejects_invalid_json(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text('{"id": "a", "text": "ok"}\n{roto\n', encoding="utf-8")

    with pytest.raises(ValueError, match="línea 2"):
        load_chunks(path)


@pytest.mark.parametrize(
    "chunk",
    [
        {"id": "a"},
        {"text": "sin id"},
        {"id": "", "text": "id vacío"},
        {"id": "a", "text": "   "},
    ],
)
def test_load_chunks_rejects_incomplete_chunks(tmp_path, chunk):
    path = tmp_path / "chunks.jsonl"
    path.write_text(json.dumps(chunk) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_chunks(path)


def test_load_chunks_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "chunks.jsonl"
    path.write_text(
        json.dumps(CHUNKS[0]) + "\n" + json.dumps(CHUNKS[0]) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicado"):
        load_chunks(path)


# --- Construcción del índice ----------------------------------------------


def test_build_index_creates_persistent_collection(built_index):
    index_dir, manifest, _ = built_index

    assert manifest["total_chunks"] == len(CHUNKS)
    assert manifest["total_sources"] == 3
    assert manifest["embedding_dimension"] == 4
    assert manifest["embedding_model"] == "fake-keyword-embedder"
    assert len(manifest["chunks_sha256"]) == 64
    assert (index_dir / MANIFEST_FILENAME).is_file()
    assert (index_dir / "chroma").is_dir()

    saved = load_index_manifest(index_dir)
    assert saved == manifest

    collection = open_collection(index_dir)
    assert collection.count() == len(CHUNKS)
    stored = collection.get(ids=["source_01-chunk-000"], include=["metadatas"])
    assert stored["metadatas"][0]["url"] == CHUNKS[0]["url"]
    assert stored["metadatas"][0]["category"] == "energia_solar"


def test_build_index_replaces_existing_collection(tmp_path, chunks_path):
    index_dir = tmp_path / "index"
    embedder = KeywordEmbedder()

    build_index(
        chunks_path=chunks_path,
        index_dir=index_dir,
        embedder=embedder,
        model_name=embedder.model_name,
    )
    reduced_path = write_chunks(tmp_path / "reduced.jsonl", CHUNKS[:1])
    manifest = build_index(
        chunks_path=reduced_path,
        index_dir=index_dir,
        embedder=embedder,
        model_name=embedder.model_name,
    )

    assert manifest["total_chunks"] == 1
    assert open_collection(index_dir).count() == 1


def test_build_index_with_empty_chunks(tmp_path):
    chunks_path = write_chunks(tmp_path / "chunks.jsonl", [])
    index_dir = tmp_path / "index"
    embedder = KeywordEmbedder()

    manifest = build_index(
        chunks_path=chunks_path,
        index_dir=index_dir,
        embedder=embedder,
        model_name=embedder.model_name,
    )

    assert manifest["total_chunks"] == 0
    assert manifest["embedding_dimension"] == 0
    assert open_collection(index_dir).count() == 0


def test_build_index_rejects_invalid_batch_size(chunks_path, tmp_path):
    with pytest.raises(ValueError):
        build_index(
            chunks_path=chunks_path,
            index_dir=tmp_path / "index",
            embedder=KeywordEmbedder(),
            batch_size=0,
        )


# --- Recuperación ----------------------------------------------------------


def test_retrieve_returns_most_relevant_chunk_with_metadata(built_index):
    index_dir, _, embedder = built_index
    retriever = Retriever(index_dir=index_dir, embedder=embedder, strategy="vector")

    results = retriever.retrieve("energía solar", k=2)

    assert results
    assert results[0].id == "source_01-chunk-000"
    assert results[0].url == CHUNKS[0]["url"]
    assert results[0].metadata["title"] == "Granjas solares"
    assert results[0].metadata["category"] == "energia_solar"
    assert results[0].score == pytest.approx(1.0 - results[0].distance)
    assert results[0].score > 0.99
    assert len(results) <= 2

    serialized = results[0].as_dict()
    assert serialized["id"] == results[0].id
    assert serialized["url"] == CHUNKS[0]["url"]
    assert serialized["text"] == results[0].text


def test_retrieve_honors_k(built_index):
    index_dir, _, embedder = built_index
    retriever = Retriever(index_dir=index_dir, embedder=embedder)

    assert len(retriever.retrieve("solar", k=1)) == 1


def test_retrieve_handles_k_larger_than_collection(built_index):
    index_dir, _, embedder = built_index
    retriever = Retriever(index_dir=index_dir, embedder=embedder)

    results = retriever.retrieve("solar", k=100)

    assert len(results) == len(CHUNKS)


def test_retrieve_filters_by_metadata(built_index):
    index_dir, _, embedder = built_index
    retriever = Retriever(index_dir=index_dir, embedder=embedder)

    results = retriever.retrieve("solar", k=5, where={"category": "empleo"})

    assert [result.id for result in results] == ["source_02-chunk-000"]
    assert results[0].url == CHUNKS[1]["url"]


def test_retrieve_from_empty_index_does_not_call_embedder(tmp_path):
    chunks_path = write_chunks(tmp_path / "chunks.jsonl", [])
    index_dir = tmp_path / "index"
    embedder = KeywordEmbedder()
    build_index(
        chunks_path=chunks_path,
        index_dir=index_dir,
        embedder=embedder,
        model_name=embedder.model_name,
    )
    retriever = Retriever(index_dir=index_dir, embedder=embedder)

    assert retriever.retrieve("cualquier consulta") == []
    assert embedder.query_calls == []


def test_retrieve_without_index_raises(tmp_path):
    retriever = Retriever(
        index_dir=tmp_path / "no-existe",
        embedder=KeywordEmbedder(),
    )

    with pytest.raises(IndexNotFoundError):
        retriever.retrieve("solar")


@pytest.mark.parametrize("query", ["", "   ", None])
def test_retrieve_rejects_empty_query(built_index, query):
    index_dir, _, embedder = built_index
    retriever = Retriever(index_dir=index_dir, embedder=embedder)

    with pytest.raises(ValueError):
        retriever.retrieve(query)


@pytest.mark.parametrize("k", [0, -1])
def test_retrieve_rejects_invalid_k(built_index, k):
    index_dir, _, embedder = built_index
    retriever = Retriever(index_dir=index_dir, embedder=embedder)

    with pytest.raises(ValueError):
        retriever.retrieve("solar", k=k)


def test_retriever_rejects_unknown_strategy(built_index):
    index_dir, _, embedder = built_index

    with pytest.raises(ValueError):
        Retriever(index_dir=index_dir, embedder=embedder, strategy="planeta")


# --- Embedder de SentenceTransformers -------------------------------------


class FakeSentenceTransformer:
    def __init__(self):
        self.calls = []

    def encode(self, texts, **kwargs):
        self.calls.append((list(texts), kwargs))
        return [[1, 0, 0] for _ in texts]


def test_sentence_transformer_embedder_loads_model_lazily():
    loaded = []
    model = FakeSentenceTransformer()

    def loader(model_name, device):
        loaded.append((model_name, device))
        return model

    embedder = SentenceTransformerEmbedder(
        "modelo-de-prueba",
        device="cpu",
        batch_size=4,
        model_loader=loader,
    )

    assert loaded == []

    vectors = embedder.embed_documents(["hola", "mundo"])

    assert loaded == [("modelo-de-prueba", "cpu")]
    assert vectors == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert model.calls[0][1]["normalize_embeddings"] is True
    assert model.calls[0][1]["batch_size"] == 4

    embedder.embed_query("consulta")
    assert len(model.calls) == 2


def test_sentence_transformer_embedder_handles_empty_input():
    loaded = []
    embedder = SentenceTransformerEmbedder(
        "modelo-de-prueba",
        model_loader=lambda name, device: (
            loaded.append(name) or FakeSentenceTransformer()
        ),
    )

    assert embedder.embed_documents([]) == []
    assert loaded == []


def test_sentence_transformer_embedder_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        SentenceTransformerEmbedder("")

    with pytest.raises(ValueError):
        SentenceTransformerEmbedder("modelo", batch_size=0)


def test_sentence_transformer_embedder_rejects_empty_query():
    embedder = SentenceTransformerEmbedder(
        "modelo-de-prueba",
        model_loader=lambda name, device: FakeSentenceTransformer(),
    )

    with pytest.raises(ValueError):
        embedder.embed_query("   ")


# --- BM25 -------------------------------------------------------------------


def test_bm25_ranks_relevant_document_first():
    index = BM25Index(
        [
            "Celsia construye granjas solares en el Valle del Cauca.",
            "Consulta las vacantes de empleo y trabajo en Celsia.",
        ]
    )

    scores = index.scores("empleo")

    assert scores[0] == 0.0
    assert scores[1] > 0.0


def test_bm25_returns_zero_scores_for_query_without_corpus_terms():
    index = BM25Index(["Celsia construye granjas solares en el Valle del Cauca."])

    assert index.scores("inexistente") == [0.0]


def test_bm25_returns_zero_scores_for_empty_query():
    index = BM25Index(["Celsia construye granjas solares en el Valle del Cauca."])

    assert index.scores("") == [0.0]
    assert index.scores("   ") == [0.0]


# --- Estrategia híbrida ------------------------------------------------------


def test_retriever_hybrid_reorders_candidates_with_keyword_scores(built_index):
    index_dir, _, _ = built_index
    retriever = Retriever(index_dir=index_dir, embedder=ArbitraryEmbedder(), strategy="hybrid")

    results = retriever.retrieve("empleo", k=3)

    assert results[0].id == "source_02-chunk-000"


def test_retriever_vector_strategy_keeps_pure_vector_ordering(built_index):
    index_dir, _, _ = built_index
    retriever = Retriever(index_dir=index_dir, embedder=ArbitraryEmbedder(), strategy="vector")

    results = retriever.retrieve("empleo", k=3)

    assert results[0].id == "source_01-chunk-000"


def test_retriever_hybrid_returns_at_most_k_results(built_index):
    index_dir, _, _ = built_index
    retriever = Retriever(index_dir=index_dir, embedder=ArbitraryEmbedder(), strategy="hybrid")

    results = retriever.retrieve("empleo", k=2)

    assert len(results) == 2


def test_retriever_hybrid_returns_fused_rrf_score(built_index):
    index_dir, _, _ = built_index
    retriever = Retriever(index_dir=index_dir, embedder=ArbitraryEmbedder(), strategy="hybrid")

    results = retriever.retrieve("empleo", k=3)

    assert results[0].id == "source_02-chunk-000"
    expected_fused = 1.0 / (RRF_K + 2) + 1.0 / (RRF_K + 1)
    assert results[0].score == pytest.approx(expected_fused)
    assert results[0].score != pytest.approx(1.0 - results[0].distance)


def test_retriever_hybrid_returns_deterministic_ranking(built_index):
    index_dir, _, _ = built_index
    retriever = Retriever(index_dir=index_dir, embedder=ArbitraryEmbedder(), strategy="hybrid")

    first = retriever.retrieve("empleo", k=3)
    second = retriever.retrieve("empleo", k=3)

    assert [result.id for result in first] == [result.id for result in second]
    assert [result.score for result in first] == [result.score for result in second]
