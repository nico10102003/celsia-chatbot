"""Construcción y gestión del índice vectorial persistente de CelsiaBot.

El índice se guarda en Chroma usando embeddings calculados explícitamente. No se
configura una ``embedding_function`` en la colección: así Chroma nunca descarga
su modelo por defecto y el embedder queda bajo control de este módulo.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from src.retrieval.embeddings import DEFAULT_MODEL_NAME, SentenceTransformerEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHUNKS_PATH = PROJECT_ROOT / "data" / "processed" / "chunks.jsonl"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "index"

COLLECTION_NAME = "celsia_chunks"
MANIFEST_FILENAME = "manifest.json"
CHROMA_DIRNAME = "chroma"
DEFAULT_BUILD_BATCH_SIZE = 64


class IndexNotFoundError(RuntimeError):
    """Se lanza cuando se consulta un índice que todavía no existe."""


def load_chunks(chunks_path=DEFAULT_CHUNKS_PATH):
    """Carga y valida los chunks generados por ``src.processing``.

    Args:
        chunks_path: Ruta al archivo JSONL de chunks.

    Returns:
        Una lista de diccionarios con al menos ``id`` y ``text``.

    Raises:
        FileNotFoundError: Si el archivo no existe.
        ValueError: Si una línea no es JSON válido o le falta un campo requerido.
    """
    chunks_path = Path(chunks_path)

    if not chunks_path.is_file():
        raise FileNotFoundError(f"No existe el archivo de chunks: {chunks_path}")

    chunks = []
    seen_ids = set()

    for line_number, line in enumerate(
        chunks_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        try:
            chunk = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"JSON inválido en {chunks_path}, línea {line_number}: {error.msg}"
            ) from error

        if not isinstance(chunk, dict):
            raise TypeError(
                f"El chunk de la línea {line_number} debe ser un objeto JSON."
            )

        chunk_id = chunk.get("id")
        text = chunk.get("text")

        if not isinstance(chunk_id, str) or not chunk_id.strip():
            raise ValueError(f"El chunk de la línea {line_number} no tiene 'id'.")

        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"El chunk de la línea {line_number} no tiene 'text'.")

        if chunk_id in seen_ids:
            raise ValueError(f"El id '{chunk_id}' está duplicado en {chunks_path}.")

        seen_ids.add(chunk_id)
        chunks.append(chunk)

    return chunks


def _chunk_metadata(chunk):
    """Extrae metadatos compatibles con Chroma, sin valores nulos."""
    metadata = {}

    for key in (
        "source_id",
        "title",
        "url",
        "category",
        "extracted_at",
        "chunk_index",
        "characters",
    ):
        value = chunk.get(key)

        if isinstance(value, (str, int, float, bool)):
            metadata[key] = value

    return metadata


def _chunks_sha256(chunks_path):
    """Calcula una huella estable del archivo de chunks."""
    digest = hashlib.sha256()

    with Path(chunks_path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _embed_all(chunks, embedder, batch_size):
    """Vectoriza todos los chunks y valida la forma de los vectores."""
    embeddings = []
    dimension = None

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        batch_embeddings = embedder.embed_documents([chunk["text"] for chunk in batch])

        if len(batch_embeddings) != len(batch):
            raise ValueError(
                "El embedder devolvió una cantidad de vectores distinta a la de textos."
            )

        for vector in batch_embeddings:
            vector = [float(value) for value in vector]

            if not vector:
                raise ValueError("El embedder devolvió un vector vacío.")

            if dimension is None:
                dimension = len(vector)
            elif len(vector) != dimension:
                raise ValueError(
                    "El embedder devolvió vectores con dimensiones inconsistentes."
                )

            embeddings.append(vector)

    return embeddings, dimension or 0


def _make_client(index_dir):
    """Abre el cliente persistente de Chroma dentro del directorio del índice."""
    chroma_dir = Path(index_dir) / CHROMA_DIRNAME
    chroma_dir.mkdir(parents=True, exist_ok=True)

    return chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def build_index(
    chunks_path=DEFAULT_CHUNKS_PATH,
    index_dir=DEFAULT_INDEX_DIR,
    embedder=None,
    model_name=None,
    batch_size=DEFAULT_BUILD_BATCH_SIZE,
    reset=True,
):
    """Construye el índice vectorial persistente a partir de los chunks.

    Args:
        chunks_path: Archivo JSONL de entrada.
        index_dir: Directorio donde se guarda Chroma y su manifiesto.
        embedder: Embedder opcional (por defecto, SentenceTransformers).
        model_name: Nombre registrado en el manifiesto. Si se omite, se toma del
            embedder.
        batch_size: Cantidad de chunks procesados por lote.
        reset: Si es ``True``, reemplaza la colección existente.

    Returns:
        El manifiesto del índice generado.
    """
    if batch_size <= 0:
        raise ValueError("batch_size debe ser mayor que cero.")

    chunks_path = Path(chunks_path)
    index_dir = Path(index_dir)
    chunks = load_chunks(chunks_path)

    if embedder is None:
        embedder = SentenceTransformerEmbedder(model_name or DEFAULT_MODEL_NAME)

    model_name = model_name or getattr(
        embedder,
        "model_name",
        embedder.__class__.__name__,
    )

    # Se vectoriza antes de tocar el índice existente: si el modelo falla, el
    # índice anterior queda intacto.
    embeddings, dimension = _embed_all(chunks, embedder, batch_size)

    index_dir.mkdir(parents=True, exist_ok=True)
    client = _make_client(index_dir)

    if reset:
        try:
            client.delete_collection(COLLECTION_NAME)
        except NotFoundError:
            pass

        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )
    else:
        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=None,
            metadata={"hnsw:space": "cosine"},
        )

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]

        collection.add(
            ids=[chunk["id"] for chunk in batch],
            embeddings=embeddings[start : start + len(batch)],
            documents=[chunk["text"] for chunk in batch],
            metadatas=[_chunk_metadata(chunk) for chunk in batch],
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "chunks_file": str(chunks_path),
        "chunks_sha256": _chunks_sha256(chunks_path),
        "index_dir": str(index_dir),
        "chroma_dir": str(index_dir / CHROMA_DIRNAME),
        "collection_name": COLLECTION_NAME,
        "embedding_model": model_name,
        "embedding_dimension": dimension,
        "total_chunks": len(chunks),
        "total_sources": len({chunk.get("source_id") for chunk in chunks}),
    }

    manifest_path = index_dir / MANIFEST_FILENAME
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)

    return manifest


def load_index_manifest(index_dir=DEFAULT_INDEX_DIR):
    """Lee el manifiesto del índice, o ``None`` si todavía no existe."""
    manifest_path = Path(index_dir) / MANIFEST_FILENAME

    if not manifest_path.is_file():
        return None

    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"No se pudo leer el manifiesto del índice: {manifest_path}"
        ) from error


def open_collection(index_dir=DEFAULT_INDEX_DIR, collection_name=COLLECTION_NAME):
    """Abre la colección persistente para consultarla."""
    manifest = load_index_manifest(index_dir)

    if manifest is None:
        raise IndexNotFoundError(
            f"No hay un índice construido en {index_dir}. "
            "Ejecuta 'python -m src.retrieval build'."
        )

    client = _make_client(index_dir)

    try:
        return client.get_collection(
            name=collection_name,
            embedding_function=None,
        )
    except NotFoundError as error:
        raise IndexNotFoundError(
            f"La colección '{collection_name}' no existe en {index_dir}."
        ) from error
