"""Limpieza y fragmentación de las fuentes extraídas para CelsiaBot."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.processing.cleaner import clean_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

CHUNKS_FILENAME = "chunks.jsonl"
MANIFEST_FILENAME = "manifest.json"

# Se intenta cortar primero por párrafos y luego por frases.
SEPARATORS = ["\n\n", "\n", ". ", ", ", " ", ""]


def make_splitter(chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Crea el fragmentador de texto usado para generar los chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size debe ser mayor que cero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError(
            "chunk_overlap debe ser mayor o igual que cero y menor que chunk_size."
        )

    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=SEPARATORS,
        keep_separator=True,
    )


def split_text(text, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """Limpia y fragmenta un texto usando la configuración indicada."""
    cleaned = clean_text(text)

    if not cleaned:
        return []

    splitter = make_splitter(chunk_size, chunk_overlap)
    return [
        chunk.strip()
        for chunk in splitter.split_text(cleaned)
        if chunk.strip()
    ]


def load_raw_documents(raw_dir=RAW_DIR):
    """Carga los documentos crudos registrados como exitosos en el manifiesto.

    Si no existe ``manifest.json``, se usan todos los archivos ``source_*.json``
    ordenados por nombre. Los archivos ausentes o con JSON inválido se omiten
    para no interrumpir el procesamiento completo.
    """
    raw_dir = Path(raw_dir)
    manifest_path = raw_dir / MANIFEST_FILENAME

    if manifest_path.is_file():
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            manifest = {}

        entries = manifest.get("results", [])
        filenames = [
            entry.get("file")
            for entry in entries
            if entry.get("status") == "success" and entry.get("file")
        ]
    else:
        filenames = [
            path.name
            for path in sorted(raw_dir.glob("source_*.json"))
        ]

    documents = []

    for filename in filenames:
        path = raw_dir / filename

        if not path.is_file():
            continue

        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        if not isinstance(document, dict) or not document.get("text"):
            continue

        document["source_id"] = path.stem
        document["file"] = filename
        documents.append(document)

    return documents


def process_document(document, splitter):
    """Limpia y fragmenta un documento crudo.

    Returns:
        Una tupla ``(chunks, result)`` donde ``chunks`` contiene los fragmentos
        generados y ``result`` resume el procesamiento de la fuente.
    """
    source_id = document.get("source_id") or document.get("file", "unknown")
    source_id = Path(source_id).stem

    result = {
        "source_id": source_id,
        "file": document.get("file"),
        "title": document.get("title"),
        "url": document.get("url"),
        "category": document.get("category"),
        "characters": len(document.get("text", "")),
    }

    cleaned = clean_text(document.get("text", ""))

    if not cleaned:
        result["reason"] = "empty_text"
        return [], result

    result["cleaned_characters"] = len(cleaned)

    chunks = []
    seen = set()

    for piece in splitter.split_text(cleaned):
        piece = piece.strip()

        if not piece or piece in seen:
            continue

        seen.add(piece)
        chunk_index = len(chunks)
        chunks.append(
            {
                "id": f"{source_id}-chunk-{chunk_index:03d}",
                "source_id": source_id,
                "title": document.get("title"),
                "url": document.get("url"),
                "category": document.get("category"),
                "extracted_at": document.get("extracted_at"),
                "chunk_index": chunk_index,
                "characters": len(piece),
                "text": piece,
            }
        )

    result["chunks"] = len(chunks)

    if not chunks:
        result["reason"] = "no_chunks"

    return chunks, result


def process_all(
    raw_dir=RAW_DIR,
    output_dir=PROCESSED_DIR,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
):
    """Procesa todas las fuentes y escribe ``chunks.jsonl`` y ``manifest.json``.

    Returns:
        El manifiesto generado, con el resumen de fuentes y chunks.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)
    splitter = make_splitter(chunk_size, chunk_overlap)

    documents = load_raw_documents(raw_dir)

    all_chunks = []
    results = []
    skipped = []

    for document in documents:
        chunks, result = process_document(document, splitter)

        if chunks:
            all_chunks.extend(chunks)
            results.append(result)
        else:
            skipped.append(result)

    output_dir.mkdir(parents=True, exist_ok=True)

    chunks_path = output_dir / CHUNKS_FILENAME
    chunks_path.write_text(
        "".join(
            json.dumps(chunk, ensure_ascii=False) + "\n"
            for chunk in all_chunks
        ),
        encoding="utf-8",
    )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "total_sources": len(documents),
        "processed_sources": len(results),
        "skipped_sources": len(skipped),
        "total_chunks": len(all_chunks),
        "results": results,
        "skipped": skipped,
    }

    manifest_path = output_dir / MANIFEST_FILENAME
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return manifest


def build_parser():
    """Construye el analizador de argumentos del módulo."""
    parser = argparse.ArgumentParser(
        prog="python -m src.processing",
        description="Limpia y fragmenta las fuentes extraídas para CelsiaBot.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directorio con los JSON crudos y su manifiesto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROCESSED_DIR,
        help="Directorio donde se escribirán los chunks y el manifiesto.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Tamaño máximo de cada chunk (por defecto {DEFAULT_CHUNK_SIZE}).",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP,
        help=f"Solapamiento entre chunks (por defecto {DEFAULT_CHUNK_OVERLAP}).",
    )
    return parser


def main(argv=None):
    """Punto de entrada de línea de comandos."""
    args = build_parser().parse_args(argv)
    manifest = process_all(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )

    print("\n--- RESUMEN DE PROCESAMIENTO ---")
    print(f"Fuentes encontradas: {manifest['total_sources']}")
    print(f"Fuentes procesadas: {manifest['processed_sources']}")
    print(f"Fuentes omitidas: {manifest['skipped_sources']}")
    print(f"Chunks generados: {manifest['total_chunks']}")
    print(f"Chunks: {Path(args.output_dir) / CHUNKS_FILENAME}")
    print(f"Manifiesto: {Path(args.output_dir) / MANIFEST_FILENAME}")

    return 0
