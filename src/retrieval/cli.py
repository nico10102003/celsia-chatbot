"""Interfaz de línea de comandos del índice de recuperación semántica."""

import argparse
from pathlib import Path

from src.retrieval.embeddings import DEFAULT_MODEL_NAME
from src.retrieval.index import (
    DEFAULT_BUILD_BATCH_SIZE,
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_DIR,
    build_index,
    load_index_manifest,
)
from src.retrieval.retriever import DEFAULT_TOP_K, Retriever


def _build_command(args):
    manifest = build_index(
        chunks_path=args.chunks,
        index_dir=args.index_dir,
        model_name=args.model,
        batch_size=args.batch_size,
        reset=not args.append,
    )

    print("\n--- RESUMEN DEL ÍNDICE ---")
    print(f"Chunks indexados: {manifest['total_chunks']}")
    print(f"Fuentes: {manifest['total_sources']}")
    print(f"Modelo: {manifest['embedding_model']}")
    print(f"Dimensión: {manifest['embedding_dimension']}")
    print(f"Índice: {Path(manifest['chroma_dir'])}")
    print(f"Manifiesto: {Path(manifest['index_dir']) / 'manifest.json'}")
    return 0


def _query_command(args):
    manifest = load_index_manifest(args.index_dir) or {}
    retriever = Retriever(
        index_dir=args.index_dir,
        model_name=args.model or manifest.get("embedding_model"),
        device=args.device,
    )
    where = {"category": args.category} if args.category else None
    results = retriever.retrieve(args.text, k=args.k, where=where)

    if not results:
        print("No se encontraron fragmentos relevantes.")
        return 0

    for position, result in enumerate(results, start=1):
        print(f"\n[{position}] score={result.score:.4f} id={result.id}")
        print(f"URL: {result.url or '(sin URL)'}")

        if result.metadata.get("title"):
            print(f"Título: {result.metadata['title']}")

        print(result.text)
        print("-" * 72)

    return 0


def build_parser():
    """Construye el analizador de argumentos con los subcomandos build/query."""
    parser = argparse.ArgumentParser(
        prog="python -m src.retrieval",
        description="Construye y consulta el índice semántico de CelsiaBot.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser(
        "build",
        help="Construye el índice vectorial a partir de chunks.jsonl.",
    )
    build_parser.add_argument(
        "--chunks",
        type=Path,
        default=DEFAULT_CHUNKS_PATH,
        help="Archivo JSONL de chunks.",
    )
    build_parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio donde se guardará el índice.",
    )
    build_parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_NAME,
        help=f"Modelo de embeddings (por defecto {DEFAULT_MODEL_NAME}).",
    )
    build_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BUILD_BATCH_SIZE,
        help=f"Chunks por lote (por defecto {DEFAULT_BUILD_BATCH_SIZE}).",
    )
    build_parser.add_argument(
        "--append",
        action="store_true",
        help="Añade los chunks a la colección existente en vez de recrearla.",
    )
    build_parser.set_defaults(func=_build_command)

    query_parser = subparsers.add_parser(
        "query",
        help="Busca fragmentos relevantes para una consulta.",
    )
    query_parser.add_argument("text", help="Texto de la consulta.")
    query_parser.add_argument(
        "--k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Cantidad de resultados (por defecto {DEFAULT_TOP_K}).",
    )
    query_parser.add_argument(
        "--category",
        default=None,
        help="Filtra por la categoría de la fuente.",
    )
    query_parser.add_argument(
        "--index-dir",
        type=Path,
        default=DEFAULT_INDEX_DIR,
        help="Directorio que contiene el índice.",
    )
    query_parser.add_argument(
        "--model",
        default=None,
        help="Modelo de embeddings; por defecto se usa el del manifiesto.",
    )
    query_parser.add_argument(
        "--device",
        default=None,
        help="Dispositivo para el modelo (por ejemplo cpu).",
    )
    query_parser.set_defaults(func=_query_command)

    return parser


def main(argv=None):
    """Punto de entrada de línea de comandos."""
    args = build_parser().parse_args(argv)
    return args.func(args)
