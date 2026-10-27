"""Evaluación reproducible de la recuperación RAG: vector vs híbrida.

Ejecuta las preguntas de ``data/evaluation/questions.json`` contra el índice
vectorial existente con ambas estrategias de recuperación (``"vector"`` y
``"hybrid"``) y guarda los resultados por estrategia en JSONL.

Uso:

    uv run python scripts/evaluate_retrieval.py

No modifica el índice ni la implementación del recuperador.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.retriever import Retriever  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "evaluation"
VECTOR_RESULTS_PATH = RESULTS_DIR / "vector_results.jsonl"
HYBRID_RESULTS_PATH = RESULTS_DIR / "hybrid_results.jsonl"

STRATEGIES = ("vector", "hybrid")
TOP_K = 5

QUESTIONS_PATH = PROJECT_ROOT / "data" / "evaluation" / "questions.json"


def load_questions(path: Path = QUESTIONS_PATH) -> list[dict]:
    """Carga las preguntas de evaluación."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []

    normalized = []
    for question in questions:
        normalized.append(
            {
                "id": str(question.get("id") or ""),
                "question": str(question.get("question") or ""),
            }
        )

    return normalized


def evaluate_question(
    retriever: Retriever,
    question_id: str,
    question: str,
    strategy: str,
    k: int = TOP_K,
) -> list[dict]:
    """Recupera hasta ``k`` fragmentos para una pregunta con la estrategia indicada."""
    results = retriever.retrieve(question, k=k)

    records = []
    for position, result in enumerate(results, start=1):
        metadata = dict(result.metadata or {})
        records.append(
            {
                "question_id": question_id,
                "question": question,
                "strategy": strategy,
                "position": position,
                "chunk_id": result.id,
                "text": result.text,
                "score": result.score,
                "distance": result.distance,
                "url": result.url,
                "title": metadata.get("title"),
                "category": metadata.get("category"),
                "source_id": metadata.get("source_id"),
            }
        )

    return records


def run_strategy(strategy: str, questions: list[dict], k: int = TOP_K) -> list[dict]:
    """Ejecuta todas las preguntas con una estrategia y devuelve los registros."""
    retriever = Retriever(strategy=strategy)
    all_records: list[dict] = []

    for entry in questions:
        records = evaluate_question(
            retriever,
            entry["id"],
            entry["question"],
            strategy,
            k=k,
        )

        all_records.extend(records)
        print(f"[{strategy}] {entry['id']} -> {len(records)} resultados")

    return all_records


def write_results(records: list[dict], path: Path) -> None:
    """Escribe los resultados en un archivo JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> int:
    questions = load_questions()

    print(f"Preguntas de evaluación: {len(questions)}")
    print(f"Top-K por pregunta: {TOP_K}")

    vector_records = run_strategy("vector", questions, k=TOP_K)
    write_results(vector_records, VECTOR_RESULTS_PATH)

    hybrid_records = run_strategy("hybrid", questions, k=TOP_K)
    write_results(hybrid_records, HYBRID_RESULTS_PATH)

    print(f"Vector: {len(vector_records)} registros -> {VECTOR_RESULTS_PATH}")
    print(f"Híbrida: {len(hybrid_records)} registros -> {HYBRID_RESULTS_PATH}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
