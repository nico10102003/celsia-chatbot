"""Pruebas del procesamiento de fuentes de CelsiaBot."""

import json

import pytest

from src.processing.cleaner import clean_text
from src.processing.processor import (
    CHUNKS_FILENAME,
    MANIFEST_FILENAME,
    load_raw_documents,
    main,
    process_all,
    split_text,
)

SAMPLE_TEXT = (
    "Celsia es una empresa de energía del Grupo Argos. "
    "Ofrece servicios de generación, distribución y comercialización de energía. "
    "Trabaja en proyectos de energía solar y eficiencia energética. "
) * 12


def write_source(raw_dir, filename, text, **overrides):
    """Crea un JSON crudo de prueba con la estructura del extractor."""
    document = {
        "title": "Fuente de prueba",
        "url": "https://www.celsia.com/es/",
        "category": "corporativo",
        "extracted_at": "2026-01-01T00:00:00+00:00",
        "text": text,
    }
    document.update(overrides)

    path = raw_dir / filename
    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def write_manifest(raw_dir, results):
    """Escribe un manifiesto crudo de prueba."""
    path = raw_dir / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "generated_at": "2026-01-01T00:00:00+00:00",
                "total_sources": len(results),
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# --- Limpieza -------------------------------------------------------------


def test_clean_text_normalizes_whitespace():
    assert (
        clean_text("  Hola   mundo \r\n\r\n\r\n segunda\tlínea ")
        == "Hola mundo\n\nsegunda línea"
    )


def test_clean_text_removes_control_characters_and_nbsp():
    assert clean_text("Hola\x00 mundo\u00a0aquí") == "Hola mundo aquí"


def test_clean_text_removes_separator_lines():
    assert clean_text("Título\n-\n—\nContenido") == "Título\n\nContenido"


def test_clean_text_collapses_consecutive_duplicates():
    assert (
        clean_text("Celsia\nCelsia\nEnergía\n\nEnergía\nSolar")
        == "Celsia\nEnergía\n\nSolar"
    )


def test_clean_text_handles_empty_values():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   \n\n---\n  ") == ""


# --- Fragmentación --------------------------------------------------------


def test_split_text_respects_chunk_size():
    chunks = split_text(SAMPLE_TEXT, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(0 < len(chunk) <= 200 for chunk in chunks)
    assert all(chunk == chunk.strip() for chunk in chunks)


def test_split_text_returns_empty_list_for_empty_text():
    assert split_text("", chunk_size=200, chunk_overlap=20) == []


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [(0, 0), (-10, 0), (100, 100), (100, 150), (100, -1)],
)
def test_split_text_rejects_invalid_configuration(chunk_size, chunk_overlap):
    with pytest.raises(ValueError):
        split_text("Texto de prueba", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


# --- Carga de documentos crudos -------------------------------------------


def test_load_raw_documents_uses_manifest_successes(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    write_source(raw_dir, "source_01.json", "Contenido válido.")
    write_source(raw_dir, "source_02.json", "No debe cargarse.")
    write_manifest(
        raw_dir,
        [
            {"status": "success", "file": "source_01.json"},
            {"status": "error", "file": "source_02.json"},
            {"status": "success", "file": "source_99.json"},
        ],
    )

    documents = load_raw_documents(raw_dir)

    assert [document["source_id"] for document in documents] == ["source_01"]
    assert documents[0]["file"] == "source_01.json"
    assert documents[0]["title"] == "Fuente de prueba"


def test_load_raw_documents_falls_back_to_glob(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    write_source(raw_dir, "source_02.json", "Contenido dos.")
    write_source(raw_dir, "source_01.json", "Contenido uno.")

    documents = load_raw_documents(raw_dir)

    assert [document["source_id"] for document in documents] == [
        "source_01",
        "source_02",
    ]


def test_load_raw_documents_skips_invalid_json(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    (raw_dir / "source_01.json").write_text("{json inválido", encoding="utf-8")
    write_source(raw_dir, "source_02.json", "Contenido válido.")
    write_manifest(
        raw_dir,
        [
            {"status": "success", "file": "source_01.json"},
            {"status": "success", "file": "source_02.json"},
        ],
    )

    documents = load_raw_documents(raw_dir)

    assert [document["source_id"] for document in documents] == ["source_02"]


# --- Procesamiento completo -----------------------------------------------


def test_process_all_writes_chunks_and_manifest(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    raw_path = write_source(raw_dir, "source_01.json", SAMPLE_TEXT)
    original_raw = raw_path.read_text(encoding="utf-8")
    write_manifest(raw_dir, [{"status": "success", "file": "source_01.json"}])

    manifest = process_all(
        raw_dir=raw_dir,
        output_dir=output_dir,
        chunk_size=200,
        chunk_overlap=20,
    )

    chunks_path = output_dir / CHUNKS_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    assert chunks_path.is_file()
    assert manifest_path.is_file()
    assert manifest["total_sources"] == 1
    assert manifest["processed_sources"] == 1
    assert manifest["skipped_sources"] == 0
    assert manifest["total_chunks"] > 1

    chunks = [
        json.loads(line)
        for line in chunks_path.read_text(encoding="utf-8").splitlines()
    ]

    assert len(chunks) == manifest["total_chunks"]
    assert len({chunk["id"] for chunk in chunks}) == len(chunks)
    assert [chunk["chunk_index"] for chunk in chunks] == list(range(len(chunks)))

    for chunk in chunks:
        assert chunk["source_id"] == "source_01"
        assert chunk["title"] == "Fuente de prueba"
        assert chunk["url"] == "https://www.celsia.com/es/"
        assert chunk["category"] == "corporativo"
        assert chunk["id"].startswith("source_01-chunk-")
        assert chunk["text"]
        assert chunk["characters"] == len(chunk["text"])
        assert chunk["characters"] <= 200

    saved_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert saved_manifest == manifest
    assert saved_manifest["results"][0]["chunks"] == len(chunks)

    # El procesamiento no debe modificar los archivos crudos.
    assert raw_path.read_text(encoding="utf-8") == original_raw


def test_process_all_skips_sources_without_useful_text(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    write_source(raw_dir, "source_01.json", SAMPLE_TEXT)
    write_source(raw_dir, "source_02.json", " \n\n---\n ")
    write_manifest(
        raw_dir,
        [
            {"status": "success", "file": "source_01.json"},
            {"status": "success", "file": "source_02.json"},
        ],
    )

    manifest = process_all(
        raw_dir=raw_dir,
        output_dir=output_dir,
        chunk_size=200,
        chunk_overlap=20,
    )

    assert manifest["total_sources"] == 2
    assert manifest["processed_sources"] == 1
    assert manifest["skipped_sources"] == 1
    assert manifest["skipped"][0]["source_id"] == "source_02"
    assert manifest["skipped"][0]["reason"] == "empty_text"


def test_process_all_produces_stable_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    write_source(raw_dir, "source_01.json", SAMPLE_TEXT)
    write_manifest(raw_dir, [{"status": "success", "file": "source_01.json"}])

    first = process_all(
        raw_dir=raw_dir,
        output_dir=output_dir,
        chunk_size=200,
        chunk_overlap=20,
    )
    first_ids = [result["source_id"] for result in first["results"]]

    second = process_all(
        raw_dir=raw_dir,
        output_dir=output_dir,
        chunk_size=200,
        chunk_overlap=20,
    )
    second_ids = [result["source_id"] for result in second["results"]]

    assert first_ids == second_ids == ["source_01"]
    assert first["total_chunks"] == second["total_chunks"]


def test_process_all_with_empty_raw_directory(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    manifest = process_all(raw_dir=raw_dir, output_dir=output_dir)

    assert manifest["total_sources"] == 0
    assert manifest["total_chunks"] == 0
    assert (output_dir / CHUNKS_FILENAME).read_text(encoding="utf-8") == ""
    assert (output_dir / MANIFEST_FILENAME).is_file()


# --- Interfaz de línea de comandos ----------------------------------------


def test_main_writes_output_and_returns_zero(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "processed"
    raw_dir.mkdir()

    write_source(raw_dir, "source_01.json", SAMPLE_TEXT)
    write_manifest(raw_dir, [{"status": "success", "file": "source_01.json"}])

    exit_code = main(
        [
            "--raw-dir",
            str(raw_dir),
            "--output-dir",
            str(output_dir),
            "--chunk-size",
            "200",
            "--chunk-overlap",
            "20",
        ]
    )

    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Chunks generados" in captured.out
    assert (output_dir / CHUNKS_FILENAME).is_file()
    assert (output_dir / MANIFEST_FILENAME).is_file()
