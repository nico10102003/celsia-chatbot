"""Descarga y extracción respetuosa de contenido público de Celsia."""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
import trafilatura

from src.scraping.sources import SOURCES

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
USER_AGENT = "CelsiaBotAcademic/1.0 (proyecto educativo)"
TIMEOUT = 20

ROBOTS_CACHE = {}


def can_fetch(url):
    """Comprueba robots.txt antes de intentar descargar una página."""
    parsed = urlparse(url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    if robots_url not in ROBOTS_CACHE:
        try:
            response = requests.get(
                robots_url,
                headers={"User-Agent": USER_AGENT},
                timeout=TIMEOUT,
            )
            response.raise_for_status()

            parser = RobotFileParser()
            parser.set_url(robots_url)
            parser.parse(response.text.splitlines())
            ROBOTS_CACHE[robots_url] = parser

        except requests.RequestException as exc:
            print(f"No se pudo comprobar robots.txt: {robots_url} ({exc})")
            ROBOTS_CACHE[robots_url] = None

    parser = ROBOTS_CACHE[robots_url]

    # Si no podemos comprobar las reglas, no descargamos la página.
    if parser is None:
        return False

    return parser.can_fetch(USER_AGENT, url)


def extract_source(source):
    """Descarga una fuente autorizada y extrae su texto principal."""
    url = source["url"]
    print(f"\nConsultando: {source['title']}")

    if not can_fetch(url):
        print("Omitida: no se pudo verificar el permiso de rastreo.")
        return {
            "title": source["title"],
            "url": url,
            "status": "skipped_robots",
        }

    try:
        response = requests.get(
            url,
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        content_type = response.headers.get("Content-Type", "")
        if "text/html" not in content_type.lower():
            print("Omitida: la respuesta no es una página HTML.")
            return {
                "title": source["title"],
                "url": url,
                "status": "skipped_content_type",
            }

        text = trafilatura.extract(
            response.text,
            include_comments=False,
            include_tables=True,
        )

        if not text or len(text.strip()) < 100:
            print("No se encontró suficiente texto útil.")
            return {
                "title": source["title"],
                "url": url,
                "status": "no_text",
            }

        document = {
            "title": source["title"],
            "url": response.url,
            "category": source["category"],
            "extracted_at": datetime.now(timezone.utc).isoformat(),
            "text": text.strip(),
        }

        filename = f"source_{SOURCES.index(source) + 1:02d}.json"
        output_path = RAW_DIR / filename
        output_path.write_text(
            json.dumps(document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        print(f"Guardado: {output_path.name} ({len(text)} caracteres)")

        return {
            "title": source["title"],
            "url": url,
            "status": "success",
            "file": filename,
            "characters": len(text),
        }

    except requests.RequestException as exc:
        print(f"Error de conexión: {exc}")
        return {
            "title": source["title"],
            "url": url,
            "status": "error",
            "error": str(exc),
        }


def main():
    """Procesa las fuentes y genera un manifiesto de resultados."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for index, source in enumerate(SOURCES):
        results.append(extract_source(source))

        if index < len(SOURCES) - 1:
            time.sleep(1)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(SOURCES),
        "results": results,
    }

    manifest_path = RAW_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    successful = sum(
        result["status"] == "success"
        for result in results
    )

    print("\n--- RESUMEN ---")
    print(f"Fuentes registradas: {len(SOURCES)}")
    print(f"Fuentes extraídas: {successful}")
    print(f"Manifiesto: {manifest_path}")


if __name__ == "__main__":
    main()
