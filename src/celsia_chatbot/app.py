import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.generation.answer import generate_answer
from src.retrieval.retriever import Retriever

RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

st.set_page_config(
    page_title="CelsiaBot",
    page_icon="⚡",
    layout="wide",
)

st.title("⚡ CelsiaBot")
st.subheader("Asistente virtual de información corporativa de Celsia")
st.write(
    "Consulta información corporativa a partir de fuentes públicas oficiales. "
    "El sistema extrae, procesa y recupera documentos para responder tus preguntas."
)


def read_json(path):
    """Lee un archivo JSON si existe y contiene información válida."""
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {}


def get_metrics():
    """Calcula métricas a partir de los manifiestos y fragmentos generados."""
    raw_manifest = read_json(RAW_DIR / "manifest.json")
    processed_manifest = read_json(PROCESSED_DIR / "manifest.json")

    results = raw_manifest.get("results", [])
    successful = [
        item for item in results if item.get("status") == "success"
    ]

    chunks_path = PROCESSED_DIR / "chunks.jsonl"
    chunks = []

    if chunks_path.exists():
        with chunks_path.open("r", encoding="utf-8-sig") as file:
            for line in file:
                if line.strip():
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

    characters = sum(
        len(str(chunk.get("text", "")))
        for chunk in chunks
    )

    tokens_estimated = round(characters / 4)

    return {
        "registered": raw_manifest.get("total_sources", len(results)),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "chunks": len(chunks),
        "characters": characters,
        "tokens": tokens_estimated,
        "processed_sources": processed_manifest.get(
            "processed_sources",
            processed_manifest.get("sources_processed", len(successful)),
        ),
        "last_scrape": raw_manifest.get("generated_at", "Sin registro"),
        "raw_results": results,
    }


metrics = get_metrics()

with st.sidebar:
    st.header("Acerca de CelsiaBot")
    st.write(
        "Asistente académico que utiliza información pública "
        "de Celsia para responder preguntas."
    )
    st.caption(
        "Comprueba los datos importantes consultando las fuentes oficiales."
    )

    st.divider()
    st.subheader("Etapas del proceso")

    stages = [
        ("1. Descubrir fuentes", metrics["registered"] > 0),
        ("2. Extraer contenido", metrics["successful"] > 0),
        ("3. Limpiar y procesar", metrics["chunks"] > 0),
        ("4. Generar fragmentos", metrics["chunks"] > 0),
        ("5. Recuperar información", True),
    ]

    for label, completed in stages:
        st.write(f"{'✅' if completed else '⏳'} {label}")

    st.caption("Última extracción registrada:")
    st.caption(str(metrics["last_scrape"]))

st.markdown("## Estado del corpus")

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("Fuentes registradas", metrics["registered"])
col2.metric("Extracciones exitosas", metrics["successful"])
col3.metric("Fragmentos", metrics["chunks"])
col4.metric("Caracteres", f'{metrics["characters"]:,}')
col5.metric("Tokens estimados", f'{metrics["tokens"]:,}')

if metrics["failed"] > 0:
    st.warning(f"Hay {metrics['failed']} fuentes que no se extrajeron correctamente.")
else:
    st.success("Las fuentes registradas se extrajeron correctamente.")

with st.expander("Ver detalle de las fuentes"):
    if metrics["raw_results"]:
        for item in metrics["raw_results"]:
            status = item.get("status", "sin estado")
            icon = "✅" if status == "success" else "⚠️"
            st.write(
                f"{icon} **{item.get('title', 'Fuente sin título')}** — "
                f"{status} — {item.get('characters', 0):,} caracteres"
            )
            if item.get("url"):
                st.markdown(f"[Abrir fuente oficial]({item['url']})")
    else:
        st.info("Todavía no hay un manifiesto de extracción.")

st.divider()
st.markdown("## Pregúntale a CelsiaBot")

@st.cache_resource
def load_retriever():
    return Retriever(strategy="hybrid")


try:
    retriever = load_retriever()
except (OSError, ValueError, RuntimeError) as exc:
    st.error(
        "No se pudo cargar el buscador. Comprueba que el índice "
        "vectorial esté creado."
    )
    st.exception(exc)
    st.stop()

question = st.text_input(
    "¿Qué quieres saber sobre Celsia?",
    placeholder="Ej.: ¿Dónde puedo consultar las vacantes?",
)

if st.button("Consultar", type="primary", disabled=not question.strip()):
    with st.spinner("Buscando información y generando respuesta..."):
        try:
            results = retriever.retrieve(question.strip(), k=5)
            answer = generate_answer(question.strip(), results)

            st.markdown("### Respuesta")
            st.markdown(answer.text)

            st.markdown("### Fuentes consultadas")
            urls = list(
                dict.fromkeys(
                    result.url
                    for result in answer.sources
                    if result.url
                )
            )

            if urls:
                for url in urls:
                    st.markdown(f"- [Consultar fuente oficial]({url})")
            else:
                st.info("No se encontraron URL asociadas a las fuentes.")

            with st.expander("Ver fragmentos recuperados"):
                if results:
                    for result in results:
                        st.markdown(f"**Fragmento:** `{result.id}`")
                        st.write(result.text)
                        st.caption(f"Puntuación: {result.score:.4f}")

                        if result.url:
                            st.markdown(
                                f"[Abrir fuente original]({result.url})"
                            )

                        st.divider()
                else:
                    st.info("No se encontraron fragmentos relevantes.")

        except (OSError, ValueError, RuntimeError) as exc:
            st.error("No fue posible procesar la consulta.")
            st.exception(exc)


