import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import streamlit as st

from src.generation.answer import generate_answer
from src.retrieval.retriever import Retriever

st.set_page_config(page_title="CelsiaBot", page_icon="⚡", layout="centered")

st.title("⚡ CelsiaBot")
st.subheader("Asistente virtual de información corporativa de Celsia")
st.write("Consulta información sobre sostenibilidad, gobierno corporativo, oportunidades laborales y otros temas publicados por Celsia.")

with st.sidebar:
    st.header("Acerca de CelsiaBot")
    st.write("Este asistente utiliza información recopilada de fuentes públicas.")
    st.caption("Verifica la información importante en las fuentes oficiales.")

@st.cache_resource
def load_retriever():
    return Retriever(strategy="hybrid")

try:
    retriever = load_retriever()
except Exception as exc:
    st.error("No se pudo cargar el buscador.")
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
            urls = list(dict.fromkeys(
                result.url for result in answer.sources if result.url
            ))

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
                            st.markdown(f"[Abrir fuente original]({result.url})")

                        st.divider()
                else:
                    st.info("No se encontraron fragmentos relevantes.")

        except Exception as exc:
            st.error("No fue posible procesar la consulta.")
            st.exception(exc)
