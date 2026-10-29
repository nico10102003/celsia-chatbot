# ⚡ CelsiaBot

**Asistente virtual de información corporativa de Celsia mediante inteligencia artificial y generación aumentada por recuperación (RAG).**

[![Python](https://img.shields.io/badge/Python-3.13-blue)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Interface-Streamlit-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-65%20passed-brightgreen)]()

## Descripción

CelsiaBot es un proyecto académico que implementa un chatbot capaz de responder preguntas sobre información corporativa de Celsia.

El sistema recopila información de fuentes públicas oficiales, procesa los documentos, genera fragmentos de texto y utiliza técnicas de recuperación híbrida para proporcionar contexto a un modelo de lenguaje.

## Objetivo

Facilitar la consulta de información corporativa de Celsia mediante una interfaz conversacional, proporcionando respuestas acompañadas de enlaces a las fuentes consultadas.

## Características principales

- Extracción automatizada de información desde sitios oficiales.
- Verificación de las reglas de `robots.txt`.
- Limpieza y procesamiento de documentos.
- División de documentos en fragmentos (*chunking*).
- Recuperación híbrida mediante búsqueda vectorial y BM25.
- Generación de respuestas mediante un modelo de lenguaje local.
- Visualización de métricas del corpus.
- Consulta de fuentes y fragmentos recuperados.
- Pruebas automatizadas con Pytest.

## Tecnologías utilizadas

| Tecnología | Función |
|---|---|
| Python | Lenguaje principal |
| UV | Gestión de dependencias |
| Streamlit | Interfaz gráfica |
| BeautifulSoup | Procesamiento web |
| Trafilatura | Extracción de contenido |
| LangChain | Procesamiento y generación |
| ChromaDB | Almacenamiento vectorial |
| Sentence Transformers | Embeddings |
| Ollama | Ejecución local del modelo |
| Pytest | Pruebas automatizadas |
| Ruff | Calidad del código |

## Arquitectura

El proyecto está organizado en las siguientes etapas:

1. **Scraping:** consulta las fuentes públicas de Celsia y extrae su contenido.
2. **Processing:** limpia los textos y genera fragmentos de información.
3. **Retrieval:** recupera los fragmentos relevantes mediante búsqueda híbrida.
4. **Generation:** utiliza el contexto recuperado para generar respuestas.
5. **Presentation:** presenta el chatbot, las métricas y las fuentes mediante Streamlit.

## Estructura del proyecto

```text
celsia-chatbot/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── evaluation/
│   └── index/
├── docs/
├── reports/
├── scripts/
├── src/
│   ├── scraping/
│   ├── processing/
│   ├── retrieval/
│   ├── generation/
│   └── celsia_chatbot/
├── tests/
├── .gitignore
├── pyproject.toml
├── uv.lock
├── README.md
└── LICENSE
```

## Instalación

Requisitos:

- Python 3.13.
- UV.
- Ollama.

Clonar el repositorio:

```bash
git clone https://github.com/nico10102003/celsia-chatbot.git
cd celsia-chatbot
```

Instalar dependencias:

```bash
uv sync
```

Descargar el modelo de lenguaje:

```bash
ollama pull deepseek-r1:1.5b
```

## Ejecución

Ejecutar la extracción de información:

```bash
uv run python -m src.scraping
```

Procesar los documentos:

```bash
uv run python -m src.processing
```

Iniciar la interfaz:

```bash
uv run streamlit run src/celsia_chatbot/app.py
```

## Pruebas

Ejecutar las pruebas automatizadas:

```bash
uv run pytest -q
```

Verificar la calidad del código:

```bash
uv run ruff check .
```

## Resultados actuales

- 10 fuentes oficiales registradas.
- 10 fuentes extraídas correctamente.
- 66 fragmentos generados.
- 65 pruebas automatizadas aprobadas.

## Fuentes de información

El sistema utiliza información pública publicada en el sitio oficial de Celsia:

https://www.celsia.com/

La información generada debe contrastarse con las publicaciones oficiales de la empresa.

## Autor

Proyecto académico desarrollado por **Nicolás Bolaños**.

## Licencia

Este proyecto se distribuye bajo la licencia MIT. Consulta el archivo `LICENSE` para conocer sus condiciones.