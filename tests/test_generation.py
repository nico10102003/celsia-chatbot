"""Pruebas de la generación de respuestas sin Ollama ni descargas de modelos."""

from types import SimpleNamespace

from langchain_ollama import ChatOllama

from src.generation.answer import (
    NO_CONTEXT_ANSWER,
    Answer,
    build_prompt,
    generate_answer,
)
from src.generation.context import build_context
from src.generation.llm import (
    DEFAULT_MODEL_NAME,
    DEFAULT_TEMPERATURE,
    create_chat_model,
)
from src.retrieval.retriever import RetrievalResult

EXPECTED_SOURCES_TEXT = (
    "Fuentes utilizadas:\n- https://www.celsia.com/es/granjas-solares/"
)


def make_result(
    chunk_id="source_01-chunk-000",
    text="Celsia construye granjas solares en el Valle del Cauca.",
    url="https://www.celsia.com/es/granjas-solares/",
    title="Granjas solares",
    category="energia_solar",
):
    return RetrievalResult(
        id=chunk_id,
        text=text,
        score=0.98,
        distance=0.02,
        metadata={"url": url, "title": title, "category": category},
    )


class FakeChatModel:
    """Chat model simulado que no toca Ollama."""

    def __init__(self, response: str = "Respuesta simulada."):
        self.response = response
        self.prompts = []

    def invoke(self, prompt, config=None):
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.response)


# --- Conexión con el modelo -------------------------------------------------


def test_create_chat_model_uses_default_deepseek_model():
    chat_model = create_chat_model()

    assert isinstance(chat_model, ChatOllama)
    assert chat_model.model == DEFAULT_MODEL_NAME
    assert chat_model.temperature == DEFAULT_TEMPERATURE


# --- Construcción del contexto ----------------------------------------------


def test_build_context_formats_fragment_with_url_and_text():
    result = make_result()

    context = build_context([result])

    assert "Fragmento 1:" in context
    assert result.text in context
    assert result.url in context
    assert result.metadata["title"] in context
    assert result.metadata["category"] in context


def test_build_context_without_results_returns_empty_string():
    assert build_context([]) == ""


# --- Construcción del prompt ------------------------------------------------


def test_build_prompt_contains_question_and_context():
    context = "Fragmento 1:\nURL: https://www.celsia.com/es/granjas-solares/"

    prompt = build_prompt("¿Cuántas granjas solares tiene Celsia?", context)

    assert "¿Cuántas granjas solares tiene Celsia?" in prompt
    assert context in prompt
    assert "Contexto recuperado" in prompt
    assert "no inventes" in prompt


# --- Generación de respuestas ---------------------------------------------


def test_generate_answer_returns_answer_with_sources():
    llm = FakeChatModel("Celsia construye granjas solares en el Valle del Cauca.")
    result = make_result()
    answer = generate_answer("¿Dónde construye Celsia sus granjas solares?", [result], llm=llm)

    assert isinstance(answer, Answer)
    assert answer.text == (
        "Celsia construye granjas solares en el Valle del Cauca.\n\n" + EXPECTED_SOURCES_TEXT
    )
    assert answer.sources == [result]
    assert answer.context == build_context([result])
    assert len(llm.prompts) == 1


def test_generate_answer_appends_sources_to_response():
    answer = generate_answer("¿Qué ofrece Celsia?", [make_result()], llm=FakeChatModel("No tengo datos."))

    assert "No tengo datos." in answer.text
    assert answer.text.endswith(EXPECTED_SOURCES_TEXT)


def test_generate_answer_builds_context_from_recovered_fragments():
    llm = FakeChatModel("Respuesta.")
    results = [
        make_result(),
        make_result(chunk_id="source_02-chunk-000", text="Empleos en Celsia."),
    ]

    answer = generate_answer("¿Qué ofrece Celsia?", results, llm=llm)

    assert "Celsia construye granjas solares" in answer.context
    assert "Empleos en Celsia." in answer.context
    assert len(answer.sources) == 2
    assert len(llm.prompts) == 1


def test_generate_answer_uses_injected_llm():
    llm = FakeChatModel("Respuesta simulada.")
    answer = generate_answer(
        "¿Cuál es el horario?",
        [make_result()],
        llm=llm,
        model_name="otro-modelo",
        temperature=0.5,
    )

    assert answer.text == "Respuesta simulada.\n\nFuentes utilizadas:\n- https://www.celsia.com/es/granjas-solares/"
    assert llm.prompts[0].startswith("Eres CelsiaBot")


def test_generate_answer_without_results_returns_not_found_answer():
    answer = generate_answer("¿Qué sé?", [], llm=FakeChatModel())

    assert isinstance(answer, Answer)
    assert answer.text == NO_CONTEXT_ANSWER
    assert answer.sources == []
    assert answer.context == ""
