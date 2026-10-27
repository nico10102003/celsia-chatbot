"""Limpieza del texto extraído de las fuentes públicas de Celsia."""

import re
import unicodedata

# Espacios horizontales (se conservan los saltos de línea).
HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
# Caracteres de control que no aportan contenido.
CONTROL_CHARACTERS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Líneas compuestas únicamente por separadores o viñetas vacías.
SEPARATOR_LINE = re.compile(r"^[\s\-–—•·*|_=~]+$")


def clean_text(text):
    """Normaliza y limpia un texto extraído de una fuente.

    La limpieza es deliberadamente conservadora: no reescribe el contenido,
    solo homogeneiza espacios, elimina ruido y evita duplicados consecutivos
    que suelen aparecer en los menús de navegación de las páginas.

    Args:
        text: Texto original extraído de la fuente.

    Returns:
        El texto limpio. Devuelve una cadena vacía si ``text`` está vacío.
    """
    if not text:
        return ""

    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    text = CONTROL_CHARACTERS.sub("", text)

    lines = []
    previous_content = None

    for raw_line in text.split("\n"):
        line = HORIZONTAL_WHITESPACE.sub(" ", raw_line).strip()

        if not line or SEPARATOR_LINE.match(line):
            # Conserva un único separador entre bloques de contenido.
            if lines and lines[-1] != "":
                lines.append("")
            continue

        # Algunos menús repiten la misma frase de forma consecutiva.
        if previous_content is not None and line.casefold() == previous_content:
            continue

        lines.append(line)
        previous_content = line.casefold()

    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)
