"""BM25 simple para la fusión con la recuperación vectorial.

Se usa como segundo ranking en la estrategia híbrida del ``Retriever``. Es una
implementación ligera de BM25 (Okapi) sin dependencias externas.
"""

import math
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Tokeniza español sin tildes ni puntuación, en minúsculas."""
    normalized = unicodedata.normalize("NFKD", text.lower())
    normalized = "".join(character for character in normalized if not unicodedata.combining(character))
    return _TOKEN_PATTERN.findall(normalized)


class BM25Index:
    """Índice BM25 (Okapi) sobre una colección de documentos."""

    def __init__(self, documents: Sequence[str], k1: float = 1.5, b: float = 0.75):
        """Configura el índice sobre los documentos indicados.

        Args:
            documents: Lista de textos sobre los que se calcularán las similitudes.
            k1: Constante de saturación del término.
            b: Normalización del largo del documento.
        """
        self.documents = [document if isinstance(document, str) else document or "" for document in documents]
        self.k1 = k1
        self.b = b
        self._document_count = len(self.documents)
        self._doc_lengths = [len(tokenize(document)) for document in self.documents]
        self._avg_length = (
            sum(self._doc_lengths) / len(self._doc_lengths) if self._doc_lengths else 0.0
        )
        self._term_frequencies = [Counter(tokenize(document)) for document in self.documents]

        document_frequency: Counter = Counter()
        for frequencies in self._term_frequencies:
            document_frequency.update(set(frequencies.keys()))

        self._idf = {
            term: math.log(1 + (self._document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def score(self, query: str, index: int) -> float:
        """Devuelve la puntuación BM25 del documento ``index`` para ``query``."""
        terms = tokenize(query)
        if not terms or self._document_count == 0 or self._avg_length == 0:
            return 0.0

        document_length = self._doc_lengths[index]
        norm = self.k1 * (1 - self.b + self.b * (document_length / self._avg_length))
        score = 0.0

        for term in set(terms):
            frequency = self._term_frequencies[index].get(term, 0)
            if frequency == 0:
                continue

            idf = self._idf.get(term, 0.0)
            score += idf * (frequency * (self.k1 + 1)) / (frequency + self.k1 * norm)

        return score

    def scores(self, query: str) -> list[float]:
        """Devuelve la puntuación BM25 para cada documento indexado."""
        return [self.score(query, index) for index in range(self._document_count)]
