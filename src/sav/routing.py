"""Classification d'une demande : `documentation` (dépannage) ou `garantie`."""

from __future__ import annotations

import re
from typing import Literal

Categorie = Literal["documentation", "garantie"]

# Numéro de série type « PX200-0042 » (lettres + chiffres, tiret, chiffres).
# Le nom de modèle « PX-200 » ne correspond volontairement pas.
SERIAL_RE = re.compile(r"\b[A-Z]{2,4}\d{2,4}-\d{3,6}\b")

_MOTS_GARANTIE = ("garantie", "sous garantie", "numéro de série", "numero de serie")


def extraire_numero_serie(texte: str) -> str | None:
    m = SERIAL_RE.search(texte.upper())
    return m.group(0) if m else None


def heuristique_categorie(question: str) -> Categorie:
    """Règle de secours déterministe, utilisée si le LLM ne répond pas clairement."""
    q = question.lower()
    if extraire_numero_serie(question) or any(mot in q for mot in _MOTS_GARANTIE):
        return "garantie"
    return "documentation"


def parse_categorie(reponse_llm: str, question: str) -> Categorie:
    """Interprète la réponse du LLM ; en cas d'ambiguïté, bascule sur l'heuristique."""
    r = reponse_llm.lower()
    a_garantie, a_doc = "garantie" in r, "documentation" in r
    if a_garantie and not a_doc:
        return "garantie"
    if a_doc and not a_garantie:
        return "documentation"
    return heuristique_categorie(question)
