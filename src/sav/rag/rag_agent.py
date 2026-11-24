"""Agent RAG multimodal : cherche des passages et des figures, puis produit une réponse SOURCÉE.

Cet agent vit côté serveur MCP ; il est exposé au reste du système par l'outil `rag_multimodal`.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from ..llm import get_llm
from ..react import build_react_agent
from . import search

SYSTEM_PROMPT = """Tu es l'agent RAG documentaire du service après-vente d'un distributeur d'appareils
électroniques. Tu réponds en français.

Méthode :
1. Appelle `rechercher_passages` avec une requête courte sur le problème décrit.
2. Appelle `rechercher_figures` pour retrouver les schémas/pictogrammes utiles.
3. Si la demande mentionne une photo (image_ref=...), appelle aussi `rechercher_figures_similaires`
   avec exactement cette référence.
4. Rédige la réponse UNIQUEMENT à partir des éléments retournés par les outils.

Règles :
- Cite chaque information avec sa source au format (manuel, page) et mentionne les figures
  pertinentes par leur légende.
- Si les éléments retrouvés ne permettent pas de répondre, dis-le clairement : n'invente rien.
- Termine par une ligne « Sources : » listant les documents, pages et figures utilisés."""


def _fmt_passages(items: list[dict]) -> str:
    if not items:
        return "Aucun passage trouvé."
    return "\n".join(
        f"[P{i}] ({it['source']}, p.{it['page']}, pertinence {it['score']}) {it['texte']}"
        for i, it in enumerate(items, 1)
    )


def _fmt_figures(items: list[dict]) -> str:
    if not items:
        return "Aucune figure trouvée."
    return "\n".join(
        f"[F{i}] ({it['source']}, p.{it['page']}, pertinence {it['score']}) "
        f"{it.get('legende') or 'figure sans légende'} — fichier : {it['fichier']}"
        for i, it in enumerate(items, 1)
    )


@tool
def rechercher_passages(requete: str) -> str:
    """Recherche des passages de texte pertinents dans les manuels (retourne source et page)."""
    return _fmt_passages(search.rechercher_passages(requete))


@tool
def rechercher_figures(requete: str) -> str:
    """Recherche des figures/schémas des manuels à partir d'une description en texte."""
    return _fmt_figures(search.rechercher_figures_par_texte(requete))


@tool
def rechercher_figures_similaires(image_ref: str) -> str:
    """Recherche les figures des manuels visuellement proches d'une photo jointe.

    Args:
        image_ref: référence de la photo, par exemple « img_px200_01 ».
    """
    try:
        return _fmt_figures(search.rechercher_figures_par_image(image_ref))
    except (ValueError, FileNotFoundError) as e:
        return f"Erreur : {e}"


@lru_cache(maxsize=1)
def get_rag_agent():
    return build_react_agent(
        get_llm(),
        [rechercher_passages, rechercher_figures, rechercher_figures_similaires],
        SYSTEM_PROMPT,
        name="agent_rag",
    )


def repondre_avec_sources(question: str, image_ref: str = "") -> str:
    """Point d'entrée synchrone appelé par l'outil MCP `rag_multimodal`."""
    contenu = question.strip()
    if image_ref:
        contenu += f"\n\n[Photo jointe : image_ref={image_ref}]"
    resultat = get_rag_agent().invoke(
        {"messages": [HumanMessage(contenu)]}, config={"recursion_limit": 12}
    )
    msg = resultat["messages"][-1].content
    return msg if isinstance(msg, str) else " ".join(b.get("text", "") for b in msg if isinstance(b, dict))
