"""CLI du SAV agentique.

    python -m sav.main "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le
                        pictogramme sur la photo jointe ?" --image img_px200_01
    python -m sav.main "Est-ce que mon imprimante est encore sous garantie ? Série PX200-0042."
    python -m sav.main                 # mode interactif
    python -m sav.main --mermaid       # affiche le graphe LangGraph au format Mermaid
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from langchain_core.tools import StructuredTool

from .graph import build_graph, creer_application, executer


async def _traiter(app, question: str, image: str | None) -> None:
    r = await executer(app, question, image)
    print(f"\n[trajet]   {' -> '.join(r['trajet'])}")
    print(f"[catégorie] {r['categorie']}   [agent] {r['agent']}   [outils] {', '.join(r['outils']) or '-'}")
    print(f"\n{r['reponse']}\n")


def _graphe_mermaid() -> str:
    """Dessin du graphe : n'a pas besoin du serveur MCP (outil distant remplacé par un bouchon)."""
    bouchon = StructuredTool.from_function(
        func=lambda question, image_ref="": "", name="rag_multimodal", description="(bouchon)"
    )
    from .fake_llm import FakeLLM  # le dessin ne nécessite ni clé API ni modèle

    return build_graph(FakeLLM(), outils_doc=[bouchon]).get_graph().draw_mermaid()


async def _main(args: argparse.Namespace) -> int:
    if args.mermaid:
        print(_graphe_mermaid())
        return 0
    try:
        app = await creer_application()
    except Exception as e:  # noqa: BLE001 - MCP injoignable, clé API manquante, etc.
        print(f"Erreur : {e}", file=sys.stderr)
        return 1

    if args.question:
        await _traiter(app, args.question, args.image)
        return 0

    print("SAV agentique — tapez 'exit' pour quitter. Photo : préfixez par 'img_xxx: '.")
    while True:
        try:
            ligne = input("Question : ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not ligne or ligne.lower() == "exit":
            break
        image = None
        if ligne.startswith("img_") and ":" in ligne:
            image, ligne = (s.strip() for s in ligne.split(":", 1))
        await _traiter(app, ligne, image)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="SAV agentique (LangGraph + MCP + RAG multimodal)")
    ap.add_argument("question", nargs="?", help="Question du client")
    ap.add_argument("--image", help="Référence de la photo jointe (ex. img_px200_01)")
    ap.add_argument("--mermaid", action="store_true", help="Affiche le graphe au format Mermaid")
    sys.exit(asyncio.run(_main(ap.parse_args())))


if __name__ == "__main__":
    main()
