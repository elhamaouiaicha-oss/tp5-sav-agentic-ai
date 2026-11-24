"""Serveur MCP (Streamable HTTP) qui publie l'outil distant `rag_multimodal`.

    python -m sav.mcp_server          # écoute sur http://127.0.0.1:23000/mcp
"""

from __future__ import annotations

import asyncio

from mcp.server.fastmcp import FastMCP

from . import config
from .rag.rag_agent import repondre_avec_sources

mcp = FastMCP(name="sav-mcp", host=config.mcp_host(), port=config.mcp_port())


@mcp.tool()
async def rag_multimodal(question: str, image_ref: str = "") -> str:
    """Répond à une question de dépannage à partir des manuels (texte + figures), avec sources.

    Args:
        question: la question du client (ex. « Que signifie le voyant orange clignotant ? »).
        image_ref: référence optionnelle de la photo/capture d'écran jointe (ex. « img_px200_01 »).
    """
    # L'agent RAG est synchrone (embeddings + LLM) : on le sort de la boucle d'événements.
    return await asyncio.to_thread(repondre_avec_sources, question, image_ref)


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()
