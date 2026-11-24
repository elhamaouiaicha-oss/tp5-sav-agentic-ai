"""Accès à ChromaDB : une collection pour les passages, une pour les figures."""

from __future__ import annotations

import threading
from typing import Any

import chromadb

from .. import config

# Un seul client par dossier, créé sous verrou : les outils de recherche sont appelés EN PARALLÈLE
# par ToolNode (appels d'outils multiples d'un même message du LLM). Créer plusieurs
# PersistentClient simultanément fait échouer Chroma (« Could not connect to tenant default_tenant »).
_verrou = threading.RLock()
_clients: dict[str, chromadb.ClientAPI] = {}


def _client() -> chromadb.ClientAPI:
    chemin = config.chroma_dir()
    cle = str(chemin)
    with _verrou:
        if cle not in _clients:
            chemin.mkdir(parents=True, exist_ok=True)
            _clients[cle] = chromadb.PersistentClient(path=cle)
        return _clients[cle]


def collections() -> tuple[Any, Any]:
    """(passages, figures). Le nom inclut le backend : les dimensions des vecteurs diffèrent."""
    suffixe = config.embeddings_backend()
    meta = {"hnsw:space": "cosine"}
    with _verrou:
        client = _client()
        return (
            client.get_or_create_collection(f"manuels_texte_{suffixe}", metadata=meta),
            client.get_or_create_collection(f"manuels_figures_{suffixe}", metadata=meta),
        )
