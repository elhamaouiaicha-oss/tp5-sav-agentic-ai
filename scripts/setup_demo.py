"""Prépare tout le nécessaire avant de lancer le serveur MCP et le CLI.

    python scripts/setup_demo.py

1. construit le registre des garanties (SQLite) ;
2. indexe les manuels PDF de data/manuals dans ChromaDB (texte + figures).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import build_registry  # noqa: E402
from sav import config  # noqa: E402
from sav.rag.ingest import ingerer_dossier  # noqa: E402


def main() -> None:
    n = build_registry.construire()
    print(f"[1/2] Registre des garanties : {n} appareils -> {config.warranty_db()}")
    print(f"[2/2] Indexation des manuels ({config.embeddings_backend()}) dans {config.chroma_dir()} ...")
    print("      ", ingerer_dossier())


if __name__ == "__main__":
    main()
