"""Configuration centralisée (lue à l'appel, donc surchargeable par l'environnement / les tests)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]

# Charge .env à la racine du dépôt (sans écraser les variables déjà définies).
load_dotenv(ROOT / ".env")


def _path(var: str, default: Path) -> Path:
    return Path(os.getenv(var, str(default))).expanduser()


def llm_model() -> str:
    """Identifiant `fournisseur:modèle` (ex. openai:gpt-4o-mini, ollama:qwen2.5:7b) ou `fake`."""
    return os.getenv("LLM_MODEL", "openai:gpt-4o-mini")


def embeddings_backend() -> str:
    """`clip` (CLIP multilingue, sentence-transformers) ou `hash` (hors-ligne, tests)."""
    return os.getenv("EMBEDDINGS_BACKEND", "clip")


def chroma_dir() -> Path:
    return _path("CHROMA_DIR", ROOT / "data" / "chroma")


def manuals_dir() -> Path:
    return _path("MANUALS_DIR", ROOT / "data" / "manuals")


def photos_dir() -> Path:
    return _path("PHOTOS_DIR", ROOT / "data" / "photos")


def figures_dir() -> Path:
    """Dossier où les figures extraites des PDF sont enregistrées."""
    return _path("FIGURES_DIR", ROOT / "data" / "figures")


def warranty_db() -> Path:
    return _path("WARRANTY_DB", ROOT / "data" / "garanties.db")


def warranty_csv() -> Path:
    return _path("WARRANTY_CSV", ROOT / "data" / "garanties.csv")


def mcp_host() -> str:
    return os.getenv("MCP_HOST", "127.0.0.1")


def mcp_port() -> int:
    return int(os.getenv("MCP_PORT", "23000"))


def mcp_url() -> str:
    return os.getenv("MCP_URL", f"http://localhost:{mcp_port()}/mcp")
