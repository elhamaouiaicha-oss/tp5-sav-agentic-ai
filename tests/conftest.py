"""Fixtures : environnement 100 % hors-ligne (embeddings `hash`, LLM `fake`, dossiers temporaires)."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))


@pytest.fixture(scope="session")
def env_demo(tmp_path_factory) -> dict[str, str]:
    """Registre + index Chroma construits dans un dossier temporaire à partir de data/."""
    tmp = tmp_path_factory.mktemp("sav")
    env = {
        "EMBEDDINGS_BACKEND": "hash",
        "LLM_MODEL": "fake",
        "CHROMA_DIR": str(tmp / "chroma"),
        "FIGURES_DIR": str(tmp / "figures"),
        "WARRANTY_DB": str(tmp / "garanties.db"),
    }
    anciennes = {k: os.environ.get(k) for k in env}
    os.environ.update(env)

    import build_registry
    from sav.rag.ingest import ingerer_dossier

    build_registry.construire()
    ingerer_dossier()
    yield env
    for k, v in anciennes.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
