"""Bout en bout : vrai serveur MCP (Streamable HTTP) en sous-processus + client MCP + graphe.

Le LLM est factice et les embeddings sont `hash` (hors-ligne) ; tout le reste est réel :
transport MCP, agent RAG, ChromaDB, registre SQLite en lecture seule, orchestrateur LangGraph.
"""

from __future__ import annotations

import asyncio
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sav.graph import creer_application, executer

ROOT = Path(__file__).resolve().parents[1]


def _port_libre() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def serveur_mcp(env_demo):
    port = _port_libre()
    env = {**os.environ, **env_demo, "MCP_PORT": str(port), "PYTHONPATH": str(ROOT / "src")}
    proc = subprocess.Popen(
        [sys.executable, "-m", "sav.mcp_server"], env=env, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(100):
            if proc.poll() is not None:
                pytest.fail(f"Le serveur MCP s'est arrêté : {proc.stdout.read().decode()}")
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.2)
        else:
            pytest.fail("Le serveur MCP n'a pas démarré à temps.")
        yield f"http://localhost:{port}/mcp"
    finally:
        proc.terminate()
        proc.wait(timeout=10)


def _lancer(url, question, image=None):
    async def run():
        app = await creer_application(url=url)
        return await executer(app, question, image)

    return asyncio.run(run())


def test_exemple_1_documentation_avec_photo(serveur_mcp):
    r = _lancer(
        serveur_mcp,
        "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme "
        "sur la photo jointe ?",
        "img_px200_01",
    )
    assert r["categorie"] == "documentation" and r["agent"] == "agent_documentaire"
    # l'agent spécialisé rend la main à l'orchestrateur, qui termine
    assert r["trajet"] == ["orchestrateur", "agent_documentaire", "orchestrateur"]
    assert r["outils"] == ["rag_multimodal"]
    assert "bourrage" in r["reponse"].lower()
    assert "PX-200_manuel_utilisateur.pdf" in r["reponse"] and "Figure 1" in r["reponse"]
    # les 3 recherches (passages, figures, photo) tournent en parallèle : aucune ne doit échouer
    assert "Error" not in r["reponse"] and "Erreur" not in r["reponse"]
    assert all(tag in r["reponse"] for tag in ("[P1]", "[F1]"))


def test_exemple_2_garantie(serveur_mcp):
    r = _lancer(
        serveur_mcp,
        "Est-ce que mon imprimante est encore sous garantie ? Son numéro de série est PX200-0042.",
    )
    assert r["categorie"] == "garantie" and r["agent"] == "agent_garantie"
    assert r["trajet"] == ["orchestrateur", "agent_garantie", "orchestrateur"]
    assert r["outils"] == ["consulter_garantie"]
    assert "PX200-0042" in r["reponse"] and '"sous_garantie": true' in r["reponse"]


def test_garantie_sans_numero_de_serie(serveur_mcp):
    r = _lancer(serveur_mcp, "Ma télé est-elle sous garantie ?")
    assert r["categorie"] == "garantie" and r["outils"] == []
    assert "numéro de série" in r["reponse"].lower()


def test_serveur_injoignable_donne_un_message_clair():
    with pytest.raises(ConnectionError, match="python -m sav.mcp_server"):
        asyncio.run(creer_application(url=f"http://localhost:{_port_libre()}/mcp"))
