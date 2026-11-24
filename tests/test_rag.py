import pytest

from sav.rag import search
from sav.rag.ingest import decouper, ingerer_dossier


def test_indexation_idempotente(env_demo):
    from sav.rag.store import collections

    texte, figs = collections()
    n_t, n_f = texte.count(), figs.count()
    assert n_f == 3 and n_t >= 3
    ingerer_dossier()  # une 2e indexation ne doit rien dupliquer
    assert (texte.count(), figs.count()) == (n_t, n_f)


def test_passage_voyant_orange(env_demo):
    top = search.rechercher_passages("voyant orange clignotant")[0]
    assert top["page"] == 2 and "bourrage" in top["texte"].lower()
    assert top["source"].endswith(".pdf")


def test_figures_ont_une_legende(env_demo):
    figs = search.rechercher_figures_par_image("img_px200_01", k=3)
    assert len(figs) == 3 and all(f["legende"].startswith("Figure") for f in figs)


def test_photo_retrouve_la_figure_du_panneau(env_demo):
    top = search.rechercher_figures_par_image("img_px200_01")[0]
    assert "Panneau de commande" in top["legende"] and top["page"] == 1


@pytest.mark.parametrize("ref", ["../../etc/passwd", "a/b", "..", "", "photo.exe", "inconnue"])
def test_reference_photo_refusee(env_demo, ref):
    with pytest.raises((ValueError, FileNotFoundError)):
        search.resoudre_photo(ref)


def test_decouper_respecte_la_taille():
    morceaux = decouper(["Phrase un. " * 60], max_chars=100)
    assert morceaux and all(len(m) <= 130 for m in morceaux)


_SCRIPT_PARALLELE = """
from concurrent.futures import ThreadPoolExecutor
from sav.rag import search
appels = (
    lambda: search.rechercher_passages("voyant"),
    lambda: search.rechercher_figures_par_texte("voyant"),
    lambda: search.rechercher_figures_par_image("img_px200_01"),
)
with ThreadPoolExecutor(max_workers=3) as pool:   # comme ToolNode : 3 outils en parallèle
    resultats = list(pool.map(lambda f: f(), appels))
assert all(resultats), resultats
"""


def test_premieres_recherches_en_parallele_dans_un_processus_neuf(env_demo):
    """Régression : sans client Chroma partagé/verrouillé, ces appels simultanés échouent
    (« Could not connect to tenant default_tenant ») lors de la 1re utilisation du processus."""
    import os
    import subprocess
    import sys
    from pathlib import Path

    racine = Path(__file__).resolve().parents[1]
    env = {**os.environ, **env_demo, "PYTHONPATH": str(racine / "src")}
    for _ in range(3):  # la course est quasi systématique sans correctif ; 3 essais pour être sûr
        r = subprocess.run([sys.executable, "-c", _SCRIPT_PARALLELE], env=env, cwd=racine,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, r.stderr[-500:]
