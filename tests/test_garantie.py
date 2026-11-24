import sqlite3
from datetime import date

import pytest

from sav import config
from sav.tools.garantie import ajouter_mois, consulter_garantie, verifier_garantie


def test_sous_garantie(env_demo):
    r = verifier_garantie("PX200-0042", today=date(2026, 11, 21))
    assert r["trouve"] and r["sous_garantie"] is True
    assert r["date_fin_garantie"] == "2031-03-12"
    assert r["jours_restants"] > 0


def test_garantie_expiree(env_demo):
    r = verifier_garantie("PX200-0007", today=date(2026, 11, 21))
    assert r["trouve"] and r["sous_garantie"] is False
    assert r["date_fin_garantie"] == "2022-06-01"
    assert r["jours_restants"] == 0


def test_numero_inconnu_ou_invalide(env_demo):
    assert verifier_garantie("PX200-9999")["trouve"] is False
    r = verifier_garantie("x' OR '1'='1")
    assert r["trouve"] is False and "invalide" in r["message"]


def test_normalisation_casse_et_espaces(env_demo):
    assert verifier_garantie("  px200-0042 ")["trouve"] is True


def test_outil_langchain(env_demo):
    r = consulter_garantie.invoke({"numero_serie": "PX200-0042"})
    assert r["modele"] == "Imprimante PX-200"


def test_registre_en_lecture_seule(env_demo):
    chemin = config.warranty_db().resolve()
    conn = sqlite3.connect(f"{chemin.as_uri()}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM garanties")
    conn.close()


def test_ajouter_mois_fin_de_mois():
    assert ajouter_mois(date(2024, 1, 31), 1) == date(2024, 2, 29)
    assert ajouter_mois(date(2025, 3, 12), 24) == date(2027, 3, 12)
