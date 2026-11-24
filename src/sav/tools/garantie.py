"""Outil local `consulter_garantie` : interroge le registre des garanties en LECTURE SEULE."""

from __future__ import annotations

import re
import sqlite3
from datetime import date, datetime
from typing import Any

from langchain_core.tools import tool

from .. import config

_FORMAT_SERIE = re.compile(r"^[A-Z]{2,4}\d{2,4}-\d{3,6}$")


def _connexion_lecture_seule() -> sqlite3.Connection:
    """Ouvre la base en mode `ro` : toute écriture lève `sqlite3.OperationalError`."""
    chemin = config.warranty_db().resolve()
    if not chemin.exists():
        raise FileNotFoundError(
            f"Registre introuvable : {chemin}. Lancez `python scripts/setup_demo.py`."
        )
    conn = sqlite3.connect(f"{chemin.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def ajouter_mois(d: date, mois: int) -> date:
    """d + `mois` mois (le jour est ramené à la fin du mois si nécessaire)."""
    total = d.year * 12 + (d.month - 1) + mois
    annee, m0 = divmod(total, 12)
    mois_cible = m0 + 1
    jours = [31, 29 if annee % 4 == 0 and (annee % 100 or annee % 400 == 0) else 28,
             31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    return date(annee, mois_cible, min(d.day, jours[m0]))


def verifier_garantie(numero_serie: str, today: date | None = None) -> dict[str, Any]:
    """Logique pure (testable) derrière l'outil."""
    today = today or date.today()
    numero = (numero_serie or "").strip().upper()
    if not _FORMAT_SERIE.match(numero):
        return {
            "trouve": False,
            "numero_serie": numero,
            "message": "Format de numéro de série invalide (attendu : ex. PX200-0042).",
        }
    with _connexion_lecture_seule() as conn:
        row = conn.execute(
            "SELECT numero_serie, modele, date_achat, duree_mois, type_garantie "
            "FROM garanties WHERE numero_serie = ?",
            (numero,),
        ).fetchone()
    if row is None:
        return {
            "trouve": False,
            "numero_serie": numero,
            "message": "Numéro de série absent du registre.",
        }
    achat = datetime.strptime(row["date_achat"], "%Y-%m-%d").date()
    fin = ajouter_mois(achat, row["duree_mois"])
    restants = (fin - today).days
    return {
        "trouve": True,
        "numero_serie": row["numero_serie"],
        "modele": row["modele"],
        "date_achat": row["date_achat"],
        "type_garantie": row["type_garantie"],
        "duree_mois": row["duree_mois"],
        "date_fin_garantie": fin.isoformat(),
        "sous_garantie": restants >= 0,
        "jours_restants": max(restants, 0),
        "date_verification": today.isoformat(),
    }


@tool
def consulter_garantie(numero_serie: str) -> dict:
    """Consulte le registre (lecture seule) et indique si l'appareil est sous garantie.

    Args:
        numero_serie: numéro de série de l'appareil, par exemple « PX200-0042 ».

    Returns:
        Un dictionnaire : trouve, modele, date_achat, date_fin_garantie,
        sous_garantie (booléen), jours_restants… ou un message d'erreur.
    """
    return verifier_garantie(numero_serie)
