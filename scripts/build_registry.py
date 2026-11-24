"""Construit le registre SQLite des garanties à partir de data/garanties.csv.

    python scripts/build_registry.py

Ce script est le SEUL endroit qui écrit dans la base. L'outil `consulter_garantie`, lui,
l'ouvre en lecture seule (mode=ro).
"""

from __future__ import annotations

import csv
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sav import config  # noqa: E402


def construire() -> int:
    db = config.warranty_db()
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    with open(config.warranty_csv(), newline="", encoding="utf-8") as f:
        lignes = list(csv.DictReader(f))
    conn = sqlite3.connect(db)
    with conn:
        conn.execute(
            """CREATE TABLE garanties (
                   numero_serie  TEXT PRIMARY KEY,
                   modele        TEXT NOT NULL,
                   date_achat    TEXT NOT NULL,
                   duree_mois    INTEGER NOT NULL,
                   type_garantie TEXT NOT NULL
               )"""
        )
        conn.executemany(
            "INSERT INTO garanties VALUES (:numero_serie, :modele, :date_achat, :duree_mois, :type_garantie)",
            lignes,
        )
    conn.close()
    return len(lignes)


if __name__ == "__main__":
    print(f"Registre construit : {construire()} appareils -> {config.warranty_db()}")
