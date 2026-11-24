"""Recherche dans ChromaDB : passages (texte) et figures (texte->image ou photo->image)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from PIL import Image

from .. import config
from .embeddings import Embedder, get_embedder
from .store import collections

_REF_PHOTO = re.compile(r"^[\w\-]+(\.(png|jpe?g|webp))?$", re.IGNORECASE)
_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def resoudre_photo(image_ref: str) -> Path:
    """Transforme une référence (ex. `img_px200_01`) en fichier de data/photos.

    Seuls les noms simples sont acceptés (pas de séparateur de chemin ni de `..`) : l'agent ne
    peut donc pas être amené à lire un fichier arbitraire du disque.
    """
    ref = (image_ref or "").strip()
    if not _REF_PHOTO.match(ref):
        raise ValueError(f"Référence de photo invalide : {image_ref!r}")
    base = config.photos_dir()
    candidats = [base / ref] if Path(ref).suffix.lower() in _EXTENSIONS else [base / f"{ref}{e}" for e in _EXTENSIONS]
    for c in candidats:
        if c.is_file():
            return c
    raise FileNotFoundError(f"Photo introuvable pour la référence {image_ref!r}")


def _interroger(col: Any, vecteur: list[float], k: int) -> list[dict[str, Any]]:
    n = col.count()
    if n == 0:
        return []
    res = col.query(
        query_embeddings=[vecteur],
        n_results=min(k, n),
        include=["documents", "metadatas", "distances"],
    )
    sortie = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        sortie.append({"texte": doc, **meta, "score": round(1.0 - float(dist), 3)})
    return sortie


def rechercher_passages(requete: str, k: int = 4, embedder: Embedder | None = None) -> list[dict]:
    embedder = embedder or get_embedder()
    col_texte, _ = collections()
    return _interroger(col_texte, embedder.embed_texts([requete])[0], k)


def rechercher_figures_par_texte(requete: str, k: int = 3, embedder: Embedder | None = None) -> list[dict]:
    embedder = embedder or get_embedder()
    _, col_figs = collections()
    return _interroger(col_figs, embedder.embed_texts([requete])[0], k)


def rechercher_figures_par_image(image_ref: str, k: int = 3, embedder: Embedder | None = None) -> list[dict]:
    embedder = embedder or get_embedder()
    _, col_figs = collections()
    with Image.open(resoudre_photo(image_ref)) as im:
        vecteur = embedder.embed_images([im.convert("RGB")])[0]
    return _interroger(col_figs, vecteur, k)
