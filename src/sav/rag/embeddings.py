"""Plongements texte + image.

* `clip`  : CLIP (sentence-transformers). Images avec `clip-ViT-B-32`, textes avec la variante
            multilingue `clip-ViT-B-32-multilingual-v1`, alignée sur le même espace -> une requête
            en français peut retrouver une figure, et une photo peut retrouver une figure proche.
* `hash`  : plongements déterministes sans modèle (sac de mots haché / vignette 16x16).
            Uniquement pour les tests hors-ligne : ils ne comprennent pas la sémantique.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np
from PIL import Image

from .. import config


class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...
    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]: ...


def _normaliser(v: np.ndarray) -> list[float]:
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(float).tolist()


class HashEmbedder:
    DIM = 256

    @staticmethod
    def _tokens(texte: str) -> list[str]:
        sans_accents = unicodedata.normalize("NFKD", texte.lower())
        sans_accents = "".join(c for c in sans_accents if not unicodedata.combining(c))
        return re.findall(r"[a-z0-9]+", sans_accents)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = np.zeros(self.DIM)
            for tok in self._tokens(t):
                v[int(hashlib.sha1(tok.encode()).hexdigest()[:8], 16) % self.DIM] += 1.0
            out.append(_normaliser(v))
        return out

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        out = []
        for im in images:
            v = np.asarray(im.convert("L").resize((16, 16)), dtype=float).ravel()
            out.append(_normaliser(v - v.mean()))
        return out


class ClipEmbedder:
    def __init__(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers est requis pour EMBEDDINGS_BACKEND=clip "
                "(pip install sentence-transformers)."
            ) from e
        self._img = SentenceTransformer("clip-ViT-B-32")
        self._txt = SentenceTransformer("sentence-transformers/clip-ViT-B-32-multilingual-v1")

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        return self._txt.encode(list(texts), normalize_embeddings=True).tolist()

    def embed_images(self, images: Sequence[Image.Image]) -> list[list[float]]:
        return self._img.encode(list(images), normalize_embeddings=True).tolist()


@lru_cache(maxsize=None)
def _creer(backend: str) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "clip":
        return ClipEmbedder()
    raise ValueError(f"EMBEDDINGS_BACKEND inconnu : {backend!r} (attendu : clip | hash)")


def get_embedder() -> Embedder:
    return _creer(config.embeddings_backend())
