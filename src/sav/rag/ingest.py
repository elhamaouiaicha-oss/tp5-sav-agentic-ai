"""Indexation des manuels PDF : texte (passages) et images (figures) -> ChromaDB.

    python -m sav.rag.ingest                # indexe data/manuals/*.pdf
    python -m sav.rag.ingest un_manuel.pdf  # indexe un fichier précis

L'indexation est idempotente : les identifiants sont des SHA-256 du contenu (upsert).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import re
from pathlib import Path

import pymupdf as fitz  # PyMuPDF (>= 1.24.3)
from PIL import Image

from .. import config
from .embeddings import Embedder, get_embedder
from .store import collections

_TAILLE_MIN_FIGURE = 60  # px : ignore logos, puces, filets…
_CAPTION_RE = re.compile(r"^\s*(figure|fig\.)\s*\d+", re.IGNORECASE)


def _sha(*parts: str | bytes) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p if isinstance(p, bytes) else p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def decouper(blocs: list[str], max_chars: int = 450) -> list[str]:
    """Regroupe des paragraphes consécutifs jusqu'à `max_chars` ; coupe les très longs par phrases."""
    morceaux: list[str] = []
    courant = ""
    for bloc in blocs:
        phrases = [bloc] if len(bloc) <= max_chars else re.split(r"(?<=[.!?])\s+", bloc)
        for ph in phrases:
            if courant and len(courant) + len(ph) + 1 > max_chars:
                morceaux.append(courant)
                courant = ""
            courant = f"{courant} {ph}".strip()
    if courant:
        morceaux.append(courant)
    return morceaux


def _blocs_texte(page: fitz.Page) -> list[tuple[float, float, str]]:
    blocs = []
    for x0, y0, x1, y1, txt, _no, kind in page.get_text("blocks"):
        if kind == 0 and txt.strip():
            blocs.append((y0, y1, " ".join(txt.split())))
    return sorted(blocs)


def _legende_proche(blocs: list[tuple[float, float, str]], rect: fitz.Rect | None) -> str:
    legendes = [(y0, y1, t) for y0, y1, t in blocs if _CAPTION_RE.match(t)]
    if not legendes:
        return ""
    if rect is None:
        return legendes[0][2]
    return min(legendes, key=lambda b: abs(b[0] - rect.y1))[2]


def ingerer_pdf(pdf: Path, embedder: Embedder | None = None) -> dict[str, int]:
    embedder = embedder or get_embedder()
    col_texte, col_figs = collections()
    figures_dir = config.figures_dir()
    figures_dir.mkdir(parents=True, exist_ok=True)

    ids_t, docs_t, metas_t = [], [], []
    ids_f, imgs_f, metas_f = [], [], []

    with fitz.open(pdf) as doc:
        for numero, page in enumerate(doc, start=1):
            blocs = _blocs_texte(page)
            for morceau in decouper([t for _, _, t in blocs]):
                ids_t.append(_sha(pdf.name, str(numero), morceau))
                docs_t.append(morceau)
                metas_t.append({"source": pdf.name, "page": numero})

            for idx, info in enumerate(page.get_images(full=True)):
                xref = info[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.n - pix.alpha >= 4:  # CMYK -> RGB
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                if min(pix.width, pix.height) < _TAILLE_MIN_FIGURE:
                    continue
                png = pix.tobytes("png")
                chemin = figures_dir / f"{pdf.stem}_p{numero}_{idx}.png"
                chemin.write_bytes(png)
                rects = page.get_image_rects(xref)
                legende = _legende_proche(blocs, rects[0] if rects else None)
                ids_f.append(_sha(pdf.name, str(numero), png))
                imgs_f.append(Image.open(io.BytesIO(png)).convert("RGB"))
                metas_f.append(
                    {
                        "source": pdf.name,
                        "page": numero,
                        "legende": legende,
                        "fichier": str(chemin),
                    }
                )

    if ids_t:
        col_texte.upsert(
            ids=ids_t,
            documents=docs_t,
            metadatas=metas_t,
            embeddings=embedder.embed_texts(docs_t),
        )
    if ids_f:
        col_figs.upsert(
            ids=ids_f,
            documents=[m["legende"] or m["fichier"] for m in metas_f],
            metadatas=metas_f,
            embeddings=embedder.embed_images(imgs_f),
        )
    return {"passages": len(ids_t), "figures": len(ids_f)}


def ingerer_dossier(dossier: Path | None = None) -> dict[str, int]:
    dossier = dossier or config.manuals_dir()
    total = {"pdf": 0, "passages": 0, "figures": 0}
    for pdf in sorted(dossier.glob("*.pdf")):
        r = ingerer_pdf(pdf)
        total["pdf"] += 1
        total["passages"] += r["passages"]
        total["figures"] += r["figures"]
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description="Indexe des manuels PDF dans ChromaDB.")
    ap.add_argument("pdf", nargs="*", type=Path, help="PDF à indexer (défaut : data/manuals/*.pdf)")
    args = ap.parse_args()
    if args.pdf:
        total = {"pdf": 0, "passages": 0, "figures": 0}
        for p in args.pdf:
            r = ingerer_pdf(p)
            total["pdf"] += 1
            total["passages"] += r["passages"]
            total["figures"] += r["figures"]
    else:
        total = ingerer_dossier()
    print(f"Indexation terminée : {total}")


if __name__ == "__main__":
    main()
