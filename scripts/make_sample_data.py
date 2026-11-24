"""Génère des données de démonstration (fictives) : manuel PDF de l'imprimante PX-200 + photo.

    python scripts/make_sample_data.py

Produit :
  data/manuals/PX-200_manuel_utilisateur.pdf   (3 pages, 3 figures légendées)
  data/photos/img_px200_01.png                 (« photo » du panneau, avec bruit)
Nécessite reportlab (extra `dev`). Les fichiers générés sont versionnés : ce script ne sert
qu'à les régénérer.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
W, H = 480, 320


# ------------------------------------------------------------------ figures (dessins simples)


def fig_panneau() -> Image.Image:
    """Panneau de commande : voyant orange clignotant (rayons) + deux voyants éteints."""
    im = Image.new("RGB", (W, H), (235, 235, 235))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((30, 40, W - 30, H - 40), 24, fill=(48, 52, 60))
    # voyant orange clignotant (à gauche) avec rayons
    cx, cy = 120, 160
    d.ellipse((cx - 38, cy - 38, cx + 38, cy + 38), fill=(255, 150, 20))
    for dx, dy in [(-62, 0), (62, 0), (0, -62), (0, 62), (-44, -44), (44, -44), (-44, 44), (44, 44)]:
        d.line((cx + dx * 0.75, cy + dy * 0.75, cx + dx, cy + dy), fill=(255, 190, 80), width=6)
    # pictogramme « feuille » sous le voyant
    d.rectangle((cx - 16, cy + 74, cx + 16, cy + 108), outline=(240, 240, 240), width=3)
    # autres voyants éteints
    d.ellipse((260, 130, 318, 188), fill=(80, 84, 92))
    d.ellipse((350, 130, 408, 188), fill=(80, 84, 92))
    d.rectangle((250, 215, 420, 235), fill=(30, 32, 38))
    return im


def fig_bourrage() -> Image.Image:
    """Bourrage papier : rouleaux et feuille froissée en diagonale."""
    im = Image.new("RGB", (W, H), (250, 250, 250))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 210, W, H), fill=(200, 205, 215))
    for x in (110, 240, 370):
        d.ellipse((x - 34, 176, x + 34, 244), fill=(70, 70, 75))
    d.polygon([(60, 90), (300, 40), (430, 190), (180, 250)], fill=(255, 255, 255), outline=(120, 120, 120))
    d.line((90, 110, 280, 70, 330, 140, 200, 210), fill=(160, 160, 160), width=3)
    d.line((350, 60, 420, 30), fill=(220, 40, 40), width=8)
    d.polygon([(420, 20), (445, 45), (400, 50)], fill=(220, 40, 40))
    return im


def fig_cartouche() -> Image.Image:
    """Remplacement de cartouche : cartouche bleue et flèche d'insertion."""
    im = Image.new("RGB", (W, H), (245, 248, 252))
    d = ImageDraw.Draw(im)
    d.rectangle((60, 190, W - 60, 290), fill=(190, 195, 205))
    d.rounded_rectangle((170, 60, 310, 200), 10, fill=(30, 90, 200))
    d.rectangle((200, 30, 280, 62), fill=(20, 60, 140))
    d.rectangle((190, 100, 290, 150), fill=(255, 255, 255))
    d.polygon([(380, 120), (440, 160), (380, 200)], fill=(40, 160, 70))
    d.rectangle((330, 148, 385, 172), fill=(40, 160, 70))
    return im


def photo_du_panneau() -> Image.Image:
    """Simule une photo : même scène que la Figure 1, recadrée, bruitée, floutée, plus sombre."""
    rnd = random.Random(7)
    im = fig_panneau().crop((8, 6, W - 6, H - 10)).resize((W, H))
    im = im.filter(ImageFilter.GaussianBlur(1.4))
    px = im.load()
    for y in range(H):
        for x in range(W):
            r, g, b = px[x, y]
            n = rnd.randint(-14, 14)
            px[x, y] = tuple(max(0, min(255, int(c * 0.88) + n)) for c in (r, g, b))
    return im


# ------------------------------------------------------------------ PDF


def _png(im: Image.Image) -> io.BytesIO:
    buf = io.BytesIO()
    im.save(buf, "PNG")
    buf.seek(0)
    return buf


def build_pdf(chemin: Path) -> None:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image as RLImage
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

    s = getSampleStyleSheet()
    legende = ParagraphStyle("legende", parent=s["Italic"], fontSize=9, spaceAfter=14)
    corps = ParagraphStyle("corps", parent=s["BodyText"], fontSize=11, leading=15, spaceAfter=8)

    def figure(im: Image.Image, texte: str):
        return [RLImage(_png(im), width=9 * cm, height=6 * cm), Paragraph(texte, legende)]

    el = [
        Paragraph("Imprimante PX-200 — Manuel utilisateur", s["Title"]),
        Paragraph("1. Panneau de commande et voyants", s["Heading2"]),
        Paragraph(
            "Le panneau de commande de l'imprimante PX-200 comporte trois voyants : le voyant "
            "d'alimentation (vert), le voyant papier (orange) et le voyant d'encre (orange). "
            "Un voyant éteint signifie que la fonction correspondante ne signale aucun problème.",
            corps,
        ),
        *figure(fig_panneau(), "Figure 1 – Panneau de commande de la PX-200 : voyant papier orange clignotant."),
        Paragraph(
            "Appuyez sur la touche Annuler pendant trois secondes pour réinitialiser l'imprimante "
            "si les voyants clignotent de façon inattendue après une mise à jour.",
            corps,
        ),
        PageBreak(),
        Paragraph("2. Signification des voyants", s["Heading2"]),
        Paragraph(
            "Voyant orange clignotant (papier) : bourrage papier ou absence de papier dans le bac. "
            "Ouvrez le capot arrière, retirez délicatement la feuille coincée dans les rouleaux, "
            "refermez le capot puis appuyez sur la touche Reprise.",
            corps,
        ),
        Paragraph(
            "Voyant orange fixe (encre) : niveau d'encre bas. L'impression reste possible, mais "
            "préparez une cartouche de remplacement. Voyant vert clignotant : impression en cours "
            "ou démarrage de l'appareil.",
            corps,
        ),
        *figure(fig_bourrage(), "Figure 2 – Bourrage papier : accès aux rouleaux par le capot arrière."),
        Paragraph(
            "Code d'erreur E-12 : chariot d'impression bloqué. Éteignez l'imprimante, retirez tout "
            "objet qui gêne le chariot et rallumez-la. Si l'erreur persiste, contactez le service "
            "après-vente.",
            corps,
        ),
        PageBreak(),
        Paragraph("3. Remplacement de la cartouche d'encre", s["Heading2"]),
        Paragraph(
            "Ouvrez le capot avant ; le chariot se place au centre. Appuyez sur la cartouche usagée "
            "pour la libérer, retirez-la, puis insérez la nouvelle cartouche jusqu'au clic. Refermez "
            "le capot : l'imprimante effectue un cycle d'amorçage d'environ deux minutes.",
            corps,
        ),
        *figure(fig_cartouche(), "Figure 3 – Remplacement de la cartouche d'encre de la PX-200."),
        Spacer(1, 6),
        Paragraph(
            "Utilisez uniquement des cartouches d'origine référencées PX-INK-200 afin de conserver "
            "la garantie constructeur.",
            corps,
        ),
    ]
    chemin.parent.mkdir(parents=True, exist_ok=True)
    SimpleDocTemplate(str(chemin), pagesize=A4, title="PX-200 Manuel utilisateur").build(el)


def main() -> None:
    build_pdf(ROOT / "data" / "manuals" / "PX-200_manuel_utilisateur.pdf")
    photos = ROOT / "data" / "photos"
    photos.mkdir(parents=True, exist_ok=True)
    photo_du_panneau().save(photos / "img_px200_01.png")
    print("Données d'exemple générées dans data/manuals et data/photos.")


if __name__ == "__main__":
    main()
