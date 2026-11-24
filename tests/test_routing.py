import pytest

from sav.routing import extraire_numero_serie, heuristique_categorie, parse_categorie

Q_DOC = "Mon imprimante PX-200 affiche un voyant clignotant. Que signifie le pictogramme sur la photo jointe ?"
Q_GAR = "Est-ce que mon imprimante est encore sous garantie ? Son numéro de série est PX200-0042."


def test_exemples_du_sujet():
    assert heuristique_categorie(Q_DOC) == "documentation"
    assert heuristique_categorie(Q_GAR) == "garantie"


def test_modele_n_est_pas_un_numero_de_serie():
    assert extraire_numero_serie("imprimante PX-200") is None
    assert extraire_numero_serie("série px200-0042") == "PX200-0042"


@pytest.mark.parametrize(
    "reponse,attendu",
    [("garantie", "garantie"), (" Documentation. ", "documentation"), ("`garantie`", "garantie")],
)
def test_parse_reponse_nette(reponse, attendu):
    assert parse_categorie(reponse, "peu importe") == attendu


def test_parse_reponse_ambigue_bascule_sur_heuristique():
    assert parse_categorie("documentation ou garantie ?", Q_GAR) == "garantie"
    assert parse_categorie("aucune idée", Q_DOC) == "documentation"
