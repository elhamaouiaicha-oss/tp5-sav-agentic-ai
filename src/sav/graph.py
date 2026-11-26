"""Graphe LangGraph : orchestrateur + agent documentaire + agent garantie.

                     ┌──────────────► agent_documentaire ──┐
    START ─► orchestrateur ◄──────────────────────────────┤   (rend la main)
                │    └──────────────► agent_garantie ─────┘
                └── réponse finale disponible ──► END
"""

from __future__ import annotations

from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph

from . import config
from .llm import get_llm
from .react import build_react_agent
from .routing import Categorie, parse_categorie
from .tools.garantie import consulter_garantie

# ----------------------------------------------------------------------------- prompts

PROMPT_ORCHESTRATEUR = """[CLASSIFIEUR] Tu es l'orchestrateur du service après-vente d'un distributeur
d'appareils électroniques. Classe la demande du client dans UNE SEULE catégorie :
- documentation : dépannage, signification d'un voyant/pictogramme/message, mode d'emploi,
  procédure (avec ou sans photo de l'écran) ;
- garantie : vérification de la garantie à partir d'un numéro de série.
Réponds par un seul mot : documentation ou garantie."""

PROMPT_DOCUMENTAIRE = """Tu es l'agent documentaire (dépannage) du service après-vente. Tu réponds en français.
Pour toute question technique, appelle l'outil `rag_multimodal` avec la question du client et,
si la demande mentionne une photo (image_ref=...), transmets exactement cette référence.
Fonde ta réponse uniquement sur le résultat de l'outil et conserve ses sources (manuel, page, figure).
Si l'outil est indisponible ou ne trouve rien, dis-le honnêtement.
Termine toujours par une ligne « Sources : » avec le manuel et les pages cités par l'outil.
N'ajoute ni image ni lien Markdown."""

PROMPT_GARANTIE = """Tu es l'agent vérificateur de garantie du service après-vente. Tu réponds en français.
Extrais le numéro de série du message et appelle l'outil `consulter_garantie`.
Si aucun numéro de série n'est fourni, demande-le au client sans appeler l'outil.
Réponds avec : le modèle, la date d'achat, la date de fin de garantie et si l'appareil est
actuellement sous garantie. N'invente aucune information absente du résultat de l'outil."""


# ----------------------------------------------------------------------------- état


class SAVState(MessagesState):
    image_ref: str | None
    categorie: Categorie | None
    reponse_finale: str | None
    agent: str | None


def texte(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict))


def construire_requete(question: str, image_ref: str | None = None) -> dict[str, Any]:
    contenu = question.strip()
    if image_ref:
        contenu += f"\n\n[Photo jointe : image_ref={image_ref}]"
    return {
        "messages": [HumanMessage(contenu)],
        "image_ref": image_ref,
        "categorie": None,
        "reponse_finale": None,
        "agent": None,
    }


# ----------------------------------------------------------------------------- noeuds


def _noeud_specialiste(agent_graph, nom: str):
    async def noeud(state: SAVState, config: RunnableConfig) -> dict:
        n_avant = len(state["messages"])
        resultat = await agent_graph.ainvoke(
            {"messages": state["messages"]}, config={**config, "recursion_limit": 12}
        )
        nouveaux = resultat["messages"][n_avant:]
        finale = next(
            (texte(m) for m in reversed(nouveaux) if isinstance(m, AIMessage) and not m.tool_calls),
            "Je n'ai pas pu produire de réponse.",
        )
        return {"messages": nouveaux, "reponse_finale": finale, "agent": nom}

    return noeud


def build_graph(llm: BaseChatModel | None = None, *, outils_doc: list | None = None):
    """Construit le graphe compilé.

    `outils_doc` : outils de l'agent documentaire. En fonctionnement normal ce sont les outils
    distants chargés via le client MCP (voir `charger_outils_mcp`).
    """
    llm = llm or get_llm()
    if not outils_doc:
        raise ValueError("outils_doc est vide : l'agent documentaire a besoin de `rag_multimodal`.")

    agent_doc = build_react_agent(llm, outils_doc, PROMPT_DOCUMENTAIRE, name="agent_documentaire")
    agent_garantie = build_react_agent(llm, [consulter_garantie], PROMPT_GARANTIE, name="agent_garantie")

    async def orchestrateur(state: SAVState) -> dict:
        # Retour d'un agent spécialisé : la réponse finale existe déjà -> rien à faire (FINISH).
        if state.get("reponse_finale") or state.get("categorie"):
            return {}
        question = next(
            (texte(m) for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), ""
        )
        sortie = await llm.ainvoke([SystemMessage(PROMPT_ORCHESTRATEUR), HumanMessage(question)])
        return {"categorie": parse_categorie(texte(sortie), question)}

    def router(state: SAVState) -> str:
        if state.get("reponse_finale"):
            return END
        return {"documentation": "agent_documentaire", "garantie": "agent_garantie"}[state["categorie"]]

    g = StateGraph(SAVState)
    g.add_node("orchestrateur", orchestrateur)
    g.add_node("agent_documentaire", _noeud_specialiste(agent_doc, "agent_documentaire"))
    g.add_node("agent_garantie", _noeud_specialiste(agent_garantie, "agent_garantie"))
    g.add_edge(START, "orchestrateur")
    g.add_conditional_edges(
        "orchestrateur",
        router,
        {"agent_documentaire": "agent_documentaire", "agent_garantie": "agent_garantie", END: END},
    )
    g.add_edge("agent_documentaire", "orchestrateur")
    g.add_edge("agent_garantie", "orchestrateur")
    return g.compile(name="sav_orchestrateur")


# ----------------------------------------------------------------------------- MCP


async def charger_outils_mcp(url: str | None = None) -> list:
    """Client MCP : récupère les outils publiés par le serveur (Streamable HTTP)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    url = url or config.mcp_url()
    client = MultiServerMCPClient({"sav": {"transport": "streamable_http", "url": url}})
    try:
        outils = await client.get_tools()
    except Exception as e:  # noqa: BLE001 - on veut un message clair pour l'utilisateur
        raise ConnectionError(
            f"Serveur MCP injoignable ({url}). Lancez-le d'abord : python -m sav.mcp_server"
        ) from e
    if "rag_multimodal" not in {o.name for o in outils}:
        raise RuntimeError(f"L'outil `rag_multimodal` n'est pas publié par {url}.")
    return [o for o in outils if o.name == "rag_multimodal"]


async def creer_application(llm: BaseChatModel | None = None, url: str | None = None):
    return build_graph(llm, outils_doc=await charger_outils_mcp(url))


# ----------------------------------------------------------------------------- exécution


async def executer(app, question: str, image_ref: str | None = None) -> dict[str, Any]:
    """Exécute une demande et renvoie la réponse + la trace (noeuds traversés, outils appelés)."""
    trajet: list[str] = []
    outils: list[str] = []
    etat: dict[str, Any] = {}
    async for maj in app.astream(construire_requete(question, image_ref), stream_mode="updates"):
        for noeud, update in maj.items():
            trajet.append(noeud)
            update = update or {}
            etat.update({k: v for k, v in update.items() if k != "messages"})
            for m in update.get("messages", []):
                if isinstance(m, AIMessage):
                    outils += [tc["name"] for tc in m.tool_calls]
    return {
        "reponse": etat.get("reponse_finale", ""),
        "categorie": etat.get("categorie"),
        "agent": etat.get("agent"),
        "trajet": trajet,
        "outils": outils,
    }
