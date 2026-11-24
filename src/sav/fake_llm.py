"""LLM factice DÉTERMINISTE (LLM_MODEL=fake) pour tester toute la plomberie hors-ligne.

Il ne « comprend » rien : il applique des règles simples (appeler chaque outil disponible avec les
bons arguments, puis recopier les résultats des outils). À n'utiliser que pour les tests et pour
vérifier que le graphe / le serveur MCP / le RAG sont correctement câblés.
"""

from __future__ import annotations

import re
import uuid
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from .routing import extraire_numero_serie, heuristique_categorie

_PHOTO = re.compile(r"\[Photo jointe\s*:\s*image_ref=([\w\-.]+)\]")


def _texte(msg: BaseMessage) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return " ".join(b.get("text", "") for b in c if isinstance(b, dict))


class FakeLLM(BaseChatModel):
    tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "fake-sav"

    def bind_tools(self, tools, **kwargs: Any):  # type: ignore[override]
        noms = [getattr(t, "name", None) or t.get("name") for t in tools]
        return self.model_copy(update={"tool_names": noms})

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._repondre(messages))])

    def _repondre(self, messages: list[BaseMessage]) -> AIMessage:
        systeme = _texte(messages[0]) if isinstance(messages[0], SystemMessage) else ""
        humains = [m for m in messages if isinstance(m, HumanMessage)]
        demande = _texte(humains[-1]) if humains else ""
        question = _PHOTO.sub("", demande).strip()

        if "[CLASSIFIEUR]" in systeme:
            return AIMessage(content=heuristique_categorie(question))

        # Résultats d'outils déjà obtenus -> réponse finale = recopie des observations.
        fin = []
        for m in reversed(messages):
            if isinstance(m, ToolMessage):
                fin.append(_texte(m))
            else:
                break
        if fin:
            return AIMessage(content="[LLM factice] " + "\n".join(reversed(fin)))

        photo = _PHOTO.search(demande)
        ref = photo.group(1) if photo else ""
        appels = []
        for nom in self.tool_names:
            args: dict[str, str] | None = None
            if nom == "rag_multimodal":
                args = {"question": question, "image_ref": ref}
            elif nom == "consulter_garantie":
                serie = extraire_numero_serie(question)
                args = {"numero_serie": serie} if serie else None
            elif nom in ("rechercher_passages", "rechercher_figures"):
                args = {"requete": question}
            elif nom == "rechercher_figures_similaires":
                args = {"image_ref": ref} if ref else None
            if args is not None:
                appels.append(
                    {"name": nom, "args": args, "id": f"call_{uuid.uuid4().hex[:12]}", "type": "tool_call"}
                )
        if appels:
            return AIMessage(content="", tool_calls=appels)
        return AIMessage(content="[LLM factice] Pouvez-vous préciser votre demande (ex. numéro de série) ?")
