"""Fabrique de LLM, pilotée par la variable d'environnement LLM_MODEL."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from . import config


def get_llm() -> BaseChatModel:
    nom = config.llm_model()
    if nom == "fake":
        # Mode hors-ligne pour tester la plomberie (graphe, MCP, RAG) sans clé API.
        from .fake_llm import FakeLLM

        return FakeLLM()
    from langchain.chat_models import init_chat_model

    return init_chat_model(nom, temperature=0)
