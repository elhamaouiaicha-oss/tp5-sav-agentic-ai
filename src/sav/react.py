"""Agent ReAct écrit « à la main » avec LangGraph (comme dans le cours), pour voir les noeuds.

        START -> llm --(appel d'outil ?)--> tools -> llm ...  --(non)--> END

Les noeuds `llm` et `tools` correspondent aux cases « LLM » et « Tool Node » du schéma.
"""

from __future__ import annotations

from typing import Any, Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition


def build_react_agent(
    llm: BaseChatModel,
    tools: Sequence[Any],
    system_prompt: str,
    name: str = "agent",
):
    llm_avec_outils = llm.bind_tools(list(tools))

    def appeler_llm(state: MessagesState) -> dict:
        reponse = llm_avec_outils.invoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [reponse]}

    async def aappeler_llm(state: MessagesState) -> dict:
        reponse = await llm_avec_outils.ainvoke([SystemMessage(system_prompt), *state["messages"]])
        return {"messages": [reponse]}

    graph = StateGraph(MessagesState)
    graph.add_node("llm", RunnableLambda(appeler_llm, aappeler_llm))
    # handle_tool_errors=True : une panne d'outil (ex. serveur MCP éteint) devient un message
    # d'erreur renvoyé au LLM au lieu de faire planter tout le graphe.
    graph.add_node("tools", ToolNode(list(tools), handle_tool_errors=True))
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "llm")
    return graph.compile(name=name)
