"""Assembly of the chunking LangGraph. Internal to the chunking engine."""

from langgraph.graph import END, START, StateGraph

from document_parser.chunking.nodes import finalize, split
from document_parser.chunking.state import ChunkingState


def build_chunking_graph() -> StateGraph:
    graph = StateGraph(ChunkingState)
    graph.add_node("split", split)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "split")
    graph.add_edge("split", "finalize")
    graph.add_edge("finalize", END)
    return graph
