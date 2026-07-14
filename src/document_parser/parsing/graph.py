"""Assembly of the parsing LangGraph. Internal to the parsing engine."""

from langgraph.graph import END, START, StateGraph

from document_parser.parsing.nodes import assemble, detect_format, extract
from document_parser.parsing.state import ParsingState


def build_parsing_graph() -> StateGraph:
    graph = StateGraph(ParsingState)
    graph.add_node("detect_format", detect_format)
    graph.add_node("extract", extract)
    graph.add_node("assemble", assemble)
    graph.add_edge(START, "detect_format")
    graph.add_edge("detect_format", "extract")
    graph.add_edge("extract", "assemble")
    graph.add_edge("assemble", END)
    return graph
