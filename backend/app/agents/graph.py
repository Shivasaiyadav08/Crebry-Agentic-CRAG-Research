"""CRAG decision graph with resumable, bounded agent-directed correction."""

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import make_nodes
from app.agents.state import GraphState


def evidence_route(state: GraphState) -> str:
    """Use trustworthy local evidence, augment partial evidence, or replace bad evidence."""
    return "refine" if state.get("crag_verdict") == "CORRECT" else "rewrite"


def validation_route(state: GraphState) -> str:
    """An invalid answer can trigger a new plan only while draft retries remain."""
    # No evidence exists to repair via answer rewriting; preserve a clear source warning.
    if "sources" in state and not state["sources"]:
        return "finalize"
    if not state.get("valid", False) and state.get("attempts", 0) <= state["settings"]["max_retries"]:
        return "plan_correction"
    return "finalize"


def correction_route(state: GraphState) -> str:
    return {
        "regenerate": "generate",
        "retrieve": "retrieve",
        "web_search": "rewrite",
        "stop": "finalize",
    }.get(state.get("correction_action", "stop"), "finalize")


# Both the live graph and saved checkpoints use these same transitions.
NEXT_NODES = {
    "retrieve": "grade",
    "grade": "assess",
    "rewrite": "web_search",
    "web_search": "refine",
    "refine": "generate",
    "generate": "validate",
    "finalize": "end",
}
CONDITIONAL_ROUTES = {
    "assess": (evidence_route, ("refine", "rewrite")),
    "validate": (validation_route, ("plan_correction", "finalize")),
    "plan_correction": (correction_route, ("generate", "retrieve", "rewrite", "finalize")),
}


def next_node(node: str, state: dict) -> str:
    """Choose the next checkpoint using the live graph's routing rules."""
    if node in CONDITIONAL_ROUTES:
        route, _ = CONDITIONAL_ROUTES[node]
        return route(state)
    return NEXT_NODES[node]


def build_graph(vectors, routing_store=None):
    nodes = make_nodes(vectors, routing_store)
    graph = StateGraph(GraphState)
    for name, function in nodes.items():
        graph.add_node(name, function)
    graph.add_conditional_edges(
        START, lambda state: state.get("resume_node", "retrieve"), {name: name for name in nodes}
    )
    for source, target in NEXT_NODES.items():
        graph.add_edge(source, END if target == "end" else target)
    for source, (route, targets) in CONDITIONAL_ROUTES.items():
        graph.add_conditional_edges(source, route, {target: target for target in targets})
    return graph.compile()
