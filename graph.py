"""Connect the two nodes in a fixed order."""

from langgraph.graph import END, START, StateGraph

from nodes import generate_page, save_page
from state import AppState


builder = StateGraph(AppState)
builder.add_node("generate", generate_page)
builder.add_node("save", save_page)
builder.add_edge(START, "generate")
builder.add_edge("generate", "save")
builder.add_edge("save", END)
app = builder.compile()
