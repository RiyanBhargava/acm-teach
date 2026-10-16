"""The two steps: ask the model for HTML, then save it."""

import os
from pathlib import Path

from langchain_groq import ChatGroq

from prompts import SYSTEM_PROMPT
from state import AppState


OUTPUT_FILE = Path(__file__).resolve().parent / "output" / "index.html"


def generate_page(state: AppState) -> dict:
    """Node 1: ask the model for one HTML page."""
    print("1. Generating the page...")
    model = ChatGroq(
        model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
        temperature=0,
        timeout=60,
        max_retries=0,
    )
    response = model.invoke([
        ("system", SYSTEM_PROMPT),
        ("human", state["request"]),
    ])
    html = response.content.strip()

    # Some models still wrap their answer in a markdown code block.
    if html.startswith("```"):
        lines = html.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        html = "\n".join(lines).strip()

    # This is a small format check, not a browser or JavaScript test.
    if "<html" not in html.lower() or "</html>" not in html.lower():
        raise ValueError("The model did not return a complete HTML page. Try again.")
    return {"html": html}


def save_page(state: AppState) -> dict:
    """Node 2: save the generated text to a fixed local file."""
    print("2. Saving the page...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(state["html"], encoding="utf-8")
    return {"file_path": str(OUTPUT_FILE)}
