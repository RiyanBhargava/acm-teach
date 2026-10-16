"""A tiny LangGraph app: generate a web page, then save it."""

import os
from pathlib import Path
from dotenv import load_dotenv

from graph import app


ROOT = Path(__file__).resolve().parent


def main() -> None:
    # Only load this independent project's .env file.
    load_dotenv(ROOT / ".env")
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("Add GROQ_API_KEY to beginner_workshop/.env first.")

    request = input("What small web app should I build? ").strip()
    if not request:
        raise ValueError("Please enter an app idea.")

    result = app.invoke({"request": request})
    print(f"\nSaved: {result['file_path']}")
    print("Review the file, then open it in your browser.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Could not build the app: {error}")
        raise SystemExit(1)
