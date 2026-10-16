# Tiny AI Web App Builder

A small AI web app builder. Describe an app in the demo interface or terminal. One model call generates a single HTML file containing its HTML, CSS, and JavaScript.

The application uses five short Python files, each with one clear purpose. Start with the [AI engineering guide](ai_engineering.md) for detailed foundations, a practical learning path, and a step-by-step project explanation.

## What became simpler?

| Original project | This project |
| --- | --- |
| Planner, architect, coder | Generate and save |
| N + 2 model invocations for N coding tasks | One model invocation per run |
| Several Pydantic models and JSON parsing | One dictionary-shaped state |
| Multiple files and a coder loop | One `index.html`, no loop |
| Studio UI and HTTP backend | Optional one-file demo interface |
| Separate prompts and file tools | One short prompt and a normal Python write |

```text
START -> generate -> save -> END
```

![LangGraph execution and state changes, from the request through generation and saving.](docs/langgraph-state.png)

The right-hand cards are snapshots of the shared dictionary, not additional nodes. See the guide for a step-by-step explanation.

This is a fixed LLM workflow. It introduces the building blocks used in agents, but the model does not select tools, run tests, or decide what happens next. Two ordinary function calls would also work; LangGraph is here to teach state, nodes, and edges.

## Independent setup

Copy this folder anywhere, or use it inside the existing repository. It imports nothing from the parent project. Python 3.12+ and a Groq API key are required; installation and generation need internet access. API usage is subject to your account's limits and billing.

Clone this repository and open its folder. If you already have `beginner_workshop` locally, open that folder instead. Run the setup commands from the folder containing `requirements.txt`.

```bash
git clone https://github.com/RiyanBhargava/acm-teach.git
cd acm-teach
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` in your editor and replace the placeholder with your own key. Then run:

```powershell
.\.venv\Scripts\python.exe frontend.py
```

These commands use the environment's Python directly, so PowerShell activation is unnecessary.

### macOS / Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
# Edit .env and add your key before running:
.venv/bin/python frontend.py
```

Open **http://127.0.0.1:8765**. Enter an idea and select **Generate app**. The left panel shows `index.html`; the right panel renders it. Use **Download HTML** to keep a copy and **Restart preview** to reset the page. Stop the server with `Ctrl+C`.

`frontend.py` contains the entire optional interface: a local Python server and embedded HTML, CSS, and JavaScript. It calls the same graph as the terminal app. Run it for demonstrations; the five core modules remain the teaching material. No frontend packages or build step are needed.

The preview runs inline JavaScript in a sandboxed iframe. External resources, network calls, and browser storage access are restricted, so request self-contained apps. This is a local demonstration interface; it binds only to `127.0.0.1`.

## First run

For the terminal-only version, run `main.py` instead of `frontend.py` using the same environment's Python.

At the prompt, enter:

> Build a counter with a number starting at 0 and Increase, Decrease, and Reset buttons.

Expected terminal flow:

```text
What small web app should I build? Build a counter ...
1. Generating the page...
2. Saving the page...
Saved: .../beginner_workshop/output/index.html
Review the file, then open it in your browser.
```

Inspect `output/index.html`, then double-click it to open it in a browser. Check that Increase adds one, Decrease subtracts one, and Reset returns to zero. Model output varies; a saved file does not guarantee correct behavior.

Each successful run **overwrites `output/index.html`**. Copy it elsewhere before generating another app if you want to keep it. A failed model call or format check happens before writing, so the previous output remains. Do not mistake an old file for new output after a failure.

Try a todo list with add/delete buttons, a color changer, or a three-question quiz. Keep requests to a few browser-only features. Login, databases, server code, and framework builds are outside this project's scope.

## Read the code in this order

1. [state.py](state.py): `AppState` describes the three dictionary keys.
2. [prompts.py](prompts.py): `SYSTEM_PROMPT` contains the instructions sent to the model.
3. [nodes.py](nodes.py): `generate_page` calls the model and checks the response; `save_page` writes the HTML.
4. [graph.py](graph.py): registers the two functions as nodes, connects them, and compiles the graph.
5. [main.py](main.py): loads settings, reads input, runs the graph, and prints the saved path.

The modules keep related code together. Splitting the code does not add model calls or graph steps. `main.py` imports the compiled graph; `graph.py` imports the node functions and state; `nodes.py` imports the prompt and state. Run `main.py` to start the application.

`TypedDict` describes dictionary keys for readers and type checkers. It does not validate model output at runtime. `total=False` allows keys to appear as the graph progresses. With this state schema, each node returns just the keys it changes; the other state keys remain available.

## Configuration and troubleshooting

`GROQ_MODEL` in `.env` selects the model; the default matches the original project. Choose a model available to your Groq account using the [official model list](https://console.groq.com/docs/models). Existing environment variables take precedence over `.env` values.

| Problem | Action |
| --- | --- |
| Missing API key | Add your key to this folder's `.env`; check the file is not named `.env.txt`. |
| Authentication failure | Check the key belongs to Groq and is valid. |
| Model unavailable | Update `GROQ_MODEL` using your account's available models. |
| Missing Python package | Install requirements using the same `.venv` Python used to run the app. |
| Rate limit or timeout | Wait and rerun; use a smaller request or check account limits and connectivity. |
| Incomplete HTML | Try a smaller app; the format check rejects missing outer HTML tags. |
| Buttons do not work | Inspect the browser console and generated code; simplify the request and regenerate. |

The client timeout is 60 seconds and automatic provider retries are disabled to make the workshop flow easy to follow. Temperature zero reduces sampling randomness; it does not guarantee identical or correct output.

Keep `.env` private. The app sends your request to Groq. Review generated HTML/JavaScript before opening it; it runs in your browser. The prompt requests no network calls, but that instruction is not an enforced sandbox. Python saves the response and never executes generated code automatically.

## Optional offline checks

The tests use a fake model response and temporary output files; they do not need an API key or make network calls.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS/Linux, replace the Python path with `.venv/bin/python`. These checks verify the Python workflow and failure behavior, not the quality of a real model's generated app.

## Folder contents

```text
beginner_workshop/
  main.py                  # Settings, terminal input, and running the app
  frontend.py              # Optional demo UI: source code + rendered website
  state.py                 # The three state fields
  prompts.py               # One short model instruction
  nodes.py                 # Generate HTML and save HTML
  graph.py                 # START -> generate -> save -> END
  requirements.txt         # Three direct dependencies
  .env.example             # Copy to .env and add your key
  .gitignore
  README.md
  ai_engineering.md   # Concepts, diagrams, and learning path
  tests/test_workflow.py    # Optional offline checks
  docs/                    # Diagrams and editable diagram sources
  output/index.html        # Created when you run the app
```

Only the application code is required reading for the coding session. Dependencies are pinned to the versions used for local verification; this is not a full transitive dependency lockfile.
