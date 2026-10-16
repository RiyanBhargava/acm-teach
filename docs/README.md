# Learning illustrations

The AI engineering guide uses two generated educational illustrations and one precise diagram of the implemented graph.

| Diagram | What it explains | Files |
| --- | --- | --- |
| Training and inference | Learning parameters versus using a trained model | [PNG](training-inference.png) |
| RAG as supplied context | Selecting reference material and including it in a request | [PNG](rag-context.png) |
| LangGraph state and execution | The actual two-node graph and its state updates | [PNG](langgraph-state.png), [SVG](langgraph-state.svg) |

The PNGs work in ordinary Markdown image viewers. Keep the `docs/` image paths available relative to the page when moving the Markdown into a documentation project. The LangGraph SVG is an editable vector source.

The two conceptual illustrations were created with the built-in image generation tool. Their full prompts are saved in [ILLUSTRATION_PROMPTS.md](ILLUSTRATION_PROMPTS.md). No image-generation packages are required to run the application.
