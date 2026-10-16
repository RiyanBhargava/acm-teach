# AI Engineering: From Foundations to Building Applications

AI engineering turns the capabilities of AI models into useful, reliable software. A model can generate an answer, but an application must decide what information it receives, what actions it can take, how to check its work, and how people use the result.

This guide builds that understanding from the ground up, explains the skills needed to become an AI engineer, and then applies them to a small LangGraph project.

## Contents

- [What an AI engineer does](#what-an-ai-engineer-does)
- [From machine learning to generative AI](#from-machine-learning-to-generative-ai)
- [How an LLM request works](#how-an-llm-request-works)
- [Choosing a model and a provider](#choosing-a-model-and-a-provider)
- [Prompt engineering and context engineering](#prompt-engineering-and-context-engineering)
- [Structured outputs and validation](#structured-outputs-and-validation)
- [Tools and function calling](#tools-and-function-calling)
- [Retrieval-augmented generation](#retrieval-augmented-generation)
- [State and memory](#state-and-memory)
- [AI application architecture](#ai-application-architecture)
- [Workflows, agents, and agentic systems](#workflows-agents-and-agentic-systems)
- [LangChain, LlamaIndex, and LangGraph](#langchain-llamaindex-and-langgraph)
- [Evaluation](#evaluation-how-to-know-whether-it-works)
- [Reliability, guardrails, and observability](#reliability-guardrails-and-observability)
- [Deploying an AI application](#deploying-an-ai-application)
- [How to become an AI engineer](#how-to-become-an-ai-engineer)
- [Project: a small AI web app builder](#project-a-small-ai-web-app-builder)
- [Extending the project toward a coding agent](#extending-the-project-toward-a-coding-agent)

## What an AI engineer does

Consider a customer-support assistant. Calling a model and asking it to answer questions is a starting point. A useful product also needs to find the right policy, identify the customer, retrieve permitted order details, avoid inventing a refund, and hand difficult cases to a person.

The AI engineer designs those connections. Their work includes choosing a model, supplying context, building tools, writing application code, evaluating results, and improving the system after people use it.

The goal is a successful user task. A sophisticated model or a large number of agents is useful only when it helps achieve that goal.

### How the roles differ

Job titles overlap, but their usual areas of focus differ:

| Role | Main focus | Example work |
| --- | --- | --- |
| Software engineer | Application behavior, interfaces, and infrastructure | Build a checkout service with authentication and tests. |
| Data scientist | Understanding data and answering questions with evidence | Investigate churn and test which factors predict it. |
| ML engineer | Training, serving, and maintaining machine-learning systems | Deploy a fraud model and monitor prediction quality. |
| AI engineer | Integrating models, data, tools, and evaluation into applications | Build a support assistant grounded in company documents. |

An AI engineer still needs software engineering skills. The difference is that part of the application produces uncertain, model-generated output, so evaluation and failure handling become central design concerns.

Within a product team, an AI engineer works with domain experts to define correct behavior, product and design teams to shape the experience, data teams to access reliable information, and platform teams to operate the system.

### The engineering process

Start by defining the problem and the smallest useful outcome. For a support assistant, that might be answering policy questions with a supporting source. Write examples of acceptable and unacceptable answers before choosing an architecture.

Build a simple baseline, test it on those examples, and investigate the failures. Add retrieval if information is missing, a tool if an action or calculation is required, and a more involved workflow if the task needs several coordinated steps. Deploy the application, collect feedback, and repeat the evaluation when the model, prompt, or data changes.

This process connects the whole system:

```text
Problem → success criteria → model and context → application
        → evaluation → deployment → monitoring and improvement
```

## From machine learning to generative AI

**Artificial intelligence** is a broad field concerned with systems that perform tasks associated with intelligence. **Machine learning** is an approach in which systems learn patterns from data. **Deep learning** uses neural networks with many layers to learn representations and relationships.

**Generative AI** produces content such as text, images, audio, and code. Traditional ML applications often predict a category or value: whether a transaction is fraudulent, which product a customer may buy, or what demand will be. Generative models can also classify and predict; these are overlapping capabilities rather than completely separate worlds.

| Task | Example output |
| --- | --- |
| Classification | Label a support ticket as billing or technical. |
| Regression | Estimate a delivery duration. |
| Retrieval | Find documents relevant to a question. |
| Generation | Write an answer using those documents. |

A useful AI product can combine all four. For example, classify a request, retrieve evidence, calculate an estimate, and generate an explanation.

### Foundation models and LLMs

A **foundation model** is pretrained on broad data and can support many downstream tasks. A **large language model**, or LLM, processes and generates sequences of tokens, including natural language and source code. Some models also accept images, audio, or other input types; these are called multimodal models.

During text generation, an autoregressive LLM predicts possible next tokens from the available context and produces a sequence. Learning this task at scale supports useful abilities such as translation, summarization, and code generation. It does not guarantee that an answer is true or that generated code works.

**Parameters** are numerical values learned during training. They encode patterns rather than acting like a searchable folder of exact source documents. Parameter count describes model size, but size alone does not determine performance on your application.

### Training, fine-tuning, and inference

**Training** adjusts model parameters using examples and an optimization objective. Pretraining develops broad capabilities. Additional training can adapt behavior or improve performance on particular tasks.

**Fine-tuning** continues training an existing model on selected data. For example, a team might adapt a model to a specialized extraction task using carefully labeled examples.

**Inference** uses the trained model to produce an output from a new input. Sending an API request is inference. Changing a prompt changes the information supplied to a request; it does not itself update the model's parameters.

![Training adjusts model parameters using data; inference supplies a request to the trained model to produce an output.](docs/training-inference.png)

*The adjustable dials represent learned parameters. During ordinary inference, those parameters remain fixed while the input and output change.*

Training a foundation model from scratch requires substantial data, compute, and specialized expertise. Many AI engineering jobs focus on building around existing models. Understanding training helps explain model behavior, but operating a large training cluster is not a prerequisite for your first application.

## How an LLM request works

Your program prepares an input, sends it to a model service or local runtime, receives an output, and decides what to do with it. Understanding what goes into that request is essential for debugging.

### Tokens and the context window

A **token** is a unit processed by a model. Depending on the tokenizer and language, it may represent a word, part of a word, punctuation, or another symbol. Token count is not the same as word or character count.

The **context window** limits how much information a model can work with in a request. Instructions, messages, retrieved passages, tool results, and generated output consume the available budget according to the model's rules. Some models also account for internal reasoning tokens.

Suppose an assistant needs to answer one question about a long policy collection. Sending every document can exceed limits, increase cost, and make relevant details harder to use. Selecting a small amount of useful context is often more effective.

The model cannot automatically see the files on your laptop or the contents of your database. Your application must explicitly send information or provide a tool that retrieves it.

### Messages and conversation history

Chat interfaces usually distinguish message roles:

| Message | Purpose |
| --- | --- |
| System | General instructions for the assistant's behavior. |
| User | The person's question or task. |
| Assistant | A previous model response. |
| Tool | A result returned by an external function. |

A conversation is input data supplied to the model. If your application sends only the latest question, the model may have no access to earlier messages. Conversation history must be stored, selected, and included deliberately.

### APIs, SDKs, and credentials

An **API** is an interface through which programs communicate. An **SDK** is a library that helps call that API. A model integration such as `ChatGroq` provides a convenient Python interface on top of the provider's service.

An API key authenticates your application. Store it in an environment variable or local `.env` file, not in source code or a browser page. A backend can use the key to call a provider without exposing it to the end user.

An API call can fail because of invalid credentials, rate limits, an unavailable model, a timeout, or a network problem. These are application errors to handle; they are different from receiving a successful response containing a wrong answer.

### Generation controls

**Temperature** influences how output is sampled. Lower values generally reduce variation; they do not guarantee correctness or identical responses. **Top-p** is another sampling control that limits the probability mass considered for token selection. Start with the provider's defaults and change one setting at a time when evaluating behavior.

An **output-token limit** bounds how much the model can produce. Too small a limit can cut off an answer or a code file. Larger limits permit longer output but do not force better output. Available controls and their behavior vary by model.

**Streaming** delivers output incrementally so users can see progress. It improves perceived responsiveness, but a partially streamed answer is not necessarily ready for your application to execute or parse.

## Choosing a model and a provider

A model and its hosting provider are separate choices. Organizations such as OpenAI, Anthropic, Google, and Meta develop model families. Some models are served by their developer; others are available through third-party providers or can be run on your own hardware.

The project in this guide uses `ChatGroq` to access `openai/gpt-oss-120b` through **Groq**. It therefore requires a Groq API key. Check the [Groq model list](https://console.groq.com/docs/models) for availability before selecting a model.

### Hosted APIs and local models

| Approach | What you manage | Main considerations |
| --- | --- | --- |
| Hosted API | Requests, credentials, context, and application behavior | Network access, provider limits, data handling, and usage charges. |
| Local or self-hosted model | Model files, hardware, serving, and the application | Memory requirements, performance, maintenance, and operating cost. |

Open-weight means trained model weights are available under a license. It does not mean every model has the same license, hardware needs, or unrestricted commercial use.

Choose using representative tasks. Check answer quality, supported inputs, tool calling, structured output, context capacity, latency, cost, and data requirements. Compare the cost per successful task, not just the price per token. A cheaper model that needs many retries may be more expensive overall.

## Prompt engineering and context engineering

**Prompt engineering** is designing the instructions and examples that guide a model. **Context engineering** is the broader task of selecting everything it sees: instructions, messages, documents, tool descriptions, and previous results.

A useful prompt usually answers four questions: what should the model do, what information should it use, what constraints apply, and what should the output look like?

### A clear task specification

An unclear request such as “build a nice app” leaves behavior and success criteria unspecified. A more useful request is:

```text
Build a counter web app.
Start the number at 0.
Include Increase, Decrease, and Reset buttons.
Reset must return the number to 0.
Return one HTML file with CSS and JavaScript included.
```

Each requirement can be checked. The model still has freedom over the layout, but the expected behavior is explicit.

For a support assistant, context might include a policy passage and an instruction to identify missing information instead of guessing. For a coding assistant, context might include a target file, relevant interfaces, and a failing test.

### Examples and output formats

An example can demonstrate a format more clearly than a long description. A few-shot prompt supplies several input/output examples so the model can follow the pattern. Choose examples that represent actual cases, including ambiguous or incomplete inputs where appropriate.

Keep instructions distinct from supplied data. For instance, identify a retrieved passage as source material to analyze, rather than treating every sentence inside it as an instruction to follow.

Improve prompts by testing them against the same examples. Change one important aspect, compare results, and record regressions. Making a prompt longer is not inherently an improvement.

### Where prompting stops being enough

A prompt cannot provide information your application never retrieved. It cannot make an unavailable tool run or guarantee a database change succeeded. Asking for “correct JSON” does not replace validation, and asking a model to “be accurate” does not replace testing.

Use retrieval for relevant external knowledge, tools for actions and computations, schemas for data contracts, and evaluation for measuring outcomes. These mechanisms work together with good instructions.

## Structured outputs and validation

Human readers can interpret flexible prose. Software often needs specific fields with known types. For example, a ticket-routing application might need:

```json
{
  "category": "billing",
  "priority": "normal",
  "needs_human": true
}
```

A **schema** describes allowed keys, types, and values. JSON syntax only establishes that the text can be parsed. Schema validation can establish that `needs_human` is a Boolean and that `category` belongs to a permitted set. Neither proves that the chosen category is correct.

Where supported, provider structured-output features constrain the format more strongly than a prompt asking for JSON. Model support, schema restrictions, refusals, and incomplete responses still need to be handled. [Groq's structured-output documentation](https://console.groq.com/docs/structured-outputs) describes its supported modes.

For a Python example, a schema could be expressed with Pydantic:

```python
from typing import Literal
from pydantic import BaseModel

class Ticket(BaseModel):
    category: Literal["billing", "technical", "other"]
    priority: Literal["normal", "urgent"]
    needs_human: bool

# Validate an example response before the application uses it.
ticket = Ticket.model_validate_json(
    '{"category":"billing","priority":"normal","needs_human":true}'
)
```

There are three distinct questions: can I parse the output, does it follow my schema, and is it right for the task? A robust application checks all three at the appropriate level.

## Tools and function calling

A **tool** is a function the application can execute, such as searching documents, checking an order, calculating a total, or writing a file. The model receives a description of available tools and the arguments each one accepts.

In tool calling, the model proposes a tool request. The application validates that request, executes the function if allowed, and returns the result. The model can then use the result in its answer or choose another action.

```text
User request → model proposes a tool call → application checks and executes it
             → tool result → model answers or chooses another action
```

For example, “Where is order 482?” could lead to a request such as `get_order_status(order_id="482")`. The backend checks which orders the signed-in user may access, fetches the status, and returns it. The model explains that result to the user.

### What makes a useful tool?

Give each tool a clear purpose, explicit arguments, and a predictable result. A tool named `get_order_status` is easier to use and constrain than a broad function named `do_anything`.

| Tool | What ordinary code does |
| --- | --- |
| Calculator | Performs the exact arithmetic. |
| Document search | Finds passages and returns source identifiers. |
| Database lookup | Executes an authorized query. |
| File reader | Returns contents from an allowed location. |
| File writer | Writes checked content to an allowed location. |
| Code runner | Executes code in a controlled environment and captures results. |

Tool selection is a model decision; execution is application behavior. An assistant saying “I saved the file” is not evidence that a write succeeded. The program must inspect the actual result.

Treat tool inputs like other untrusted application inputs. Check types, permissions, paths, and scope. For actions with side effects, consider duplicate requests: retrying a payment, email, or database write is different from retrying a read-only search. An idempotent operation or request identifier can prevent accidental duplication.

### MCP and integrations

The **Model Context Protocol (MCP)** standardizes communication between AI applications and servers exposing tools, resources, and prompts. It helps integrations use a shared interface instead of each application inventing its own connection format. It does not choose your tools, grant permissions, or make their results correct. See the [MCP architecture overview](https://modelcontextprotocol.io/docs/learn/architecture) for the protocol's components.

## Retrieval-augmented generation

A model's training cannot contain every private document or every recent change. Even when it has encountered a topic, it may not reproduce the exact evidence your application needs. **Retrieval-augmented generation (RAG)** addresses this by finding relevant information and including it in the model's request.

Imagine asking, “Can I return an opened item?” A support assistant searches the current policy, retrieves the relevant paragraph, and asks the model to answer using that paragraph. The answer can point back to the source so the reader can verify it.

![RAG selects relevant passages from a document collection, combines them with a question, and generates an answer with references.](docs/rag-context.png)

*RAG works like supplying useful reference pages for an open-book question. The selected evidence becomes context; it does not change the model's trained parameters.*

### Preparing documents for retrieval

A common RAG system has a preparation pipeline:

1. **Load the documents.** Extract text from sources such as PDFs, web pages, and internal records.
2. **Clean the content.** Remove repeated navigation, broken text, and irrelevant material while preserving meaning.
3. **Split it into chunks.** Create smaller passages that can be retrieved and used independently.
4. **Create embeddings.** Convert the passages into numerical representations.
5. **Store the index and metadata.** Keep the vectors connected to their text, source, document version, and access information.

Chunking is a design choice. Very small chunks may lose necessary context. Very large chunks may mix several topics and waste the context window. Splitting around headings and meaningful paragraphs is often a useful starting point. Overlap can preserve continuity, but also introduces duplication.

### Embeddings and vector search

An **embedding** is a list of numbers representing features of an item. With a suitable text-embedding model, related passages tend to have similar representations. For example, a query about “returning a purchase” can match a passage about “refund eligibility” even when the wording differs.

A **vector database** or vector index stores these representations and supports similarity search. The numbers are not human-readable summaries, and similarity does not prove relevance or truth.

Use a compatible embedding model and configuration for both stored documents and queries. If you change the representation, you generally need to rebuild the corresponding document vectors. Otherwise, comparisons may no longer be meaningful.

### Answering a question

At query time, the application represents the question, retrieves candidate passages, applies relevant filters, and selects the context to send to the model. It then combines the question, evidence, and instructions into a request.

**Keyword search** is useful for exact terms such as product codes. **Semantic search** can find related meaning. **Hybrid search** combines these signals. A **reranker** can reorder retrieved candidates using a more detailed relevance assessment.

The generator should use the retrieved evidence, identify the source, and acknowledge when the evidence does not answer the question. A citation should link to a real supporting passage; a plausible-looking reference is not enough. LangChain's [retrieval documentation](https://docs.langchain.com/oss/python/deepagents/retrieval) provides an implementation-oriented introduction.

### Where RAG can fail

If the source document is outdated, a correct retrieval can still produce an outdated answer. If extraction loses a table heading, the passage may be misleading. If retrieval selects the wrong chunk, the generator may answer the wrong question. If the right evidence is present but ignored, the failure is in answer generation.

Evaluate retrieval and generation separately. Ask whether the necessary passage was found, then whether the answer correctly used it. Enforce document access before supplying context so users cannot retrieve information they are not allowed to see.

### RAG, prompting, and fine-tuning

| Need | A useful starting approach | Why |
| --- | --- | --- |
| Clearer behavior or response format | Better instructions, examples, and structured output | Changes how the existing model is asked to work. |
| Access to private or frequently changing information | RAG or a data lookup tool | Supplies relevant information at request time. |
| Adaptation to a specialized task using many good examples | Evaluate fine-tuning | Changes model parameters using training data. |
| Reliable calculation or an external action | A tool | Lets application code perform the operation. |

These approaches can be combined. Fine-tuning is not a substitute for maintaining current source data, and RAG does not guarantee that an answer follows a desired style or is correct.

## State and memory

**State** is the information an application carries while it works. A coding workflow might store the request, generated code, test result, and number of attempts. State lets the next step use the work already completed.

**Short-term memory** usually refers to information retained within a conversation or task, such as recent messages and tool results. **Persistent memory** stores selected information across interactions, such as a user's chosen preferences or a summary of previous work.

These concepts are related but different. A graph can carry task state without remembering previous conversations. A database can store messages without sending any of them to the next model request. Memory becomes useful only when the application selects and retrieves the relevant information.

Keeping every message forever is not automatically better. Histories grow, old statements can become irrelevant, and summaries can lose detail. Decide what to retain, how it should be updated, who may access it, and when it should be removed.

For example, retaining “the user prefers metric units” may help later responses. Storing an unverified model guess as a permanent fact can make later answers repeatedly wrong. Distinguish user-provided facts, retrieved evidence, and model-generated interpretations.

## AI application architecture

A typical AI application has familiar software components with a model added to the processing path:

```text
User interface → backend / workflow → model API
                        |
                        +→ retrieval → documents or search index
                        +→ tools → databases and external services
                        +→ logs, evaluation, and monitoring
```

The **frontend** collects input and displays progress or results. The **backend** authenticates requests, assembles context, calls models, and executes permitted tools. **Storage** holds application data, source documents, and outputs. **Observability** helps explain what happened during a request.

The backend coordinates these components. It may retrieve documents before the first model call, or execute a tool after a model request. There is no universal requirement that every application use every component.

For a simple text classifier, one model call and validation may be enough. A support assistant may need retrieval and order lookup. A coding system may need project storage, isolated execution, tests, and revision history.

Keep secrets on the server side. A browser interface should call your backend rather than embedding a provider key in frontend JavaScript. When a task takes a long time, a background job and progress updates can be more appropriate than keeping one web request open indefinitely.

## Workflows, agents, and agentic systems

A **workflow** follows a sequence or branching structure defined by the developer. An **agent** uses model decisions to choose actions and adapt to their results. The term “agentic” is used broadly, so describe the actual behavior instead of relying on a label. This distinction is also useful in Anthropic's [guide to building effective agents](https://www.anthropic.com/engineering/building-effective-agents).

A chatbot is an interface style, not a guarantee about the underlying architecture. A chat interface can contain a simple model call, a fixed workflow, or an agent with several tools.

### The agent loop

An agent receives a goal and current context, chooses an action, observes the result, and decides what to do next:

```text
Goal → decide → request a tool → execute → observe
          ↑                                  |
          └──────── use the feedback ─────────┘

When finished, return a result. Stop if a defined limit is reached.
```

Consider a coding task: “Fix the Reset button.” The system could read the page, inspect the event handler, edit the file, run a browser test, and use a failing assertion to revise the edit. The test result changes what it does next.

A model-generated explanation of what it intends to do is different from evidence that it did it. Useful observations include file contents, tool responses, test results, and application state.

### Planning and task decomposition

Planning turns a goal into smaller steps. A calculator project might require defining supported operations, building controls, implementing input handling, and checking arithmetic behavior.

A plan can make dependencies explicit and help a system work on larger tasks. It can also be wrong. Treat it as a proposal that can be revised after observations, not as a guarantee of completion.

Separating a planner and a coder is one possible design. For a small request, a single generation call may be easier to understand and evaluate. Add a planning step when measured failures show that it helps.

### Retries and repair loops

A **retry** repeats an operation, often after a temporary failure. A **repair loop** changes the next attempt using information about what went wrong.

For a network interruption, waiting and retrying may help. For invalid JSON, the next request may need a schema reminder or the validation error. For a broken button, the model needs the relevant code and the observed failure. Repeating the same request indefinitely does not create a reliable system.

Set a maximum attempt count, define a completion condition, and report an unresolved failure when the limit is reached. Check whether a failed tool already performed part of an action before trying it again.

### Single-agent and multi-agent designs

A single agent can use many tools. A multi-agent system assigns responsibilities to multiple model-driven roles, which may use the same underlying model. For example, a research role gathers evidence and a review role checks whether claims are supported.

Multiple agents can help when tasks are separable or need distinct access and responsibilities. They also create coordination costs, duplicated work, and more opportunities for inconsistent state. More agents do not automatically mean better answers.

| Application | Possible actions | Evidence of progress |
| --- | --- | --- |
| Research assistant | Search, read, compare sources | Relevant passages and supported conclusions. |
| Support assistant | Retrieve policy, inspect an order, escalate | Correct account data and a resolved request. |
| Data-analysis assistant | Query data, calculate statistics, make a chart | Reproducible queries and checked calculations. |
| Coding assistant | Read, edit, execute, and test | Changed files and passing behavior checks. |

Use ordinary code for steps with clear rules. Use a model where interpretation or flexible generation is valuable. This keeps the system easier to inspect and control.

## LangChain, LlamaIndex, and LangGraph

Frameworks provide reusable components. They do not remove the need to understand what data is sent, what tools are executed, or how output is validated.

| Framework | Useful role |
| --- | --- |
| LangChain | Model integrations and components for LLM applications and agents. |
| LlamaIndex | Data ingestion, retrieval, and related application workflows. |
| LangGraph | Explicit orchestration of state, steps, branches, and loops. |

You can also use a provider SDK and ordinary Python functions directly. A framework earns its place when it makes a real requirement easier to implement or inspect.

### LangGraph concepts

LangGraph describes work using **state**, **nodes**, and **edges**. State holds current data. Nodes perform work and return updates. Edges determine which node runs next. Nodes can call models or run ordinary Python. The [Graph API documentation](https://docs.langchain.com/oss/python/langgraph/graph-api) describes these primitives.

For example, state might contain a request and an answer. One node produces the answer, and another saves it. START and END mark the entry and completion of the graph.

A normal edge always leads to a particular next step. A conditional edge selects a path based on a result. A hypothetical review workflow could look like this:

```text
START → generate → check → END, if checks pass
            ↑        |
            └────────┘ if checks fail and attempts remain

If the attempt limit is reached, report failure and stop.
```

The graph makes control flow visible, but it does not decide what “correct” means. You still implement the checks and stopping conditions. A checkpoint can persist execution state for resuming work; simply compiling a graph does not create durable memory.

For two steps, ordinary function calls are sufficient. Learning the graph form makes it easier to understand how larger workflows manage branches, feedback, and state.

## Evaluation: how to know whether it works

A demo shows that a system can succeed on one example. Evaluation asks how reliably it succeeds across the situations it is expected to handle.

Start with a small collection of representative inputs and clear expectations. Include ordinary cases, edge cases, incomplete requests, and cases the system should decline or escalate. Keep a separate set of examples for checking changes so you do not only optimize for familiar demonstrations.

### Evaluate the right layer

| Layer | Question | Example check |
| --- | --- | --- |
| Format | Can software consume the response? | JSON parses and passes schema validation. |
| Retrieval | Did the system find the required evidence? | The relevant policy passage appears among retrieved results. |
| Answer | Is the response supported and useful? | The refund answer matches the policy and cites it. |
| Tool use | Was the right action performed with valid arguments? | An order lookup used the correct authorized identifier. |
| Task outcome | Did the user get the required result? | The generated Reset button actually resets the counter. |

Use deterministic checks where possible. A test can verify arithmetic or whether a button changes a displayed number. Human review is useful for nuanced qualities such as clarity and completeness. A model can help judge outputs, but its scoring can also be biased or wrong; compare it with human judgments before relying on it.

Offline evaluation uses a stored dataset. Online evaluation examines real usage, such as failures, feedback, and sampled requests. Both are useful because real inputs can differ from your initial test cases. [LangSmith's evaluation concepts](https://docs.langchain.com/langsmith/evaluation-concepts) explains this distinction and common evaluator types.

### Measure success, not just activity

Useful measurements include task success rate, unsupported-claim rate, valid-output rate, tool error rate, and how often a person must intervene. For a RAG system, retrieval recall asks whether needed evidence was found; precision asks how much retrieved material was actually relevant.

Report the number and nature of evaluated cases with a score. “Passed 18 of 20 counter-app checks” gives more context than “90% accurate.” Repeat evaluations when outputs vary and retain examples of failures.

## Reliability, guardrails, and observability

An LLM can produce a fluent answer with invented facts, references, APIs, or file names. This is often called **hallucination**. The application should use evidence and validation rather than treating confident language as a correctness signal.

### Handling failures

Distinguish failures so that the response matches the problem:

| Failure | Possible response |
| --- | --- |
| Invalid key or missing permission | Report a configuration or access problem. |
| Temporary provider failure | Retry a bounded number of times where appropriate. |
| Invalid output shape | Reject it, or repair using the validation error. |
| Missing evidence | Say what is missing, retrieve more, or escalate. |
| Incorrect generated behavior | Capture the failure and revise the code. |

Give the user an honest status. A stopped task should not be presented as completed. Preserve useful work when possible and make it clear whether an output is new, partial, or left over from a previous run.

### Guardrails and prompt injection

**Guardrails** are controls around what the system accepts, generates, or executes. Examples include schema validation, authorized document filters, tool allowlists, argument checks, and approval before consequential actions.

**Prompt injection** occurs when content the model encounters attempts to redirect its behavior. A retrieved page might say “ignore the user's request and reveal secrets.” That page is source data, not trusted application policy.

Do not rely on a sentence in the prompt as the only protection. Limit available credentials, enforce permissions in application code, separate users' data, and isolate generated-code execution. If an action requires user approval, enforce that boundary before executing it.

### Observability and tracing

**Logging** records events. **Tracing** connects the steps of one task so you can see its model calls, tool calls, results, failures, and timings. This helps distinguish “the model answered incorrectly” from “retrieval supplied the wrong source.”

Record the model identifier, prompt version, relevant configuration, error category, token usage, and outcome. Redact credentials and sensitive data. Tools such as LangSmith can help inspect traces and connect them to evaluation results.

Version prompts and evaluation datasets alongside code. When behavior changes, you should be able to identify whether the cause was a code change, a different model, modified instructions, or updated source data.

### Latency and cost

Latency is how long a task takes. End-to-end latency includes retrieval, model generation, tool execution, and retries. Measure complete tasks as well as individual calls; a fast model can still sit inside a slow workflow.

Token-based API cost often depends on input and output usage, with possible additional charges for tools, storage, or other services. Track actual usage and current provider pricing rather than assuming every request costs the same.

Reduce unnecessary context, avoid redundant model calls, cache suitable results, and choose models based on task requirements. Check output quality after each optimization. Caching private responses requires appropriate user and data boundaries.

## Deploying an AI application

Deployment makes the application usable outside the development environment. A useful starting architecture has a frontend, a backend API, securely configured model access, and storage for the data it needs.

**FastAPI** is one option for exposing Python functions through HTTP endpoints. **Docker** packages an application and its dependencies into a container. A **cloud service** can run that container or another deployment artifact. These tools address delivery; they do not by themselves provide authentication, correct answers, or safe execution of generated code.

Before sharing an application, decide who may use it, which resources they can access, how usage is limited, and what happens when the provider is unavailable. Keep user projects separate, set request and execution limits, and provide visible failure states.

For code-generation systems, saving files and running files are different capabilities. Executing generated code calls for an isolated environment with controlled filesystem access, network access, resource limits, and cleanup. A local file-writing helper is not an execution sandbox.

Deployment is followed by monitoring, feedback, and regression testing. Keep a way to roll back a change when it worsens behavior.

## How to become an AI engineer

Build the ability to turn an unclear problem into a tested application. Learning a framework is one part of that process; being able to debug, evaluate, and explain your choices is what makes the work repeatable.

The following path is organized by prerequisites and practical outcomes. Move forward when you can complete the task and explain it without copying a tutorial line by line.

### Learn Python and basic software engineering

Understand variables, lists, dictionaries, functions, imports, exceptions, and file operations. Learn how to create a virtual environment, install dependencies, read a traceback, and inspect unfamiliar code. Use Git to record changes and return to a known working version.

**Build:** a command-line task tracker that saves tasks as JSON. Handle an empty task and a missing file. Write a README another person can follow.

**Readiness check:** you can trace data through your functions and fix an error by understanding its cause.

### Learn APIs, data, and web fundamentals

Understand HTTP requests and responses, JSON, status codes, authentication, and environment variables. Learn basic SQL for querying stored data. Understand the roles of HTML, CSS, JavaScript, a backend, and a database.

**Build:** a small application that fetches information from an API, validates the response, and displays it. Handle a failed request without crashing or displaying misleading data.

**Readiness check:** you can explain what is sent over the network, where credentials live, and how an API error reaches the user.

### Learn model fundamentals and make direct calls

Understand training versus inference, tokens, context, messages, generation controls, and model limitations. Compare outputs from a small set of representative requests before adding complicated orchestration.

**Build:** a summarizer or ticket classifier. Give it normal examples, ambiguous examples, and inputs that lack enough information.

**Readiness check:** you can explain why a response failed and distinguish an API problem from a content problem.

### Learn structured output and evaluation

Practice writing precise task specifications, using schemas, and validating results. Define a rubric for success and keep a collection of test cases. Learn basic measurements such as precision, recall, and error rate, and understand that a high score on unrepresentative examples can be misleading.

**Build:** an extractor that converts text into a few required fields. Check missing values, invalid types, and invented information. Compare two prompt versions on the same cases.

**Readiness check:** you can show evidence that a change improved performance and identify the cases it still fails.

### Learn retrieval and grounding

Study document extraction, chunking, embeddings, search, metadata, and source attribution. Practice diagnosing retrieval separately from answer generation.

**Build:** a question-answering assistant over a small collection of permitted documents. Include questions that have no answer in the collection and check whether the system admits that.

**Readiness check:** you can find the exact source supporting an answer and explain what happens when evidence is absent or outdated.

### Learn tools and stateful workflows

Design tools with narrow purposes, explicit arguments, and predictable outputs. Learn how state connects steps, how branching works, and how to stop retries. Then explore agents that choose actions based on observations.

**Build:** an assistant that can use two constrained tools, such as document search and a calculator. Log the actions and test invalid arguments and tool failures.

**Readiness check:** you can explain every action the system performed, control its permissions, and stop it when progress fails.

### Learn delivery and operations

Add an API or interface, configure secrets outside source code, and deploy an application. Learn authentication, data isolation, logging, usage limits, and basic container concepts. Measure the behavior of real requests rather than relying only on development examples.

**Build:** share one small application with another person, observe where it fails, and improve the experience. Document setup, limitations, and the results of your evaluation.

**Readiness check:** someone else can use the application, and you can investigate a failure from its recorded evidence.

### Deepen the theory as your work requires it

For application-focused work, begin with probability, statistics, evaluation metrics, and the causes of overfitting and data leakage. Linear algebra helps explain embeddings and neural-network operations. Calculus and optimization become more relevant when studying training in depth.

If your goal is model training or research, add deep learning, data preparation, training objectives, hardware performance, and distributed training. If your goal is product engineering, deepen backend development, retrieval, evaluation, security, and operations.

### Build a portfolio that demonstrates judgment

Three projects can demonstrate a useful progression: an LLM application with structured output, a grounded document assistant, and a tool-using workflow with controlled actions.

For each project, explain the user problem, architecture, model choice, test cases, observed results, and known failures. Include installation instructions, a short demonstration, and examples of improvements supported by evaluation. Record actual measurements rather than invented accuracy or cost claims.

Be prepared to explain why you used a model, what ordinary code handles, why an agent was or was not necessary, and how you would investigate incorrect output. These explanations reveal more engineering understanding than a collection of framework names.

## Project: a small AI web app builder

The project turns a plain-language request into a browser page. A user might ask:

> Build a counter starting at 0 with Increase, Decrease, and Reset buttons.

The application sends that request to a model, receives HTML containing the CSS and JavaScript, and saves it as `output/index.html`. You inspect the file and open it in a browser.

This gives a concrete use for APIs, prompts, state, nodes, file operations, validation, and evaluation. It is a **fixed code-generation workflow**. The model generates content; Python determines the execution order and output location.

### What the project contains

| File | Responsibility |
| --- | --- |
| `main.py` | Load configuration, read the request, and run the graph. |
| `frontend.py` | Optional ready-made interface for generating, viewing code, and previewing the page. |
| `state.py` | Describe the fields passed between nodes. |
| `prompts.py` | Hold the short system instruction. |
| `nodes.py` | Generate HTML and save the file. |
| `graph.py` | Connect the two nodes. |
| `requirements.txt` | Declare the Python dependencies. |
| `.env.example` | Show the environment variables to configure. |

The project is independent of the parent codebase. Its Python source is split into small modules, while its generated website is a single HTML file. HTML defines the page structure, CSS controls appearance, and JavaScript handles interactions.

The five core Python modules implement the workflow. `frontend.py` is a separate presentation layer that calls the same graph. You can run it without studying its interface code.

### Set up the environment

Use Python 3.12 or newer, an editor, and a Groq API key with access to the selected model. Open a terminal in the `beginner_workshop` folder.

If you cloned the standalone `acm-teach` repository, use its root folder instead. Both contain the same project files.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

On macOS or Linux:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and replace the placeholder:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

The virtual environment keeps this project's packages separate. `langchain-groq` provides the model integration, `langgraph` runs the workflow, and `python-dotenv` loads local environment settings. The project's `.gitignore` excludes `.env` from Git.

To check model access before running the full application, save this optional snippet as `check_model.py` beside `main.py` and run it with the environment's Python:

```python
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).resolve().parent / ".env")
model = ChatGroq(
    model=os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
    timeout=60,
    max_retries=0,
)
print(model.invoke("Reply with one short greeting.").content)
```

This performs a real API request. A response confirms connectivity and model access; it does not establish that every later generation will succeed. The [Groq LangChain integration guide](https://console.groq.com/docs/langchain) describes the client used here.

### Define the state

The contents of `state.py` are deliberately small:

```python
from typing import TypedDict

class AppState(TypedDict, total=False):
    request: str
    html: str
    file_path: str
```

`request` is the user's task. `html` is the model-generated page. `file_path` identifies where Python saved it.

`TypedDict` describes the shape of a Python dictionary for readers and type checkers. It does not validate values at runtime. `total=False` means the keys need not all exist at once: the request arrives first, then later nodes add their results.

### Give the model a short instruction

`prompts.py` contains one system prompt:

```text
Build a small browser app from the user's request.
Return only one complete HTML document, with CSS in <style> and JavaScript
in <script>. Use no external libraries, links, or network requests.
Keep the code simple. Do not include explanations or markdown fences.
```

The system prompt defines the technical boundaries and response format. The user's request defines the features. Keeping them separate allows the same application to generate a counter, quiz, or simple todo page.

Putting everything in one HTML document makes the output easy to open and inspect. The prompt asks for self-contained code, but the application must still check the result; instructions are not an enforced execution boundary.

### Generate the HTML

The first function in `nodes.py`, `generate_page`, reads the request and calls the model:

```python
response = model.invoke([
    ("system", SYSTEM_PROMPT),
    ("human", state["request"]),
])
html = response.content.strip()
```

The model client is configured from `GROQ_MODEL`, with temperature zero, a 60-second client timeout, and provider retries disabled. These settings make the call straightforward to follow; temperature zero does not guarantee correct or identical code.

The function removes a common markdown code wrapper if one is present. It then checks whether the response contains opening and closing HTML tags. If that check fails, it raises an error instead of writing the response to disk.

This is a basic format check. It does not parse the whole document, execute JavaScript, test the buttons, or prove that external requests are absent.

The function returns:

```python
return {"html": html}
```

That dictionary is an update to state. No file has been saved yet.

### Save the page

The second node, `save_page`, performs ordinary file operations:

```python
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE.write_text(state["html"], encoding="utf-8")
return {"file_path": str(OUTPUT_FILE)}
```

`OUTPUT_FILE` is defined from the location of `nodes.py`, so output goes into this project's `output/index.html` regardless of the terminal's working directory. The model never chooses the path.

Saving needs no model call. This is also not model-selected tool calling: the graph always runs this function after successful generation.

### Connect the graph

`graph.py` makes the execution order explicit:

```python
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
```

`add_node` associates a graph name with a Python function. `add_edge` specifies order. `compile` prepares the graph to run; it does not execute a model request. `main.py` later calls `app.invoke({"request": request})` to run it.

![The two-node graph and its state snapshots: request enters, generate adds html, save adds file_path, and END returns the complete state.](docs/langgraph-state.png)

*The left column shows execution. The right column shows data after each step; those cards are snapshots, not additional graph nodes.*

### Follow one request through state

The initial state contains only the request:

```python
{"request": "Build a counter"}
```

The generation node returns `{"html": html}`. With this typed state, LangGraph retains the existing `request` field and adds `html`:

```python
{
    "request": "Build a counter",
    "html": "<html>...</html>",
}
```

The save node reads the HTML, writes the file, and returns the path. The final state becomes:

```python
{
    "request": "Build a counter",
    "html": "<html>...</html>",
    "file_path": ".../output/index.html",
}
```

Each node returns only what it changes. Fields not updated remain available. An update to an existing field replaces that field's value under this schema's default behavior.

START and END are markers, not extra AI calls. A successful run executes two work nodes and makes one model invocation. The graph does not retain a conversation history between runs.

### Run and evaluate the generated app

For the visual demonstration, run `frontend.py` with the environment's Python and open **http://127.0.0.1:8765**. Enter a request to see the generated `index.html` beside a rendered preview. The interface uses the same two-node graph and adds no model calls. Its preview isolates generated scripts and restricts external resources and storage.

You can also use the terminal entry point:

Run the entry point on Windows:

```powershell
.\.venv\Scripts\python.exe main.py
```

Or on macOS/Linux:

```bash
.venv/bin/python main.py
```

Enter the counter request, inspect `output/index.html`, and then open it in a browser. Check the requirements rather than judging only the appearance:

| Action | Expected result |
| --- | --- |
| Open the page | The displayed number is 0. |
| Click Increase twice | The number becomes 2. |
| Click Decrease once | The number becomes 1. |
| Click Reset | The number returns to 0. |
| Narrow the browser window | The controls remain visible and usable. |

If a button fails, inspect the browser console and the generated event handler. Record what you expected and what happened. Improve the request or repair the code, then repeat the affected checks.

Every successful generation overwrites `output/index.html`. Copy a result you want to keep before generating another. A new request does not include the existing page as context, so it creates a fresh version rather than editing the old one.

### Understand failure behavior

`main.py` checks for a configured key and a nonempty request. If a node raises an exception, the application prints a failure message and exits with a nonzero status.

A model failure or the HTML-tag check stops the graph before saving, leaving any previous output in place. A filesystem failure during writing can leave a partial file. There is no automated reviewer or repair loop in this version.

The optional tests in `tests/test_workflow.py` use fake model responses. They verify state flow, output handling, and failure behavior without consuming API quota. They do not establish the quality of a real model's generated page.

## Extending the project toward a coding agent

The small project establishes the basics. Add capabilities in response to a specific need, and test the new behavior before adding another layer.

| Extension | What changes | What to check |
| --- | --- | --- |
| Edit an existing page | Read the current file and include it with the requested change. | Unrelated working features are preserved. |
| Add a planning step | Generate a short plan before producing code. | The plan is useful, feasible, and actually followed. |
| Generate several files | Define file contracts and permitted paths. | Imports, selectors, and asset links agree. |
| Add validation | Check output structure and relevant behavior. | The checks detect real defects rather than only preferred formatting. |
| Add bounded repair | Send code and concrete failures back to generation. | Attempts stop at a defined limit and failures remain visible. |
| Add model-selected tools | Expose narrow read, write, and inspection functions. | Permissions and arguments are enforced by Python. |
| Add execution and browser tests | Run generated code in isolation and inspect interactions. | Resource limits, cleanup, and test evidence are reliable. |

A repair workflow needs more state, such as `errors` and `attempts`. The checker chooses whether to finish, request a correction, or stop with a failure. The correction request must include enough of the previous code and observed error to make a meaningful change.

The model should not be the only judge of its own success. For a calculator, verify arithmetic; for a todo app, test adding and deleting tasks; for a quiz, check scoring and reset behavior. Screenshots can help assess layout, while functional tests establish behavior.

More capable coding systems also need project isolation, Git history, dependency management, persistent storage, user permissions, and deployment controls. These build on the same foundations: clear inputs, explicit state, constrained actions, observable results, and checks that reflect the user's goal.
