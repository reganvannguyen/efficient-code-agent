# Efficient Code Agent

**A lightweight, local-first coding agent harness designed to make small language models more effective on everyday hardware.**

> **Status:** Early development. The local model chat command and stateless interactive prompt loop are implemented; autonomous agent capabilities remain planned.

## Why this project?

Most coding agents benefit from large models, long context windows, and generous compute budgets. This project explores a different question:

**Can a small local coding model solve more software engineering tasks when its harness intelligently manages context, retrieves relevant code, and verifies its work?**

The initial hardware target is a **16 GB Apple Silicon MacBook Air**. The goal is not to train a new model or develop a new inference engine; it is to improve **task success under practical memory and compute constraints**.

## Planned capabilities

### 1. Self-editing working context (CLM-inspired)

Rather than relying solely on a growing transcript or automatic summarization, the agent will be able to propose edits to a compact working context:

- Preserve relevant discoveries and decisions.
- Replace outdated hypotheses with verified findings.
- Remove no-longer-useful details.
- Maintain continuity across multiple model calls.

The harness will validate context edits, preserve immutable instructions and task requirements, and enforce a deterministic token-budget fallback. Whether proactive self-editing actually improves results will be tested against a conventional baseline.

### 2. Code-aware repository retrieval

Use **ripgrep** for fast lexical search and **Tree-sitter** for syntax-aware parsing. Planned retrieval features include symbol lookup, lightweight repository maps, and targeted code snippets. The baseline will use bounded ripgrep search first; Tree-sitter enhancements come later, behind the same core repository tool interface.

The aim is to supply the model with **the right code**, not the entire repository. Embedding-based retrieval is intentionally deferred until benchmarks justify the additional memory and complexity.

### 3. Persistent task memory

Store inspectable, agent-editable notes in **Markdown**, initially including a working context and task findings. The model proposes working-note changes through `update_context`; the harness validates and persists them. Selected findings can be retrieved with bounded `read_memory` when needed. A separate `save_memory` tool is deferred.

Full execution history stays separate from the condensed working context, so important evidence can be recovered without including every prior tool result in each prompt.

### 4. Verification-driven agent loop

A Python runtime will coordinate repeated **inspect → plan → edit → test → evaluate** cycles. The agent may request repository searches, file reads and edits, or test commands. The harness executes approved tools and returns their results to the model.

Planned protections include workspace boundaries, command validation, step/time budgets, bounded prompts, and explicit completion criteria. A model's completion proposal is distinct from runtime verification, and independent benchmark tests determine whether a coding task was actually resolved.

### 5. Hardware-aware context budgeting

Constrain prompt sizes and retrieved-code volume for small quantized local models. Track task success alongside resource costs such as tokens, runtime, and peak memory. Inference will be provided by **llama.cpp**, rather than a custom inference engine.

## Planned architecture

```text
CLI
 |
 v
Python Agent Runtime
 |-- Context Manager (CLM-inspired edits + token limits)
 |-- Repository Retrieval (ripgrep + Tree-sitter)
 |-- Coding Tools (search, read, edit, test)
 |-- Persistent Memory (Markdown)
 |-- Run Tracing / Benchmarks (SQLite)
 |
 v
llama.cpp local HTTP server
 |
 v
Quantized coding LLM (GGUF)
```

The **Python harness calls the local model** through llama.cpp's HTTP interface. Model outputs may request tools or context edits; Python validates and performs those operations before the next inference call.

## Technology stack

| Technology | Role |
| --- | --- |
| Python | CLI, agent runtime, tool execution, context management |
| llama.cpp + GGUF | Local inference for quantized coding models |
| Tree-sitter | Syntax-aware code parsing and symbol extraction |
| ripgrep (`rg`) | Fast repository and text search |
| Markdown | Human-readable working context and persistent task notes |
| SQLite | Supporting storage for run history, tracing, and benchmark measurements |
| pytest | Automated tests and evaluation helpers |

**Interface:** CLI first, potentially CLI only. FastAPI and React are optional future additions, not part of the initial scope.

## Local development

Install ripgrep (`rg`) before using the repository `list_files` tool. On macOS,
install it with `brew install ripgrep`; other platforms can use their package
manager or the [ripgrep installation guide](https://github.com/BurntSushi/ripgrep#installation).

From the repository root, create and activate a local virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
```

Install the project with its development tools:

```sh
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

Run the checks:

```sh
python -m pytest
ruff check src tests
ruff format --check src tests
```

After installing, run `harness` to open interactive mode. Start `llama-server`
separately before sending a prompt.

Run `deactivate` to leave the environment. The `.venv/` directory is ignored by Git.

## Evaluation strategy

The project will compare the **same local model on the same coding tasks** using:

1. A minimal baseline harness.
2. A harness with code-aware retrieval.
3. A harness with proactive context editing.
4. A combined optimized harness.

Primary metric: **coding-task success rate**, evaluated with tests or task-specific checks.

Secondary metrics: token usage, agent steps, elapsed time, and peak memory. Controlled comparisons and ablation tests will help identify which techniques provide measurable benefits. We also plan an external comparison with **mini-SWE-agent** on selected tasks, documenting differences in tools and policies. Locally evaluated tasks will not be presented as official SWE-bench scores.

**No accuracy or efficiency improvements are claimed until benchmarks have been run.**

## Development roadmap

Development follows seven incremental phases. Each phase should deliver working behavior, tests, and an explicit acceptance criterion:

1. **Local model foundation:** Connect Python to a quantized coding model through llama.cpp.
2. **Read-only agent:** Implement a bounded autonomous loop with safe file inspection and prompt budgeting.
3. **Coding and verification:** Add ripgrep search, file editing, and permitted test execution.
4. **Baseline evaluation:** Record repeatable task outcomes, SQLite traces, token usage, timing, and memory measurements.
5. **CLM-inspired context:** Add model-initiated working-note edits and context strategies.
6. **Code-aware retrieval:** Add Tree-sitter symbol indexing, ranking, and compact repository maps.
7. **Comparative benchmarking:** Evaluate baseline, retrieval-only, context-only, and combined variants, plus an external harness comparison.

See [Development Roadmap](docs/development-roadmap.md) for implementation tasks, tests, and completion criteria. The roadmap is the authoritative source for implementation order.

For a beginner-friendly guide to the implemented modules, request flow, APIs,
and tests, see the [Implemented Code Map](docs/code-map.md).

## Design documentation

- [Implemented Code Map](docs/code-map.md) — current Python modules, chat flow, and tests.
- [Architecture](docs/architecture.md) — system components, boundaries, and technology choices.
- [Agent Runtime](docs/agent-runtime.md) — agent loop, actions, safeguards, and verification.
- [Context and Memory](docs/context-and-memory.md) — editable working notes, prompt budgets, and memory strategies.
- [Repository Retrieval](docs/repository-retrieval.md) — lexical search, Tree-sitter, and bounded code snippets.
- [Benchmarking](docs/benchmarking.md) — task selection, fair comparisons, and evaluation metrics.
- [Development Roadmap](docs/development-roadmap.md) — phased implementation plan and acceptance criteria.
- [Local Model Setup](docs/local-model.md) — tested llama.cpp model, launch configuration, and HTTP smoke test.

## Local model CLI

Start `llama-server` separately with the desired GGUF model. Then run `harness`
to enter interactive mode. Each prompt is sent as its own request without
conversation history; enter `/exit` to quit. Blank lines are ignored. The CLI
prints each response and any token usage reported by the server.

```bash
harness
You > Say hello in one sentence.
```

The `python -m harness` command opens the same interactive loop. For one prompt
without entering interactive mode, use:

```bash
python -m harness chat "Say hello in one sentence."
# Optional settings: --base-url http://127.0.0.1:9931 --model MODEL.gguf
#                    --timeout 120 --max-tokens 256
```

The `harness chat PROMPT` command prints the response to stdout and token usage
to stderr when the server reports it. See [Local Model Setup](docs/local-model.md)
for the tested server configuration and details.

The initial release will focus on a single local machine and repository, not distributed infrastructure or cloud deployment.

## Project principles

- **Measure task success first.** Speed matters, but correct solutions matter more.
- **Keep the system inspectable.** Context edits, tool calls, and outcomes should be explainable and reproducible.
- **Avoid unnecessary services.** Start with a local CLI, files, and SQLite.
- **Build a baseline before optimizing.** Every added technique should justify its complexity.

## License

No license has been selected yet.
