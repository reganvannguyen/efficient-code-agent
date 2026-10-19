# Efficient Code Agent

**A lightweight, local-first coding agent harness designed to make small language models more effective on everyday hardware.**

> **Status:** Early development / architecture planning. The capabilities and commands below describe the intended design, not yet implemented functionality.

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

Use **ripgrep** for fast lexical search and **Tree-sitter** for syntax-aware parsing. Planned retrieval features include symbol lookup, lightweight repository maps, and targeted code snippets.

The aim is to supply the model with **the right code**, not the entire repository. Embedding-based retrieval is intentionally deferred until benchmarks justify the additional memory and complexity.

### 3. Persistent task memory

Store inspectable, agent-editable notes in **Markdown**, initially including a working context and task findings. The harness—not the model directly—handles file reads, writes, validation, and persistence.

Full execution history stays separate from the condensed working context, so important evidence can be recovered without including every prior tool result in each prompt.

### 4. Verification-driven agent loop

A Python runtime will coordinate repeated **inspect → plan → edit → test → evaluate** cycles. The agent may request repository searches, file reads and edits, or test commands. The harness executes approved tools and returns their results to the model.

Planned protections include workspace boundaries, command validation, step/time budgets, and explicit completion criteria.

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

## Evaluation strategy

The project will compare the **same local model on the same coding tasks** using:

1. A minimal baseline harness.
2. A harness with code-aware retrieval.
3. A harness with proactive context editing.
4. A combined optimized harness.

Primary metric: **coding-task success rate**, evaluated with tests or task-specific checks.

Secondary metrics: token usage, agent steps, elapsed time, and peak memory. Controlled comparisons and ablation tests will help identify which techniques provide measurable benefits.

**No accuracy or efficiency improvements are claimed until benchmarks have been run.**

## Development roadmap

- [ ] Establish Python package and CLI entry point.
- [ ] Connect to a local llama.cpp server.
- [ ] Build a minimal tool-using coding-agent loop.
- [ ] Add workspace safeguards and test execution.
- [ ] Create a reproducible baseline task suite.
- [ ] Implement Markdown working context and context-edit tools.
- [ ] Implement ripgrep search and Tree-sitter repository mapping.
- [ ] Add token-budget safeguards.
- [ ] Add SQLite tracing and resource metrics.
- [ ] Run comparative benchmarks and document findings.

## Example CLI (proposed)

```bash
# Illustrative interface; not implemented yet
harness run "Find and fix the failing authentication test" --repo ./example-project
```

The initial release will focus on a single local machine and repository, not distributed infrastructure or cloud deployment.

## Project principles

- **Measure task success first.** Speed matters, but correct solutions matter more.
- **Keep the system inspectable.** Context edits, tool calls, and outcomes should be explainable and reproducible.
- **Avoid unnecessary services.** Start with a local CLI, files, and SQLite.
- **Build a baseline before optimizing.** Every added technique should justify its complexity.

## License

No license has been selected yet.
