# Repository instructions

This is a Python, local-first coding-agent harness using llama.cpp. Follow
`docs/development-roadmap.md` for implementation order and consult the relevant
subsystem document before changing its behavior. Keep changes small and preserve
existing architecture, conventions, and unrelated work.

## Orchestration

The main agent (GPT-6.1 Sol) owns requirements, planning, architectural decisions,
review, and the final response. These instructions explicitly request automatic
delegation of well-defined, nontrivial implementation work to the `coder` agent
(GPT-6 Luna, high reasoning); the user need not request delegation each time.

- Handle trivial edits directly when delegation would cost more than it saves.
- Resolve ambiguity and choose the approach before delegating. Avoid reading all
  implementation files first: give the coder relevant paths and known findings.
- Delegate one bounded task at a time, with scope, acceptance criteria, relevant
  constraints, and required checks. Use `coder` when custom agent selection is
  available; otherwise explicitly select `gpt-6-luna` with `high` reasoning and
  include the coder instructions. Use a fresh context rather than copying the
  entire conversation when supported.
- Do not spawn extra explorers or reviewers by default. The coder must not
  delegate further. Reuse the same coder for a focused correction instead of
  restarting discovery or entering repeated delegation loops. Sol takes over
  when escalation or repeated failure makes that more efficient.
- Sol reviews the diff and check results against the acceptance criteria before
  reporting completion. Inspect relevant code and rerun checks only when needed
  to resolve a concern; do not duplicate the coder's entire investigation.
- Use deterministic search, lint, and tests for verification. Request concise
  findings and failure excerpts, not entire files or successful command logs.
- Do not use Max or Ultra reasoning by default. Never silently substitute a
  different model if Luna is unavailable; report the limitation.

## Implementation and verification

Implement the smallest complete change, with relevant tests for changed behavior.
Do not add speculative abstractions, dependencies, or unrelated cleanup. Preserve
user edits. Escalate unclear requirements and architectural decisions to Sol.

Use the existing virtual environment when available. Relevant project checks:

```sh
python -m pytest
ruff check src tests
ruff format --check src tests
```

Start with checks relevant to the change. Report changed paths, behavior, checks
and results, and unresolved issues briefly. Distinguish checks run from checks
not run, and never claim efficiency or correctness without evidence.

## Implemented code map

Keep `docs/code-map.md` aligned with the implemented Python package. Whenever a
change alters source modules, functions, classes, signatures, responsibilities,
or call flow, update the code map in the same change. Before completing source
work, verify its descriptions and links against the implementation. Document
implemented behavior only; label planned behavior as future work. The map is
maintained manually and is not updated by an automatic generator or watcher.
