# Codex orchestration

This setup configures the Codex development assistant, not the application's
local llama.cpp runtime. Start a new Codex session in the repository:

```sh
codex --strict-config
```

Give a normal coding request. `AGENTS.md` explicitly requests that Sol delegate
bounded, nontrivial implementation to `coder`, then review the diff and results.
Small edits stay with Sol. The defaults are Sol Low and Luna High, with one
concurrent subagent. `.codex/agents/coder.toml` is discovered automatically by
the installed CLI; no legacy role table or experimental feature flag is needed.

## Verification

Verified on 2026-11-02 with Codex CLI 0.160.0:

- `codex app-server --strict-config --stdio` accepted the configuration.
- App-server `config/read` with this repository as `cwd` returned the project
  layer, `gpt-6.1-sol` / `low`, agents enabled, one concurrent subagent, and
  default subagent settings `gpt-6-luna` / `high`.
- The installed model catalog and app-server `model/list` advertised both model
  identifiers and the configured reasoning efforts.
- A read-only `codex exec --strict-config --json -s read-only` smoke test asked
  Sol to delegate inspection of `pyproject.toml` to `coder`. It completed without
  edits. Session records confirmed the parent was Sol Low and its child had
  role `coder`, model Luna, and effort High. The spawn used `fork_turns = "none"`.
- `codex debug prompt-input` included the repository delegation instructions.

Smoke-test parent session: `01a19e91-e3ed-7170-b55e-bff52b84f2c3`.
Coder session: `01a19e91-f315-79a3-9ef8-0326b7714dc1`.

In the CLI, use `/subagents` to inspect delegated threads; supported app clients
also show subagent activity. For authoritative routing evidence, inspect the
session JSONL under `~/.codex/sessions/`: the child's `session_meta` identifies
the parent and `agent_role`, and `turn_context` records `model` and `effort`.
The agent's prose alone is not proof of which model ran.

To repeat the routing smoke test:

```sh
codex exec --strict-config --json -s read-only \
  'Do not edit files. Delegate inspection of pyproject.toml to coder. Ask for the project name, wait for the result, and report it.'
```

## Limits

Delegation is an instruction-driven model decision, not a deterministic router.
The smoke test explicitly requested delegation; automatic decisions on normal
coding tasks and total token savings have not been benchmarked. Each subagent
still consumes its own instructions, tools, reasoning, and output tokens.

Project configuration requires a trusted repository (already trusted locally at
verification time). New sessions load the files; existing chats may retain their
previous settings. Client model selections, command-line overrides, account
availability, managed policy, and higher-priority instructions can affect the
effective setup. A client must support local custom agents to use `coder`.
When only explicit model selection is supported, AGENTS.md specifies Luna High
as the fallback; report unavailable routing rather than silently substituting.

Global settings and permissions were preserved. The current user configuration
selects the `priority` service tier, which is inherited by subagents and can
increase usage cost; this setup does not change that preference.

Official references: [Subagents](https://learn.chatgpt.com/docs/agent-configuration/subagents)
and [Configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference).
