# Local model setup

This records the first local inference smoke test for issue #3. The model ran on the 16 GiB Apple Silicon target and answered a chat request through `llama-server` without a cloud API.

## Tested setup

- Runtime: `llama.cpp` 0.7.0, build 11669, commit `620d2225c`.
- Server binary: `/opt/homebrew/bin/llama-server`.
- Machine memory: 16 GiB unified memory.
- Model: Qwen2.5-Coder-7B-Instruct, 7B parameters, Q4_K_M quantization, GGUF format.
- Model file: `$HOME/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf` (4,683,074,336 bytes; about 4.68 GB).
- Upstream model reference and license: [Qwen2.5-Coder-7B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF), which lists Apache-2.0. The original download URL and repository revision were not captured, so confirm the file's source if it came from a mirror or republisher.

The weights are stored in `$HOME/models`, outside this repository. `.gitignore` also excludes root-level `models/` and all `*.gguf` files as a safeguard.

## Launch command and configuration

The running process was observed with these arguments:

```sh
llama-server -m /Users/regannguyen/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf -c 8192
```

Portable equivalent for a checkout on another machine:

```sh
llama-server -m "$HOME/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf" -c 8192
```

`-c 8192` sets an 8,192-token context window. The server used its default host (`127.0.0.1`) and port (`9931`). CPU thread count, batch sizes, and GPU-layer settings were not overridden; they use the installed version's defaults. This bounds context, not total system RAM.

## HTTP smoke test

With the server running, this request was sent directly to its local chat endpoint:

```sh
curl http://127.0.0.1:9931/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf",
    "messages": [
      {"role": "user", "content": "Say hello in one sentence."}
    ],
    "max_tokens": 64
  }'
```

The server returned `Hello!` with `finish_reason: stop`. The response reported 3 completion tokens, 78.50 prompt tokens/second, and 20.89 generated tokens/second. A separate UI response showed 22.63 tokens/second; these are short single-request observations, not benchmark results.

The Python CLI can send a single non-streaming request:

```sh
python -m harness chat "Say hello in one sentence."
```

The one-shot command defaults to `http://127.0.0.1:9931`, model
`Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf`, a 120-second timeout, and 256 maximum
tokens. Override them with `--base-url`, `--model`, `--timeout`, and
`--max-tokens`. The response is written to stdout; token counts, when reported,
are written to stderr. HTTP, connection, timeout, and malformed-response errors
are shown as concise CLI errors.

## Interactive CLI

Install the package with `python -m pip install -e '.[dev]'`, start
`llama-server` separately as shown above, then run `harness` (or
`python -m harness`). The CLI prompts with `You > `; each nonblank line is
sent as a separate request with no conversation history. Enter `/exit` to quit.
End of input and Ctrl+C also exit cleanly. Client errors are printed to stderr
and the prompt remains available for another request.

### Python CLI smoke test — 2026-11-02

With `llama-server` running using the configuration above, this single local
request succeeded:

```sh
.venv/bin/python -m harness chat 'Say hello in one sentence.' --max-tokens 64
```

It printed `Hello!` and reported 35 prompt tokens, 3 completion tokens, and 38
total tokens. This records one successful smoke test, not repeated reliability
or benchmark evidence.

After the earlier curl request, `ps` reported the server at 3,667,024 KiB RSS (about 3.5 GiB) at roughly 13 minutes of uptime. This is a point-in-time process RSS value, not peak or total system memory usage. No startup failure was observed.
