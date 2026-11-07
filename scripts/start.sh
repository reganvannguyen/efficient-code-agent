#!/usr/bin/env bash
set -euo pipefail

PORT="${LLAMA_PORT:-9931}"
MODEL="${LLAMA_MODEL:-$HOME/models/Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf}"
SERVER_BIN="${LLAMA_SERVER_BIN:-$(which llama-server 2>/dev/null || echo "/opt/homebrew/bin/llama-server")}"
CTX_SIZE="${LLAMA_CTX:-8192}"
LOG_FILE="/tmp/llama-server.log"

STARTED_SERVER=0
SERVER_PID=""

cleanup() {
    if [ "$STARTED_SERVER" -eq 1 ] && [ -n "$SERVER_PID" ]; then
        echo ""
        echo "Stopping llama-server (PID: $SERVER_PID)..."
        kill "$SERVER_PID" 2>/dev/null || true
        wait "$SERVER_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

# 1. Check if server is already responding
if curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "Local llama-server is already running on port $PORT."
else
    if [ ! -x "$SERVER_BIN" ]; then
        echo "Error: llama-server binary not found at $SERVER_BIN" >&2
        exit 1
    fi
    if [ ! -f "$MODEL" ]; then
        echo "Error: Model file not found at $MODEL" >&2
        exit 1
    fi

    echo "Starting llama-server with $(basename "$MODEL") on port $PORT..."
    "$SERVER_BIN" -m "$MODEL" -c "$CTX_SIZE" --port "$PORT" >"$LOG_FILE" 2>&1 &
    SERVER_PID=$!
    STARTED_SERVER=1

    echo "Waiting for model to load..."
    RETRIES=30
    until curl -s "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do
        sleep 1
        RETRIES=$((RETRIES - 1))
        if [ "$RETRIES" -le 0 ]; then
            echo "Error: Server failed to start within 30 seconds. Check logs at $LOG_FILE" >&2
            exit 1
        fi
    done
    echo "Model ready!"
fi

# 2. Launch harness CLI
echo "Launching harness..."
uv run python -m harness "$@"

