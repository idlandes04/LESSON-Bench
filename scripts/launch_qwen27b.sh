#!/usr/bin/env bash
# Launch llama-server with Qwen3.5-27B from LM Studio model files.
# Uses Metal (Apple Silicon GPU) for inference.
#
# Think/nothink is toggled per-request via extra_body, NOT server flags.
# This server supports both modes simultaneously.
#
# Usage:
#   ./scripts/launch_qwen27b.sh          # port 8080 (default)
#   ./scripts/launch_qwen27b.sh 8090     # custom port

set -euo pipefail

PORT="${1:-8080}"
MODEL_PATH="$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf"
LLAMA_SERVER="$(dirname "$0")/../llama.cpp/build/bin/llama-server"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$LLAMA_SERVER" ]; then
    echo "ERROR: llama-server not found at $LLAMA_SERVER"
    exit 1
fi

echo "Starting llama-server on port $PORT with Qwen3.5-27B-Q4_K_M..."
echo "Model: $MODEL_PATH"
echo "GPU layers: 99 (Metal)"
echo ""

exec "$LLAMA_SERVER" \
    --model "$MODEL_PATH" \
    --jinja \
    --reasoning-format deepseek \
    --port "$PORT" \
    --ctx-size 16384 \
    --n-gpu-layers 99 \
    --threads 8 \
    --parallel 1 \
    --flash-attn on \
    --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0
