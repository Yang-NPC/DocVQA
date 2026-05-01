#!/usr/bin/env bash
set -euo pipefail

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-VL-8B-Thinking}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
PID_FILE="${PID_FILE:-/tmp/vllm_qwen3vl_thinking.pid}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_qwen3vl_thinking.log}"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE")"
  if kill -0 "$old_pid" 2>/dev/null; then
    echo "Stopping existing vLLM process from $PID_FILE: $old_pid"
    kill "$old_pid" || true
    sleep 5
    kill -9 "$old_pid" 2>/dev/null || true
  fi
fi

port_pids="$(lsof -ti tcp:"$PORT" 2>/dev/null || true)"
if [[ -n "$port_pids" ]]; then
  echo "Stopping process(es) on port $PORT: $port_pids"
  kill $port_pids || true
  sleep 5
  kill -9 $port_pids 2>/dev/null || true
fi

rm -f "$LOG_FILE"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

nohup python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_ID" \
  --served-model-name "$MODEL_ID" \
  --dtype bfloat16 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
  --reasoning-parser qwen3 \
  --host "$HOST" \
  --port "$PORT" \
  --trust-remote-code \
  > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Started vLLM thinking server PID: $(cat "$PID_FILE")"
echo "Model: $MODEL_ID"
echo "URL: http://$HOST:$PORT/v1"
echo "Log: $LOG_FILE"
