# Shared settings, sourced by the other scripts. Edit here, not in the scripts.

# Model weights live on the NFS share (root partition has ~400 GB free).
# If /mnt/data is not writable at top level, ask the admin for /mnt/data/$USER.
DATA_ROOT="${DATA_ROOT:-/mnt/data/$USER}"
export HF_HOME="$DATA_ROOT/hf"

MODEL="${MODEL:-Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8}"
SERVED_NAME="${SERVED_NAME:-agent-llm}"
TP="${TP:-2}"                    # tensor parallel over both L4s
MAX_LEN="${MAX_LEN:-32768}"   # 64K leaves no KV-cache headroom on 2x L4 with FP8 30B
GPU_UTIL="${GPU_UTIL:-0.90}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${PROXY_PORT:-8001}"

PY="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.venv/bin/python"
