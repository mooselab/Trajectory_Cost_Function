#!/usr/bin/env bash
# Start vLLM in the foreground (run inside tmux). Ready when "Application startup complete" appears.
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
mkdir -p logs
# JIT kernel builds at startup call `ninja` etc. by name; the venv is not activated otherwise.
export PATH="$PWD/.venv/bin:$PATH"

# Single node: keep gloo/NCCL bootstrap on loopback. With two NICs (public + 10.20.0.x),
# gloo otherwise picks an interface its TCP send cannot complete on and hangs for 30 min.
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
# astra2: driver reports GPU P2P as available, but it only works in one direction
# (dist_test.py: NCCL all_reduce hangs on rank 1). Route GPU<->GPU traffic via host memory
# and skip vLLM's custom all-reduce, which relies on P2P/CUDA IPC.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
# FlashInfer's sampler is JIT-compiled with the system nvcc (CUDA 12.0), which rejects flags
# from the CUDA 13 toolchain vLLM was built for. Use PyTorch's sampler instead (no JIT).
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# Exposes POST /reset_prefix_cache (server listens on 127.0.0.1 only), so each trajectory
# starts with a cold prefix cache instead of reusing the previous trial's prompt.
export VLLM_SERVER_DEV_MODE="${VLLM_SERVER_DEV_MODE:-1}"

exec .venv/bin/vllm serve "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --tensor-parallel-size "$TP" \
  --max-model-len "$MAX_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --enable-prefix-caching \
  --enable-prompt-tokens-details \
  --seed 0 \
  --disable-custom-all-reduce \
  --host 127.0.0.1 --port "$VLLM_PORT" \
  ${VLLM_EXTRA_ARGS:-} \
  2>&1 | tee "logs/vllm_$(date +%Y%m%d_%H%M).log"
