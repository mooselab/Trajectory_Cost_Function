#!/usr/bin/env bash
# S1 + S2: loaded-idle baseline, microbenchmark under the sampler, then fit alpha/beta.
# Prerequisite: vLLM is running (02_serve_vllm.sh) and nobody else is using the machine.
# Optional: data/calib/idle_bare.csv measured BEFORE starting vLLM (see README).
#
#   ./03_calibrate.sh                 # full grid, ~30-60 min
#   QUICK=1 ./03_calibrate.sh         # smoke test, ~5 min
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
D=data/calib
mkdir -p "$D"

echo "== preflight: other GPU processes / load =="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
uptime
curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null || { echo "vLLM not reachable on :$VLLM_PORT"; exit 1; }

IDLE_SEC=300; BENCH_ARGS=(--cache-repeat)
if [[ "${QUICK:-0}" == 1 ]]; then
  IDLE_SEC=30; BENCH_ARGS+=(--ctx 1024,8192 --out 64,512 --reps 2)
fi

echo "== loaded-idle baseline (${IDLE_SEC}s, do not send requests) =="
$PY sampler.py -o "$D/idle_loaded.csv" --duration "$IDLE_SEC"

echo "== microbenchmark under sampler =="
$PY sampler.py -o "$D/samples.csv" & SPID=$!
trap 'kill $SPID 2>/dev/null || true' EXIT
sleep 2
$PY microbench.py -o "$D/calls.jsonl" --base-url "http://127.0.0.1:$VLLM_PORT/v1" "${BENCH_ARGS[@]}"
sleep 2
kill $SPID; wait $SPID || true
trap - EXIT

IDLE_BARE=()
[[ -f "$D/idle_bare.csv" ]] && IDLE_BARE=(--idle-bare "$D/idle_bare.csv")
$PY calibrate.py --samples "$D/samples.csv" --calls "$D/calls.jsonl" \
  --idle-loaded "$D/idle_loaded.csv" ${IDLE_BARE[@]+"${IDLE_BARE[@]}"} --out-dir results/calib
