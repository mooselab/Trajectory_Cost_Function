#!/usr/bin/env bash
# S3: run ONE Terminal-Bench trial (one trajectory) under the energy sampler and LLM logging proxy.
#
#   bash 04_run_trial.sh <task> <config> <rep>
#   bash 04_run_trial.sh fix-git summarize-on 1
#
# config = Terminus-2 context policy:
#   summarize-on   agent default: summarize when < 8000 free tokens (~24.7K context at MAX_LEN 32K)
#   summarize-off  never summarize
#   summarize-8k   summarize once the context exceeds ~8K tokens (aggressive)
# Prerequisites: vLLM running (02_serve_vllm.sh), `docker` usable without sudo, nobody else on the GPUs.
# Output: data/traj/<task>__<config>__r<rep>/ with samples.csv, llm_calls.jsonl, jobs/ (Harbor), logs.
set -euo pipefail
cd "$(dirname "$0")"
source config.sh
export PATH="$HOME/.local/bin:$PATH"

TASK=$1; CFG=$2; REP=$3
case "$CFG" in
  summarize-on)  SUM=true;  THR=8000 ;;
  summarize-off) SUM=false; THR=8000 ;;
  summarize-8k)  SUM=true;  THR=$((MAX_LEN - 8192)) ;;
  *) echo "unknown config: $CFG"; exit 1 ;;
esac
DATASET="${DATASET:-terminal-bench@2.0}"
# Empty = model default from generation_config.json (Qwen3-Coder: temperature 0.7, top_p 0.8, top_k 20),
# so repetitions differ and retry budget can be analysed post hoc. Set TEMPERATURE=0 for near-deterministic runs.
TEMPERATURE="${TEMPERATURE:-}"
TEMP_ARG=()
[[ -n "$TEMPERATURE" ]] && TEMP_ARG=(--ak "temperature=$TEMPERATURE")
ID="${TASK}__${CFG}__r${REP}"
D="data/traj/$ID"
[[ -e "$D" ]] && { echo "$D already exists; delete it to rerun"; exit 1; }
mkdir -p "$D"

echo "== preflight =="
docker info >/dev/null 2>&1 || { echo "docker not usable by $USER (run sudo_setup.sh, then log in again)"; exit 1; }
curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null || { echo "vLLM not reachable on :$VLLM_PORT"; exit 1; }
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader | tee "$D/gpu_procs.txt"
uptime | tee "$D/uptime.txt"

# Cold prefix cache, so this trial is not cheaper just because the previous one left its prompt cached.
curl -sf -X POST "http://127.0.0.1:$VLLM_PORT/reset_prefix_cache" > "$D/cache_reset.txt" \
  || echo "WARN: prefix-cache reset failed (is VLLM_SERVER_DEV_MODE=1?)" | tee "$D/cache_reset.txt"

$PY llm_proxy.py --upstream "http://127.0.0.1:$VLLM_PORT" --port "$PROXY_PORT" --log "$D/llm_calls.jsonl" \
  > "$D/proxy.log" 2>&1 & PXID=$!
$PY sampler.py -o "$D/samples.csv" 2> "$D/sampler.log" & SPID=$!
trap 'kill $SPID $PXID 2>/dev/null || true' EXIT
sleep 3

echo "== harbor: $ID =="
date +%s.%N > "$D/t_start_wall"
set +e
harbor run -d "$DATASET" -i "$TASK" -a terminus-2 \
  -m "hosted_vllm/$SERVED_NAME" \
  --ak "api_base=http://127.0.0.1:$PROXY_PORT/v1" \
  --ak "model_info={\"max_input_tokens\":$MAX_LEN,\"max_output_tokens\":4096,\"input_cost_per_token\":0,\"output_cost_per_token\":0}" \
  ${TEMP_ARG[@]+"${TEMP_ARG[@]}"} \
  --ak "session_id_headers=[\"X-Session-Id\"]" \
  --ak "enable_summarize=$SUM" \
  --ak "proactive_summarization_threshold=$THR" \
  -n 1 -o "$D/jobs" --job-name "$ID" --yes 2>&1 | tee "$D/harbor.log"
RC=${PIPESTATUS[0]}
set -e
date +%s.%N > "$D/t_end_wall"
sleep 3

kill $SPID $PXID; wait $SPID $PXID 2>/dev/null || true
trap - EXIT
echo "harbor exit=$RC; $(wc -l < "$D/llm_calls.jsonl") model calls logged; outputs in $D"
