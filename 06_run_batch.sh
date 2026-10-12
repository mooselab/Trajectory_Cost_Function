#!/usr/bin/env bash
# S3 batch: loaded-idle baseline, then every task x config x repetition via 04_run_trial.sh.
# Resumable: finished trials (t_end_wall present) are skipped; unfinished ones are moved aside and rerun.
# Order: repetition-major, tasks shuffled per repetition (fixed seed), so slow drift is not
# confounded with any one task or config.
#
#   bash 06_run_batch.sh <task_list> "<configs>" <reps>
#   bash 06_run_batch.sh tasks/pilot.txt summarize-on 1
#   bash 06_run_batch.sh tasks/final.txt "summarize-on summarize-off" 3
set -euo pipefail
cd "$(dirname "$0")"
source config.sh

LIST=$1; CONFIGS=$2; REPS=$3
B="data/traj/_batches/$(date +%Y%m%d_%H%M)"
mkdir -p "$B"

if systemctl is-active --quiet ollama || pgrep -x llama-server >/dev/null; then
  [[ "${ALLOW_BUSY:-0}" == 1 ]] || { echo "Ollama is running: stop it first (sudo systemctl stop ollama) or set ALLOW_BUSY=1"; exit 1; }
fi
curl -sf "http://127.0.0.1:$VLLM_PORT/v1/models" >/dev/null || { echo "vLLM not reachable on :$VLLM_PORT"; exit 1; }

mapfile -t TASKS < <(grep -v '^[[:space:]]*$' "$LIST")
PLAN=()
for rep in $(seq 1 "$REPS"); do
  for t in $(printf '%s\n' "${TASKS[@]}" | shuf --random-source=<(yes "$rep")); do
    for c in $CONFIGS; do PLAN+=("$t $c $rep"); done
  done
done
printf '%s\n' "${PLAN[@]}" > "$B/plan.txt"

echo "== loaded-idle baseline (${IDLE_SEC:-120}s) ==" | tee "$B/progress.log"
$PY sampler.py -o "$B/idle_loaded.csv" --duration "${IDLE_SEC:-120}"

n=${#PLAN[@]}; i=0
for item in "${PLAN[@]}"; do
  i=$((i + 1))
  read -r t c r <<< "$item"
  D="data/traj/${t}__${c}__r${r}"
  if [[ -f "$D/t_end_wall" ]]; then echo "[$i/$n] skip $t $c r$r (done)" | tee -a "$B/progress.log"; continue; fi
  [[ -e "$D" ]] && mv "$D" "${D}.incomplete_$(date +%s)"
  echo "[$i/$n] $(date +%H:%M) $t $c r$r" | tee -a "$B/progress.log"
  bash 04_run_trial.sh "$t" "$c" "$r" > "$B/${t}__${c}__r${r}.log" 2>&1 \
    || echo "  trial script failed, see $B/${t}__${c}__r${r}.log" | tee -a "$B/progress.log"
  sleep "${GAP_SEC:-30}"
done
echo "batch done: $B" | tee -a "$B/progress.log"
