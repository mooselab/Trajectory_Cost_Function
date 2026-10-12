#!/usr/bin/env bash
# Run the oracle (reference solution, no model, no GPU) on candidate tasks. Pulls each task's
# Docker image ahead of the measured runs and flags tasks whose environment or tests are broken.
#
#   bash 05_oracle_check.sh                      # tasks/candidates.txt
#   bash 05_oracle_check.sh tasks/other.txt
set -euo pipefail
cd "$(dirname "$0")"
export PATH="$HOME/.local/bin:$PATH"

LIST="${1:-tasks/candidates.txt}"
JOB="oracle_$(date +%Y%m%d_%H%M)"
ARGS=()
while read -r t; do [[ -n "$t" ]] && ARGS+=(-i "$t"); done < "$LIST"

nice -n 10 harbor run -d "${DATASET:-terminal-bench@2.0}" "${ARGS[@]}" -a oracle \
  -n "${N_CONCURRENT:-2}" -o data/oracle --job-name "$JOB" --yes -q

.venv/bin/python - "data/oracle/$JOB" <<'EOF'
import glob, json, os, sys
rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*", "result.json"))):
    r = json.load(open(f))
    reward = (r.get("verifier_result") or {}).get("rewards", {}).get("reward")
    exc = (r.get("exception_info") or {}).get("exception_type")
    rows.append((r["task_name"], reward, exc))
ok = [t for t, rw, e in rows if rw == 1.0 and not e]
print(f"\n{len(ok)}/{len(rows)} tasks pass with the oracle")
for t, rw, e in rows:
    if t not in ok:
        print(f"  FAIL {t}: reward={rw} exception={e}")
open("tasks/oracle_ok.txt", "w").write("\n".join(ok) + "\n")
print("passing tasks written to tasks/oracle_ok.txt")
EOF
