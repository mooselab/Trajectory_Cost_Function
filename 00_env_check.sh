#!/usr/bin/env bash
# Probe what this server allows for the experiment. Read-only; needs no sudo.
# Output: env_check_<timestamp>.txt (send this file back for review).
out="env_check_$(date +%Y%m%d_%H%M).txt"
{
echo "== containers =="
for b in docker podman apptainer singularity; do printf '%-12s %s\n' "$b" "$(command -v $b || echo -)"; done
docker info 2>&1 | head -3
echo "groups: $(id -Gn)"
echo "subuid: $(grep "^$USER:" /etc/subuid 2>/dev/null || echo none)"
echo "subgid: $(grep "^$USER:" /etc/subgid 2>/dev/null || echo none)"
ls -l /usr/bin/newuidmap /usr/bin/newgidmap 2>&1

echo; echo "== RAPL =="
for d in /sys/class/powercap/intel-rapl:*; do
  printf '%s name=%s ' "$d" "$(cat $d/name 2>/dev/null)"
  if cat "$d/energy_uj" >/dev/null 2>&1; then echo "readable"; else echo "NOT readable"; fi
done
echo "perf_event_paranoid=$(cat /proc/sys/kernel/perf_event_paranoid)"

echo; echo "== GPU =="
nvidia-smi --query-gpu=index,name,pstate,power.draw,power.limit,memory.used,memory.total --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi topo -m

echo; echo "== other users / services =="
uptime
systemctl status ollama --no-pager 2>&1 | head -5
who
command -v sinfo >/dev/null && { sinfo; squeue; }

echo; echo "== disk =="
df -h "$HOME" /mnt/data
ls -ld /mnt/data /mnt/data/* 2>&1 | head -20
touch "/mnt/data/.w_$USER" 2>/dev/null && { echo "/mnt/data writable at top level"; rm -f "/mnt/data/.w_$USER"; } || echo "/mnt/data NOT writable at top level"

echo; echo "== tools =="
for b in uv tmux git curl python3; do printf '%-8s %s\n' "$b" "$(command -v $b || echo -)"; done
python3 -c "import venv, ensurepip" >/dev/null 2>&1 && echo "python venv: ok" || echo "python venv: missing (use uv)"
} 2>&1 | tee "$out"
echo; echo "saved to $out"
