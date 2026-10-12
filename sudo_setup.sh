#!/usr/bin/env bash
# One-time system setup for the energy experiment (needs a sudoer).
#   sudo bash sudo_setup.sh
# Afterwards: log out of SSH (and `tmux kill-server`) so the new docker group takes effect.
set -euo pipefail
[[ $EUID -eq 0 ]] || { echo "run with: sudo bash $0"; exit 1; }
U="${SUDO_USER:?run with sudo from the account that runs the experiment}"
# Group allowed to read the RAPL counters; defaults to the primary group of that account.
G="${RAPL_GROUP:-$(id -gn "$U")}"

echo "== 1. RAPL energy counters readable by group $G (persists via udev) =="
cat > /etc/udev/rules.d/99-rapl.rules <<EOF
ACTION=="add", SUBSYSTEM=="powercap", KERNEL=="intel-rapl:*", RUN+="/bin/chgrp $G /sys%p/energy_uj", RUN+="/bin/chmod 0440 /sys%p/energy_uj"
EOF
udevadm control --reload
udevadm trigger --subsystem-match=powercap --action=add || true
for f in /sys/class/powercap/intel-rapl:*/energy_uj; do chgrp "$G" "$f"; chmod 0440 "$f"; done
ls -l /sys/class/powercap/intel-rapl:*/energy_uj

echo "== 2. Docker usable by $U =="
command -v docker >/dev/null || apt-get install -y docker.io
docker compose version >/dev/null 2>&1 || apt-get install -y docker-compose-v2
systemctl is-active --quiet docker || systemctl start docker
usermod -aG docker "$U"
id "$U"

echo "done. Now: tmux kill-server; exit; then ssh in again and run: id -nG; docker run --rm hello-world"
