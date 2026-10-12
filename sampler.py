#!/usr/bin/env python3
"""Sample cumulative GPU (NVML) and CPU (RAPL/powercap) energy counters to CSV.

Columns:
  t_mono, t_wall          CLOCK_MONOTONIC and wall-clock seconds
  gpu{i}_j, gpu{i}_w      cumulative GPU energy (J) and instantaneous power (W)
  gpu{i}_util, gpu{i}_mem_mib   GPU utilization (%) and memory used (MiB)
  cpu_pkg{n}_j            cumulative CPU package energy (J), wrap-around corrected
  sub_{name}_{ids}_j      RAPL sub-domains (e.g. core); already included in the package,
                          so analysis must NOT add them to the total
  cpu_util_pct, ram_used_mib    whole-machine CPU utilization (%) and RAM used (MiB), Linux only

Usage:
  python sampler.py -o samples.csv               # until Ctrl-C / SIGTERM
  python sampler.py -o idle.csv --duration 300   # fixed duration
"""
import argparse
import csv
import glob
import os
import signal
import sys
import time


def log(msg):
    print(f"[sampler] {msg}", file=sys.stderr, flush=True)


def open_gpus():
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception as e:  # no driver / no package
        log(f"NVML unavailable: {e}")
        return None, []
    handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(pynvml.nvmlDeviceGetCount())]
    for i, h in enumerate(handles):
        try:
            pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        except pynvml.NVMLError as e:
            log(f"GPU{i}: total-energy counter not supported ({e})")
            return None, []
    log(f"NVML: {len(handles)} GPU(s)")
    return pynvml, handles


class RaplDomain:
    def __init__(self, path, col):
        self.path = path
        self.col = col
        with open(os.path.join(path, "max_energy_range_uj")) as f:
            self.max_range = int(f.read())
        self.last = self._raw()  # raises PermissionError if not readable
        self.acc = 0

    def _raw(self):
        with open(os.path.join(self.path, "energy_uj")) as f:
            return int(f.read())

    def read_j(self):
        raw = self._raw()
        d = raw - self.last
        if d < 0:  # counter wrapped
            d += self.max_range
        self.acc += d
        self.last = raw
        return self.acc / 1e6


def open_rapl():
    domains = []
    for p in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
        ids = os.path.basename(p).split(":")[1:]
        try:
            with open(os.path.join(p, "name")) as f:
                name = f.read().strip().replace("-", "")
        except OSError:
            continue
        col = f"cpu_pkg{ids[0]}_j" if len(ids) == 1 else f"sub_{name}_{'_'.join(ids)}_j"
        try:
            domains.append(RaplDomain(p, col))
            log(f"RAPL: {os.path.basename(p)} ({name}) -> {col}")
        except PermissionError:
            log(f"RAPL: {os.path.basename(p)} not readable -- ask the admin for read access to energy_uj")
        except OSError as e:
            log(f"RAPL: {os.path.basename(p)} skipped ({e})")
    if not domains:
        log("RAPL: no readable domains, CPU energy will be missing")
    return domains


def read_cpu_times():
    """(total, idle) jiffies from /proc/stat; None where /proc is unavailable."""
    try:
        with open("/proc/stat") as f:
            v = [int(x) for x in f.readline().split()[1:]]
    except OSError:
        return None
    return sum(v), v[3] + v[4]  # idle + iowait


def read_ram_used_mib():
    m = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, val = line.split(":", 1)
            m[k] = int(val.split()[0])
    return (m["MemTotal"] - m["MemAvailable"]) / 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until killed")
    args = ap.parse_args()

    nvml, gpus = open_gpus()
    rapl = open_rapl()

    stop = False

    def on_signal(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    last_cpu = read_cpu_times()
    has_proc = last_cpu is not None

    header = ["t_mono", "t_wall"]
    for i in range(len(gpus)):
        header += [f"gpu{i}_j", f"gpu{i}_w", f"gpu{i}_util", f"gpu{i}_mem_mib"]
    header += [d.col for d in rapl]
    if has_proc:
        header += ["cpu_util_pct", "ram_used_mib"]

    period = 1.0 / args.hz
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        t_begin = next_t = time.monotonic()
        last_flush = t_begin
        n = 0
        while not stop:
            row = [f"{time.monotonic():.6f}", f"{time.time():.6f}"]
            for h in gpus:
                row.append(f"{nvml.nvmlDeviceGetTotalEnergyConsumption(h) / 1000:.3f}")
                row.append(f"{nvml.nvmlDeviceGetPowerUsage(h) / 1000:.3f}")
                row.append(str(nvml.nvmlDeviceGetUtilizationRates(h).gpu))
                row.append(f"{nvml.nvmlDeviceGetMemoryInfo(h).used / 2**20:.0f}")
            for d in rapl:
                row.append(f"{d.read_j():.6f}")
            if has_proc:
                cur = read_cpu_times()
                dt, di = cur[0] - last_cpu[0], cur[1] - last_cpu[1]
                row.append(f"{100 * (1 - di / dt):.1f}" if dt else "")
                row.append(f"{read_ram_used_mib():.0f}")
                last_cpu = cur
            w.writerow(row)
            n += 1
            now = time.monotonic()
            if now - last_flush > 1.0:
                f.flush()
                last_flush = now
            if args.duration and now - t_begin >= args.duration:
                break
            next_t += period
            time.sleep(max(0.0, next_t - time.monotonic()))
    log(f"wrote {n} samples to {args.out}")


if __name__ == "__main__":
    main()
