#!/usr/bin/env python3
"""Align a trajectory's logs with the energy samples and apply the cost function.

Per trial directory (written by 04_run_trial.sh):
  samples.csv                         sampler.py output (cumulative GPU/CPU energy)
  llm_calls.jsonl                     llm_proxy.py log, one line per model call (X-Session-Id)
  jobs/*/<trial>/result.json          Harbor phase timestamps and reward
  jobs/*/<trial>/agent/trajectory.json  Terminus-2 steps (commands issued)

Inside Harbor's agent_execution window, time is split into
  model = proxy-logged model calls
  tool  = a command is executing: from the agent's keystrokes until bash shows its prompt again,
          taken from the terminal recording (agent/recording.cast)
  wait  = everything else: the agent waiting or processing between calls, and the time before the
          first and after the last model call
and the measured energy of every interval is read from the cumulative counters. Without a recording,
the whole gap between two calls counts as tool time.

Estimates use coefficients fitted on the S2 microbenchmark (--calib):
  E_model = sum over calls of variant(ctx, cached, out) + w * t_call     (w = loaded-idle power)
  E_tool + E_wait = w * (t_tool + t_wait)                                 (no tool calibration yet)

Usage:
  python align_trial.py data/traj/fix-git__summarize-on__r1 [more dirs] --calib results/calib --out results/trials.csv
Writes <dir>/aligned/{intervals.csv,trial.json,power_curve.png} and one row per trial to --out.
"""
import argparse
import datetime as dt
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energylib import cpu_cols, gpu_cols, interval_energy, load_samples, read_jsonl  # noqa: E402

VARIANTS = {
    "original": lambda c, k, o: np.c_[c, o],
    "cached_split": lambda c, k, o: np.c_[c - k, k, o],
    "cached_split_ctxout": lambda c, k, o: np.c_[c - k, k, o, c * o],
}
COLORS = {"model": "#4C72B0", "tool": "#DD8452", "wait": "#BBBBBB"}
PROMPT_MARK = "\x1b[?2004h"  # bash re-enables bracketed paste when its prompt comes back


def iso(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()


def fit_variants(calib_dir):
    d = pd.read_csv(os.path.join(calib_dir, "calls_energy.csv"))
    w = json.load(open(os.path.join(calib_dir, "params.json")))["idle_loaded_W"]["total"]
    c, k, o, t, y = (d[x].fillna(0).to_numpy(float)
                     for x in ("prompt_tokens", "cached_tokens", "completion_tokens", "t_model", "E_total"))
    coefs = {n: np.linalg.lstsq(f(c, k, o), y - w * t, rcond=None)[0] for n, f in VARIANTS.items()}
    return w, coefs


def busy_periods(tdir, mono, call_ends):
    """Command-execution periods [(t0, t1), ...] in monotonic time, from the terminal recording.

    A period starts at the agent's first keystroke and ends when bash shows its prompt again.
    The recording header has 1 s resolution, so its offset is refined by matching keystrokes to
    the model-call ends just before them (the agent types right after each response)."""
    path = os.path.join(tdir, "agent", "recording.cast")
    if not os.path.exists(path):
        return None
    lines = open(path, errors="replace").read().splitlines()
    ev = []
    for x in lines[1:]:
        try:
            ev.append(json.loads(x))
        except ValueError:  # last line cut off when a timed-out trial was killed
            continue
    base = mono(json.loads(lines[0])["timestamp"])
    keys = np.array([base + t for t, kind, _ in ev if kind == "i"])
    ends = np.asarray(sorted(call_ends))
    if len(keys) and len(ends):
        best = None
        for d in np.arange(-1.5, 1.5, 0.005):
            lag = (keys + d)[:, None] - ends[None, :]
            ok = np.where(lag >= 0, lag, np.inf).min(axis=1)
            hit = ok <= 0.3
            key = (int(hit.sum()), -float(np.median(ok[hit])) if hit.any() else 0.0)
            if best is None or key > best[0]:
                best = (key, d)
        base += best[1]
    periods, start = [], None
    for t, kind, data in ev:
        if kind == "i" and start is None:
            start = base + t
        elif kind == "o" and start is not None and PROMPT_MARK in data:
            periods.append((start, base + t))
            start = None
    if start is not None:
        periods.append((start, base + ev[-1][0]))
    return periods


def split_gap(t0, t1, periods):
    """Split a between-calls gap into tool (inside a busy period) and wait pieces."""
    pieces, cur = [], t0
    for p0, p1 in periods:
        a, b = max(p0, cur), min(p1, t1)
        if b <= a:
            continue
        if a > cur:
            pieces.append(("wait", cur, a))
        pieces.append(("tool", a, b))
        cur = b
    if t1 > cur:
        pieces.append(("wait", cur, t1))
    return pieces


def align(trial_dir, w, coefs):
    res_path = glob.glob(os.path.join(trial_dir, "jobs", "*", "*", "result.json"))[0]
    tdir = os.path.dirname(res_path)
    res = json.load(open(res_path))
    steps = json.load(open(os.path.join(tdir, "agent", "trajectory.json")))["steps"]
    samples = load_samples(os.path.join(trial_dir, "samples.csv"))
    off = float((samples["t_wall"] - samples["t_mono"]).median())
    mono = lambda wall: wall - off  # noqa: E731

    a0, a1 = (mono(iso(res["agent_execution"][k])) for k in ("started_at", "finished_at"))
    calls = [c for c in read_jsonl(os.path.join(trial_dir, "llm_calls.jsonl"))
             if c.get("status") == 200 and a0 <= c["t_start_mono"] <= a1
             and (c.get("session_id") or res["trial_name"]).startswith(res["trial_name"])]
    calls.sort(key=lambda c: c["t_start_mono"])

    periods = busy_periods(tdir, mono, [c["t_end_mono"] for c in calls])
    iv = []
    prev_end = a0
    for i, c in enumerate(calls):
        t0, t1 = c["t_start_mono"], min(c["t_end_mono"], a1)
        if t0 > prev_end:
            if i == 0:
                iv.append({"kind": "wait", "t0": prev_end, "t1": t0})
            elif periods is None:
                iv.append({"kind": "tool", "t0": prev_end, "t1": t0})
            else:
                iv += [{"kind": k, "t0": x0, "t1": x1} for k, x0, x1 in split_gap(prev_end, t0, periods)]
        iv.append({"kind": "model", "t0": t0, "t1": t1, "ctx": c["prompt_tokens"],
                   "cached": c.get("cached_tokens") or 0, "out": c["completion_tokens"]})
        prev_end = t1
    if a1 > prev_end:
        iv.append({"kind": "wait", "t0": prev_end, "t1": a1})
    for x in iv:
        e = interval_energy(samples, x["t0"], x["t1"])
        x.update(dur=x["t1"] - x["t0"], E_gpu=e["gpu"], E_cpu=e["cpu"], E_total=e["total"])
    ivdf = pd.DataFrame(iv)

    m = ivdf[ivdf["kind"] == "model"]
    ctx, cached, out, tm = (m[x].to_numpy(float) for x in ("ctx", "cached", "out", "dur"))
    t_by = ivdf.groupby("kind")["dur"].sum()
    e_by = ivdf.groupby("kind")["E_total"].sum()
    task, cfg, rep = os.path.basename(os.path.normpath(trial_dir)).split("__")
    row = {
        "trial": os.path.basename(os.path.normpath(trial_dir)), "task": task, "config": cfg, "rep": rep,
        "has_recording": periods is not None,
        # no verifier result (e.g. agent crashed on ContextLengthExceeded) counts as a failed attempt
        "reward": float((res.get("verifier_result") or {}).get("rewards", {}).get("reward") or 0.0),
        "exception": (res.get("exception_info") or {}).get("exception_type"),
        "n_model_calls": len(m), "n_steps": len(steps),
        "t_agent": a1 - a0, **{f"t_{k}": float(t_by.get(k, 0)) for k in COLORS},
        "sum_ctx": ctx.sum(), "sum_cached": cached.sum(), "sum_out": out.sum(), "sum_ctx_x_out": (ctx * out).sum(),
        "E_measured": ivdf["E_total"].sum(), "E_measured_gpu": ivdf["E_gpu"].sum(),
        "E_measured_cpu": ivdf["E_cpu"].sum(), **{f"E_{k}": float(e_by.get(k, 0)) for k in COLORS},
    }
    rest = w * (row["t_tool"] + row["t_wait"])
    for name, f in VARIANTS.items():
        est = float((f(ctx, cached, out) @ coefs[name]).sum() + w * tm.sum() + rest)
        row[f"E_est_{name}"] = est
        row[f"err_pct_{name}"] = 100 * (est - row["E_measured"]) / row["E_measured"]

    out_dir = os.path.join(trial_dir, "aligned")
    os.makedirs(out_dir, exist_ok=True)
    ivdf.to_csv(os.path.join(out_dir, "intervals.csv"), index=False)
    json.dump(row, open(os.path.join(out_dir, "trial.json"), "w"), indent=2, default=float)
    plot(samples, ivdf, res, mono, a0, row, os.path.join(out_dir, "power_curve.png"))
    return row


def plot(samples, ivdf, res, mono, a0, row, path):
    t = samples["t_mono"].to_numpy()
    eg = samples[gpu_cols(samples)].sum(axis=1).to_numpy()
    ec = samples[cpu_cols(samples)].sum(axis=1).to_numpy() if cpu_cols(samples) else np.zeros_like(eg)
    k = 3  # +-0.3 s centred difference
    tt = t[k:-k]
    pg = (eg[2 * k:] - eg[:-2 * k]) / (t[2 * k:] - t[:-2 * k])
    pc = (ec[2 * k:] - ec[:-2 * k]) / (t[2 * k:] - t[:-2 * k])
    lo = mono(iso(res["agent_setup"]["started_at"])) - 3
    end_phase = res.get("verifier") or res["agent_execution"]
    hi = mono(iso(end_phase["finished_at"])) + 3
    sel = (tt >= lo) & (tt <= hi)

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for x in ivdf.itertuples():
        ax.axvspan(x.t0 - a0, x.t1 - a0, color=COLORS[x.kind], alpha=0.35, lw=0)
    ax.plot(tt[sel] - a0, (pg + pc)[sel], color="black", lw=1.2, label="GPU + CPU")
    ax.plot(tt[sel] - a0, pg[sel], color="#4C72B0", lw=0.8, label="GPU")
    ax.plot(tt[sel] - a0, pc[sel], color="#C44E52", lw=0.8, label="CPU")
    for ph, ls in (("agent_execution", "-"), ("verifier", ":")):
        if not res.get(ph):
            continue
        for key in ("started_at", "finished_at"):
            ax.axvline(mono(iso(res[ph][key])) - a0, color="gray", lw=0.8, ls=ls)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[c], alpha=0.35) for c in COLORS]
    leg1 = ax.legend(handles, [f"{c} ({row['E_' + c] / 1000:.1f} kJ)" for c in COLORS], loc="upper left", fontsize=8)
    ax.add_artist(leg1)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("time since agent start (s)")
    ax.set_ylabel("power (W)")
    ax.set_title(f"{row['trial']}: {row['E_measured'] / 1000:.1f} kJ in {row['t_agent']:.0f} s, "
                 f"{row['n_model_calls']} model calls, reward {row['reward']}"
                 + (f" ({row['exception']})" if row["exception"] else ""), fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trials", nargs="+")
    ap.add_argument("--calib", default="results/calib")
    ap.add_argument("--out", default="results/trials.csv")
    args = ap.parse_args()
    w, coefs = fit_variants(args.calib)
    rows = []
    for d in args.trials:
        try:
            rows.append(align(d, w, coefs))
        except (IndexError, FileNotFoundError, KeyError) as e:
            print(f"skip {d}: {e!r}")
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    cols = ["trial", "reward", "n_model_calls", "t_agent", "t_model", "t_tool", "t_wait",
            "E_measured", "E_model", "E_tool", "E_wait"] + [f"err_pct_{n}" for n in VARIANTS]
    print(df[cols].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
