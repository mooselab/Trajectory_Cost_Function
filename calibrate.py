#!/usr/bin/env python3
"""Fit the model-call term of the cost function from microbench data.

  E_call = alpha * context + beta * output + t_model * overhead_model

overhead_model is fixed to the loaded-idle power (vLLM up, no requests), because t_model is
almost determined by context and output and a free three-way fit is collinear. alpha and beta
are then fit by least squares on the remaining (dynamic) energy. The free fit is also reported
for comparison.

Outputs in --out-dir: params.json, calls_energy.csv, calib_scatter.png, energy_vs_tokens.png

Usage:
  python calibrate.py --samples samples.csv --calls calls.jsonl \
      --idle-loaded idle_loaded.csv [--idle-bare idle_bare.csv] --out-dir results/calib
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from energylib import interval_energy, load_samples, mean_power, read_jsonl  # noqa: E402


def r2(y, pred):
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True)
    ap.add_argument("--calls", required=True)
    ap.add_argument("--idle-loaded", required=True)
    ap.add_argument("--idle-bare")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    p_loaded = mean_power(load_samples(args.idle_loaded))
    p_bare = mean_power(load_samples(args.idle_bare)) if args.idle_bare else None
    print(f"idle power, model loaded: total {p_loaded['total']:.1f} W "
          f"(gpu {p_loaded['gpu']:.1f}, cpu {p_loaded['cpu']:.1f})")
    if p_bare:
        print(f"idle power, bare:         total {p_bare['total']:.1f} W "
              f"(gpu {p_bare['gpu']:.1f}, cpu {p_bare['cpu']:.1f})")
    if p_loaded["cpu"] == 0:
        print("WARNING: no CPU energy in samples (RAPL not readable); results are GPU-only")

    samples = load_samples(args.samples)
    rows = []
    for c in read_jsonl(args.calls):
        e = interval_energy(samples, c["t_start_mono"], c["t_end_mono"])
        rows.append({**c, "t_model": c["t_end_mono"] - c["t_start_mono"],
                     "E_gpu": e["gpu"], "E_cpu": e["cpu"], "E_total": e["total"]})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out_dir, "calls_energy.csv"), index=False)

    miss = df[df["kind"] == "miss"]
    ctx = miss["prompt_tokens"].to_numpy(float)
    out = miss["completion_tokens"].to_numpy(float)
    t = miss["t_model"].to_numpy(float)
    y = miss["E_total"].to_numpy(float)
    omega = p_loaded["total"]

    # constrained fit: overhead_model = loaded idle power
    (alpha, beta), *_ = np.linalg.lstsq(np.column_stack([ctx, out]), y - omega * t, rcond=None)
    pred = alpha * ctx + beta * out + omega * t
    mape = float(np.mean(np.abs(pred - y) / y) * 100)

    # free fit, for comparison (expect unstable coefficients)
    X3 = np.column_stack([ctx, out, t])
    free, *_ = np.linalg.lstsq(X3, y, rcond=None)
    cond = float(np.linalg.cond(X3 / X3.std(axis=0)))

    # repeatability: coefficient of variation per (ctx, out) cell
    cv = miss.groupby(["ctx_target", "out_target"])["E_total"].agg(lambda s: s.std() / s.mean())

    params = {
        "alpha_J_per_context_token": float(alpha),
        "beta_J_per_output_token": float(beta),
        "overhead_model_W": omega,
        "overhead_wait_W": omega,
        "idle_loaded_W": p_loaded,
        "idle_bare_W": p_bare,
        "fit": {"n_calls": int(len(miss)), "R2": float(r2(y, pred)), "MAPE_pct": mape},
        "free_fit": {"alpha": float(free[0]), "beta": float(free[1]), "overhead_model_W": float(free[2]),
                     "R2": float(r2(y, X3 @ free)), "condition_number_standardized": cond},
        "repeatability_cv": {"max": float(cv.max()), "median": float(cv.median())},
    }

    hit = df[df["kind"] == "hit"]
    if len(hit):
        paired = hit.merge(miss, on=["ctx_target", "out_target", "rep"], suffixes=("_hit", "_miss"))
        params["cache_hit"] = {
            "n": int(len(paired)),
            "median_cached_fraction": float((paired["cached_tokens_hit"] / paired["prompt_tokens_hit"]).median()),
            "median_energy_ratio_hit_over_miss": float((paired["E_total_hit"] / paired["E_total_miss"]).median()),
        }

    with open(os.path.join(args.out_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(json.dumps(params, indent=2))

    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(y, pred, s=12)
    lim = [0, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, color="gray", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured energy per call (J)")
    ax.set_ylabel("estimated energy per call (J)")
    ax.set_title(f"calibration: R²={params['fit']['R2']:.3f}, MAPE={mape:.1f}%")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "calib_scatter.png"), dpi=150)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for o, g in miss.groupby("out_target"):
        m = g.groupby("ctx_target")["E_total"].agg(["mean", "std"])
        axes[0].errorbar(m.index, m["mean"], yerr=m["std"], marker="o", capsize=3, label=f"output={o}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xlabel("context tokens")
    axes[0].set_ylabel("energy per call (J)")
    axes[0].legend()
    for o, g in miss.groupby("out_target"):
        axes[1].scatter(g["t_model"], g["E_total"], s=10, label=f"output={o}")
    axes[1].set_xlabel("call duration (s)")
    axes[1].set_ylabel("energy per call (J)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "energy_vs_tokens.png"), dpi=150)
    print(f"figures and params written to {args.out_dir}")


if __name__ == "__main__":
    main()
