"""Shared helpers: load sampler CSVs and integrate energy over time intervals."""
import json

import numpy as np
import pandas as pd


def load_samples(path):
    return pd.read_csv(path)


def gpu_cols(df):
    return [c for c in df.columns if c.startswith("gpu") and c.endswith("_j")]


def cpu_cols(df):
    # package domains only; sub-domains (core, ...) are already inside the package
    return [c for c in df.columns if c.startswith("cpu_pkg") and c.endswith("_j")]


def interval_energy(df, t0, t1):
    """Energy (J) between monotonic times t0 and t1, interpolated from cumulative counters.

    Returns {"gpu": J, "cpu": J, "total": J}.
    """
    t = df["t_mono"].to_numpy()
    if t0 < t[0] or t1 > t[-1]:
        raise ValueError(f"interval [{t0:.1f}, {t1:.1f}] outside samples [{t[0]:.1f}, {t[-1]:.1f}]")

    def delta(cols):
        return float(sum(np.interp(t1, t, df[c].to_numpy()) - np.interp(t0, t, df[c].to_numpy()) for c in cols))

    gpu, cpu = delta(gpu_cols(df)), delta(cpu_cols(df))
    return {"gpu": gpu, "cpu": cpu, "total": gpu + cpu}


def mean_power(df):
    """Average power (W) over the whole sample file: {"gpu", "cpu", "total"}."""
    t0, t1 = df["t_mono"].iloc[0], df["t_mono"].iloc[-1]
    e = interval_energy(df, t0, t1)
    return {k: v / (t1 - t0) for k, v in e.items()}


def read_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]
