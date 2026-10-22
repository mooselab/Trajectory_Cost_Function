#!/usr/bin/env python3
"""Paper-level evaluation over aligned trajectories (align_trial.py -> results/trials.csv).

RQ1  estimated vs. measured trajectory energy. Baselines (tokens-only, time-only) are fitted with
     leave-one-task-out; the cost-function variants use S2-calibrated coefficients (no fitting on
     trajectories). Pearson, Spearman, MAPE with a bootstrap 95% CI.
RQ2  energy split into model / tool / wait, per task.
EpG  energy per accomplished goal per configuration, and the retry-budget trade-off: with budget k,
     attempts r1..rk are billed until the first success.

Usage:
  python evaluate.py --trials results/trials.csv [--configs summarize-off summarize-8k] --out results/eval
"""
import argparse
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

FORMULAS = {"formula (original)": "E_est_original", "formula (cached split)": "E_est_cached_split",
            "formula (cached split + ctx*out)": "E_est_cached_split_ctxout"}


def loto(d, cols, offset=None):
    """Leave-one-task-out least-squares prediction of E_measured from cols (no intercept).
    With offset, only E_measured - offset is fitted and the offset is added back."""
    X, y, pred = d[cols].to_numpy(float), d["E_measured"].to_numpy(float), np.empty(len(d))
    off = np.zeros(len(d)) if offset is None else np.asarray(offset, float)
    for t in d["task"].unique():
        te = (d["task"] == t).to_numpy()
        coef, *_ = np.linalg.lstsq(X[~te], (y - off)[~te], rcond=None)
        pred[te] = X[te] @ coef + off[te]
    return pred


def fit_all(d, cols, offset=None):
    """Coefficients fitted on all trajectories (for interpretation, not for evaluation)."""
    off = np.zeros(len(d)) if offset is None else np.asarray(offset, float)
    coef, *_ = np.linalg.lstsq(d[cols].to_numpy(float), d["E_measured"].to_numpy(float) - off, rcond=None)
    return dict(zip(cols, coef.tolist()))


def metrics(pred, y, rng):
    rank = lambda a: pd.Series(a).rank().to_numpy()  # noqa: E731
    ape = np.abs(pred - y) / y
    boots = [ape[rng.integers(0, len(y), len(y))].mean() for _ in range(1000)]
    return {"pearson": float(np.corrcoef(pred, y)[0, 1]), "spearman": float(np.corrcoef(rank(pred), rank(y))[0, 1]),
            "MAPE_pct": 100 * float(ape.mean()), "MAPE_ci_lo": 100 * float(np.percentile(boots, 2.5)),
            "MAPE_ci_hi": 100 * float(np.percentile(boots, 97.5)), "n": int(len(y))}


def retry_budget(d):
    rows = []
    reps = sorted(d["rep"].unique(), key=lambda r: int(str(r).lstrip("r")))
    for cfg, g in d.groupby("config"):
        for k in range(1, len(reps) + 1):
            billed, solved = 0.0, 0
            for _, tg in g.groupby("task"):
                tg = tg.set_index("rep").reindex(reps[:k]).dropna(subset=["E_measured"])
                for _, r in tg.iterrows():
                    billed += r["E_measured"]
                    if r["reward"] == 1.0:
                        solved += 1
                        break
            # check: the repetitions have no natural order, so also average over every order of them
            billed_all, solved_all = 0.0, 0.0
            for _, tg in g.groupby("task"):
                att = list(zip(tg["E_measured"], tg["reward"]))
                perms = list(itertools.permutations(att))
                for p in perms:
                    for e, rw in p[:k]:
                        billed_all += e / len(perms)
                        if rw == 1.0:
                            solved_all += 1 / len(perms)
                            break
            n_tasks = g["task"].nunique()
            rows.append({"config": cfg, "budget": k, "tasks_solved": solved, "n_tasks": n_tasks,
                         "success_rate": solved / n_tasks, "billed_kJ": billed / 1000,
                         "EpG_kJ": billed / 1000 / solved if solved else np.nan,
                         "tasks_solved_all_orders": solved_all, "billed_kJ_all_orders": billed_all / 1000,
                         "EpG_kJ_all_orders": billed_all / 1000 / solved_all if solved_all else np.nan})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", default="results/trials.csv")
    ap.add_argument("--configs", nargs="*", help="default: all configs in the file")
    ap.add_argument("--calib", default="results/calib", help="for the loaded-idle power w")
    ap.add_argument("--out", default="results/eval")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = pd.read_csv(args.trials)
    if args.configs:
        d = d[d["config"].isin(args.configs)].reset_index(drop=True)
    y, rng = d["E_measured"].to_numpy(float), np.random.default_rng(0)

    # regression variants: A refits only the non-model powers on top of the calibrated model part,
    # B refits every coefficient from trajectories (the "regression method")
    w = json.load(open(os.path.join(args.calib, "params.json")))["idle_loaded_W"]["total"]
    d["sum_uncached"] = d["sum_ctx"] - d["sum_cached"]
    model_part = d["E_est_cached_split"] - w * (d["t_tool"] + d["t_wait"])
    full_cols = ["sum_uncached", "sum_cached", "sum_out", "t_model", "t_tool", "t_wait"]
    preds = {"tokens-only (LOTO fit)": loto(d, ["sum_ctx", "sum_out"]),
             "time-only (LOTO fit)": loto(d, ["t_agent"]),
             **{k: d[c].to_numpy(float) for k, c in FORMULAS.items()},
             "regression A: calibrated model + fitted tool/wait power (LOTO)": loto(d, ["t_tool", "t_wait"], model_part),
             "regression B: all coefficients fitted (LOTO)": loto(d, full_cols)}
    coefs = {"A (W)": fit_all(d, ["t_tool", "t_wait"], model_part), "B": fit_all(d, full_cols),
             "calibrated w (W)": w}
    json.dump(coefs, open(os.path.join(args.out, "regression_coefs.json"), "w"), indent=2)
    print("Fitted on all trajectories:", json.dumps(coefs, indent=1))
    pd.DataFrame({"trial": d["trial"], "task": d["task"], "config": d["config"], "reward": d["reward"],
                  "measured": y, **preds}).to_csv(os.path.join(args.out, "rq1_points.csv"), index=False)
    rq1 = pd.DataFrame([{"predictor": k, **metrics(p, y, rng)} for k, p in preds.items()])
    rq1.to_csv(os.path.join(args.out, "rq1_predictors.csv"), index=False)
    print("RQ1\n" + rq1.round(3).to_string(index=False))

    show = ["tokens-only (LOTO fit)", "time-only (LOTO fit)", "formula (cached split)"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), sharex=True, sharey=True)
    lim = [0, max(y.max(), max(preds[k].max() for k in show)) / 1000 * 1.05]
    for ax, k in zip(axes, show):
        m = rq1.set_index("predictor").loc[k]
        for ok, mk in ((1.0, "o"), (0.0, "x")):
            s = (d["reward"] == ok).to_numpy()
            ax.scatter(y[s] / 1000, preds[k][s] / 1000, marker=mk, s=18, label="solved" if ok else "failed")
        ax.plot(lim, lim, color="gray", lw=0.8, ls="--")
        ax.set_title(f"{k}\nMAPE {m.MAPE_pct:.1f}%, ρ {m.spearman:.2f}", fontsize=9)
        ax.set_xlabel("measured (kJ)")
    axes[0].set_ylabel("estimated (kJ)")
    axes[0].legend(fontsize=8)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rq1_scatter.png"), dpi=150)

    # same comparison on log-log axes, so the many small trajectories are visible; the band is ±10%
    m = rq1.set_index("predictor")
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.3), sharex=True, sharey=True)
    lo, hi = max(1.0, y.min() / 1000 * 0.7), y.max() / 1000 * 1.3
    xs = np.array([lo, hi])
    for ax, k, name in zip(axes, show, ["tokens-only", "time-only", "cost function"]):
        ax.fill_between(xs, xs * 0.9, xs * 1.1, color="#DDDDDD", alpha=0.6, lw=0, label="±10%")
        ax.plot(xs, xs, color="gray", lw=0.8, ls="--")
        for i, cfg in enumerate(sorted(d["config"].unique())):
            sel = (d["config"] == cfg).to_numpy()
            ax.scatter(y[sel] / 1000, preds[k][sel] / 1000, s=16, alpha=0.8, marker="o^sD"[i % 4],
                       color=["#4C72B0", "#DD8452", "#55A868", "#C44E52"][i % 4], label=cfg)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(f"{name}\nMAPE {m.loc[k, 'MAPE_pct']:.1f}%", fontsize=10)
        ax.set_xlabel("measured energy (kJ, log scale)")
    axes[0].set_ylabel("estimated energy (kJ, log scale)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rq1_scatter_log.png"), dpi=200)

    share = d.groupby("task")[["E_model", "E_tool", "E_wait", "E_measured"]].mean()
    share = share[["E_model", "E_tool", "E_wait"]].div(share["E_measured"], axis=0).sort_values("E_tool")
    share.to_csv(os.path.join(args.out, "rq2_shares.csv"))
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(share) + 1))
    left = np.zeros(len(share))
    for col, color in (("E_model", "#4C72B0"), ("E_tool", "#DD8452"), ("E_wait", "#BBBBBB")):
        ax.barh(share.index, share[col], left=left, color=color, label=col[2:])
        left += share[col].to_numpy()
    ax.set_xlabel("share of trajectory energy")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "rq2_shares.png"), dpi=150)

    cfg = d.groupby("config").agg(trajectories=("trial", "count"), success_rate=("reward", "mean"),
                                  mean_kJ=("E_measured", lambda s: s.mean() / 1000),
                                  mean_turns=("n_model_calls", "mean"),
                                  cache_hit=("sum_cached", "sum"), ctx=("sum_ctx", "sum"),
                                  total_kJ=("E_measured", lambda s: s.sum() / 1000), solved=("reward", "sum"))
    cfg["cache_hit"] = cfg["cache_hit"] / cfg.pop("ctx")
    cfg["EpG_kJ"] = cfg["total_kJ"] / cfg["solved"].replace(0, np.nan)
    cfg.to_csv(os.path.join(args.out, "configs.csv"))
    print("\nConfigurations\n" + cfg.round(3).to_string())

    rb = retry_budget(d)
    rb.to_csv(os.path.join(args.out, "retry_budget.csv"), index=False)
    print("\nRetry budget\n" + rb.round(3).to_string(index=False))
    print(f"\noutputs in {args.out}")


if __name__ == "__main__":
    main()
