"""Figures from run_experiments.py outputs. Usage: python scripts/plots.py [--seed 0] [--event 0]
Writes results/fig_*.png."""
import argparse
import ast
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import incidents  # noqa: E402

RES = ROOT / "results"
COLORS = {"static": "#555555", "naive": "#d95f02", "ours": "#1b9e77", "reset": "#7570b3"}


def bars():
    m = pd.read_csv(RES / "metrics.csv")
    m = m[m.period != "all"]
    g = m.groupby(["period", "horizon", "variant"]).mae.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)
    for ax, period in zip(axes, ["normal", "disrupted"]):
        d = g[g.period == period]
        hs, vs = ["15min", "30min", "60min"], [v for v in COLORS if v in set(d.variant)]
        w = 0.8 / len(vs)
        for j, v in enumerate(vs):
            r = d[d.variant == v].set_index("horizon").reindex(hs)
            ax.bar(np.arange(3) + j * w, r["mean"], w, yerr=r["std"], label=v, color=COLORS[v])
        ax.set_xticks(np.arange(3) + w * (len(vs) - 1) / 2, hs)
        ax.set_title(f"{period} period")
        ax.set_ylabel("MAE (mph)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig_mae_bars.png", dpi=150)


def incident(seed, k):
    ev = pd.read_csv(RES / f"incidents_seed{seed}.csv", converters={"sensors": ast.literal_eval})
    e = ev.iloc[k]
    df = data.load_speed_df()
    speed = incidents.inject(df.values, ev)
    t0 = data.test_start(len(speed))
    lo, hi = e.onset - 36, e.end + 48
    s = e.sensors[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    t = np.arange(lo, hi)
    a1.plot(df.index[t], np.where(speed[t, s] == 0, np.nan, speed[t, s]), "k", lw=2, label="actual")
    for v, c in COLORS.items():
        f = RES / "replays" / f"seed{seed}_{v}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        p1 = z["preds"][:, 0]  # t+1 forecasts, row i issued at t0 + i, target t0 + i + 1
        a1.plot(df.index[t], p1[t - 1 - t0, s], color=c, label=f"{v} t+1")
        sig = z["signal"]
        if len(sig):
            keep = (sig[:, 0] >= lo) & (sig[:, 0] < hi)
            a2.plot(df.index[sig[keep, 0].astype(int)], sig[keep, 1], color=c, label=v)
        for al in z["alarms"]:
            if lo <= al < hi:
                a2.axvline(df.index[int(al)], color=c, ls=":", lw=1)
    for a in (a1, a2):
        a.axvspan(df.index[e.onset], df.index[e.end], color="red", alpha=0.08)
    a1.set_ylabel(f"speed, sensor {s} (mph)")
    a2.set_ylabel("network t+1 error (mph)")
    a1.legend(frameon=False, ncol=3)
    a2.legend(frameon=False, ncol=4)
    fig.tight_layout()
    fig.savefig(RES / f"fig_incident_seed{seed}_{k}.png", dpi=150)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event", type=int, default=0)
    a = ap.parse_args()
    bars()
    incident(a.seed, a.event)
    print("figures written to", RES)
