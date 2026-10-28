"""Figures from run_experiments.py outputs.
Usage: python scripts/plots.py [--seed 0] [--event9 6] [--event24 5] [--typical24 9]
The default incidents (seed 0): event9 and event24 are where the methods gain most at 15 min,
picked to show the mechanism (best cases); typical24 is the incident closest to the rule's
median gain. The bar charts show the averages.
Writes results/fig_mae_bars_dur{9,24}.png and results/fig_incident_seed{S}_dur{D}_{K}.png."""
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
# fixed categorical order (dataviz reference palette, validated for light mode); ours = node-global
COLORS = {"static": "#2a78d6", "naive": "#eb6834", "head": "#1baf7a", "node-global": "#eda100",
          "rule-severe": "#e87ba4"}
LABELS = {"static": "static (a)", "naive": "naive (b)", "head": "head only (spec c)",
          "node-global": "ours: node adapter", "rule-severe": "severe-slowdown rule"}
INK, MUTED = "#1f1f1e", "#6b6a64"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color="#e6e5e0", lw=0.8)
    ax.set_axisbelow(True)


def bars(dur):
    m = pd.read_csv(RES / "metrics.csv")
    m = m[(m.dur == dur) & m.variant.isin(COLORS) & (m.period != "all")]
    g = m.groupby(["period", "horizon", "variant"]).mae.agg(["mean", "std"]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    hs = ["15min", "30min", "60min"]
    for ax, period in zip(axes, ["normal", "disrupted"]):
        d = g[g.period == period]
        w = 0.8 / len(COLORS)
        for j, v in enumerate(COLORS):
            r = d[d.variant == v].set_index("horizon").reindex(hs)
            x = np.arange(3) + j * w
            ax.bar(x, r["mean"], w * 0.92, yerr=r["std"], color=COLORS[v], label=LABELS[v],
                   error_kw=dict(ecolor=MUTED, lw=1))
            for xi, val in zip(x, r["mean"]):
                ax.text(xi, val, f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=INK, rotation=90)
        ax.set_xticks(np.arange(3) + w * (len(COLORS) - 1) / 2, ["15 min", "30 min", "60 min"])
        ax.set_title(f"{period} period", color=INK)
        ax.set_ylabel("MAE (mph), mean over 3 seeds", color=INK)
        ax.margins(y=0.15)
        style(ax)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Test split, synthetic blockages of {dur * 5} min", color=INK)
    fig.tight_layout()
    fig.savefig(RES / f"fig_mae_bars_dur{dur}.png", dpi=150)
    plt.close(fig)


def incident(seed, dur, k, note="a best case, see the averages in the bar charts"):
    ev = pd.read_csv(RES / f"incidents_seed{seed}_dur{dur}.csv", converters={"sensors": ast.literal_eval})
    e = ev.iloc[k]
    df = data.load_speed_df()
    speed = incidents.inject(df.values, ev)
    t0 = data.test_start(len(speed))
    lo, hi = e.onset - 24, e.end + 36
    t = np.arange(lo, hi)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for ax, (col, hz) in zip(axes, [(1, "15 min"), (2, "30 min")]):
        s = e.sensors[0]
        ax.plot(df.index[t], np.where(speed[t, s] == 0, np.nan, speed[t, s]), color=INK, lw=2, label="actual")
        for v in ["static", "node-global", "rule-severe"]:
            z = np.load(RES / "replays" / f"seed{seed}_dur{dur}_{v}.npz")
            step = [0, 3, 6][col]  # saved horizons are 1, 3, 6, 12 steps ahead
            p = z["preds"][:, col]  # row i issued at t0 + i, target t0 + i + step
            ax.plot(df.index[t], p[t - step - t0, s], color=COLORS[v], lw=2, label=f"{LABELS[v]}, {hz} ahead")
            if v == "rule-severe":
                for al in z["alarms"]:
                    if lo <= al < hi:
                        ax.axvline(df.index[int(al)], color=COLORS[v], ls=":", lw=1)
        ax.axvspan(df.index[e.onset], df.index[e.end], color="#e34948", alpha=0.08, lw=0)
        ax.set_ylabel(f"speed at sensor {s} (mph)", color=INK)
        ax.legend(frameon=False, fontsize=8, loc="lower left")
        style(ax)
    axes[0].set_title(f"Seed {seed}, incident {k} ({note}): "
                      f"forecasts at their target time, shaded = blockage, dotted = alarms", color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(RES / f"fig_incident_seed{seed}_dur{dur}_{k}.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--event9", type=int, default=6)
    ap.add_argument("--event24", type=int, default=5)
    ap.add_argument("--typical24", type=int, default=9, help="incident closest to the median gain")
    a = ap.parse_args()
    for dur, k in ((9, a.event9), (24, a.event24)):
        bars(dur)
        incident(a.seed, dur, k)
    incident(a.seed, 24, a.typical24, note="a typical case: median gain of the rule")
    print("figures written to", RES)
