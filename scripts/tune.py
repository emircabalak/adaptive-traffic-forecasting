"""Validation harness for the online method (protocol in docs/EXPERIMENTS.md).

Usage:
  python scripts/tune.py '{"name": "E0-ours", "policy": "head", "beta": 1.0}'
  python scripts/tune.py --diag            # E1: where the static disrupted error comes from

A config is a JSON object: "name", "policy" (a key of adapt.POLICIES), optional "det" (detector
overrides, default results/detector.json) and any keyword arguments of that policy. Static
forecasts per seed are cached, so each config costs one replay per seed. Appends one row per
config to results/tuning.csv.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as rx  # noqa: E402  (also sets up src/ imports and blocks idle sleep)
from run_experiments import adapt, data, drift, gwn, incidents, metrics, replay  # noqa: E402

SEEDS = [100, 101, 102]
CACHE = ROOT / "results" / "tune_cache"


def setup(dur=9):
    ckpt = sorted((ROOT / "checkpoints").glob("metr_exp1_best_*.pth"))[-1]
    df = data.load_speed_df()
    speed, tod = df.values, replay.time_of_day(df.index)
    scaler = data.official_scaler()
    _, adj = data.load_adj()
    n_train, n_val, _ = data.split_bounds(len(speed))
    v0 = replay.SEQ - 1 + n_train
    v1 = v0 + n_val - 1 - replay.SEQ
    scen = []
    for seed in SEEDS:
        ev = incidents.schedule(adj, speed, v0 + replay.SEQ, v1, seed=seed, ramp=2, duration=dur)
        series = incidents.inject(speed, ev)
        normal, disrupted = rx.period_masks(ev, v0, v1 - v0 + 1, speed.shape[1])
        scen.append(dict(seed=seed, ev=ev, series=series, y=replay.targets(series, v0, v1 - v0 + 1),
                         normal=normal, disrupted=disrupted))
    return dict(dur=dur, ckpt=ckpt, speed=speed, tod=tod, scaler=scaler, v0=v0, v1=v1, scen=scen,
                band=rx.normal_band(ckpt, speed, tod, scaler))


def static_preds(S, sc):
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"static_seed{sc['seed']}_dur{S['dur']}.npy"
    if not f.exists():
        out = replay.replay(gwn.build(rx.DEV, S["ckpt"]), sc["series"], S["tod"], S["scaler"], S["v0"], S["v1"], device=rx.DEV)
        np.save(f, out["preds"])
    return np.load(f)


def scores(p, sc):
    r = {}
    for period in ("normal", "disrupted"):
        for h, i in metrics.HORIZONS.items():
            r[f"{period[:3]}{h[:-3]}"] = metrics.masked_metrics(p[:, i], sc["y"][:, i], extra_mask=sc[period][:, i])["mae"]
    return r


def run(cfg, S):
    cfg = dict(cfg)
    name, kind = cfg.pop("name"), cfg.pop("policy")
    det = {**rx.detector_config(), **cfg.pop("det", {})}
    oracle = cfg.pop("oracle_offset", None)  # diagnostics: alarm exactly this many steps after each onset
    rows = []
    for sc in S["scen"]:
        if oracle is not None:
            det["oracle"] = [int(o) + oracle for o in sc["ev"].onset]
        base = scores(static_preds(S, sc), sc)
        model = gwn.build(rx.DEV, S["ckpt"])
        pol = adapt.POLICIES[kind](model, S["tod"], S["scaler"], rx.DEV, band=(*S["band"]["det"], 1.0), **det, **cfg)
        out = replay.replay(model, sc["series"], S["tod"], S["scaler"], S["v0"], S["v1"], pol, rx.DEV)
        r = scores(out["preds"], sc)
        st = drift.detection_stats(pol.alarms, sc["ev"], S["v0"], S["v1"], rx.TAIL)
        upd = [u[1] for u in getattr(pol, "updates", [])]
        rows.append({**{k: r[k] for k in r}, **{f"static_{k}": base[k] for k in base},
                     "detected": st["detected"], "n_events": st["n_events"], "false_per_day": st["false_per_day"],
                     "update_ms": 1000 * np.mean(upd) if upd else 0.0})
    m = pd.DataFrame(rows).mean()
    res = dict(name=name, dur=S["dur"], config=json.dumps({"policy": kind, **cfg, "oracle_offset": oracle}, default=str),
               dis_avg=np.mean([m["dis15"], m["dis30"], m["dis60"]]),
               static_dis_avg=np.mean([m["static_dis15"], m["static_dis30"], m["static_dis60"]]),
               norm15_delta=m["nor15"] - m["static_nor15"], **m.to_dict(), time=time.strftime("%Y-%m-%d %H:%M"))
    res["dis_gain"] = res["static_dis_avg"] - res["dis_avg"]
    res["ok"] = res["norm15_delta"] <= 0.005
    f = ROOT / "results" / "tuning.csv"
    pd.DataFrame([res]).to_csv(f, mode="a", header=not f.exists(), index=False, float_format="%.4f")
    print(f"{name} [dur {S['dur']}]: dis_avg {res['dis_avg']:.3f} (static {res['static_dis_avg']:.3f}, gain {res['dis_gain']:+.3f}) "
          f"| dis15/30/60 {m['dis15']:.3f}/{m['dis30']:.3f}/{m['dis60']:.3f} "
          f"| norm15 delta {res['norm15_delta']:+.4f} {'OK' if res['ok'] else 'VIOLATES'} "
          f"| detected {m['detected']:.1f}/{m['n_events']:.1f}, false/day {m['false_per_day']:.2f}, update {m['update_ms']:.0f} ms")
    return res


def diag(S):
    """E1: split the static disrupted absolute error by when the forecast was issued."""
    rows = []
    for sc in S["scen"]:
        p = static_preds(S, sc)
        y, dis = sc["y"], sc["disrupted"]
        issue = S["v0"] + np.arange(len(p))
        for e in sc["ev"].itertuples():
            for i, name in [(issue < e.onset, "issued before onset"),
                            ((issue >= e.onset) & (issue <= e.end), "issued during incident"),
                            (issue > e.end, "issued after release")]:
                for h, k in metrics.HORIZONS.items():
                    tt = issue + k + 1  # target time
                    own = i & (tt >= e.onset) & (tt <= e.end + rx.TAIL)  # this incident's window only
                    m = np.zeros_like(dis[:, k])
                    m[np.ix_(own, e.sensors)] = True
                    m &= y[:, k] != 0
                    rows.append(dict(seed=sc["seed"], horizon=h, when=name,
                                     abs_err_sum=np.abs(p[:, k] - y[:, k])[m].sum(), n=m.sum()))
    d = pd.DataFrame(rows).groupby(["horizon", "when"])[["abs_err_sum", "n"]].sum()
    d["mae"] = d.abs_err_sum / d.n
    d["share_of_error"] = d.abs_err_sum / d.groupby("horizon").abs_err_sum.transform("sum")
    print(d[["n", "mae", "share_of_error"]].round(3).to_string())
    d.to_csv(ROOT / "results" / "diag_error_split.csv")


if __name__ == "__main__":
    args = sys.argv[1:]
    dur = 9
    if "--dur" in args:  # hold length of the synthetic blockage in steps (spec: 9 = 45 min)
        i = args.index("--dur")
        dur = int(args[i + 1])
        del args[i: i + 2]
    S = setup(dur)
    if args == ["--diag"]:
        diag(S)
    else:
        for a in args:
            run(json.loads(a), S)
