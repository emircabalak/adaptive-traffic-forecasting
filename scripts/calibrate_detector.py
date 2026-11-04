"""Choose the drift detector on the validation period only.

The static model never changes, so each validation scenario is replayed once and the per-sensor
t+1 errors are stored. Every (signal aggregation, detector, parameter) combination then runs on
those stored errors, which takes seconds. Validation seeds (100+) never overlap test seeds.
Selection: highest detection rate with at most MAX_FALSE false alarms per normal day, ties broken
by mean delay. Writes results/calibration.csv and results/detector.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import drift  # noqa: E402
import gwn  # noqa: E402
import incidents  # noqa: E402
import replay  # noqa: E402

DEV = "cuda:0"
SEEDS = [100, 101, 102]
TAIL = 12
MAX_FALSE = 1.0
GRID = [("adwin", dict(delta=d)) for d in [1e-5, 1e-4, 1e-3, 2e-3, 1e-2, 0.05, 0.1, 0.3]] + \
       [("ph", dict(threshold=t, delta=d)) for t in [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 400] for d in [0.005, 0.05, 0.2]]


def residual_stats(model, speed, tod, scaler, n_train, ckpt, cache=ROOT / "results" / "residual_stats.npz"):
    """Mean/std of the static model's absolute t+1 error per (time-of-day slot, sensor) over the
    training period. Training data only, so validation and test stay unseen."""
    if cache.exists() and str(np.load(cache)["ckpt"]) == str(ckpt):
        z = np.load(cache)
        return z["mu"], z["sd"]
    t0, t1 = replay.SEQ - 1, replay.SEQ - 1 + n_train - 1
    p = replay.replay(model, speed, tod, scaler, t0, t1, device=DEV)["preds"][:, 0]
    y = speed[t0 + 1: t1 + 2]
    err = np.where(y != 0, np.abs(p - y), np.nan)
    slot = np.arange(t0 + 1, t1 + 2) % drift.STEPS_PER_DAY
    mu = np.full((drift.STEPS_PER_DAY, speed.shape[1]), np.nan)
    sd = mu.copy()
    with np.errstate(all="ignore"):
        for k in range(drift.STEPS_PER_DAY):
            mu[k], sd[k] = np.nanmean(err[slot == k], 0), np.nanstd(err[slot == k], 0)
    # Cells with no valid training reading fall back to the sensor's overall stats
    mu = np.where(np.isnan(mu), np.nanmean(err, 0), mu)
    sd = np.maximum(np.where(np.isnan(sd), np.nanstd(err, 0), sd), 0.5)  # 0.5 mph floor
    np.savez(cache, mu=mu, sd=sd, ckpt=str(ckpt))
    return mu, sd


def run_detector(signal, steps_now, kind, params):
    det = drift.make_detector(kind, **params)
    alarms = []
    for now, x in zip(steps_now, signal):
        if np.isnan(x):
            continue
        det.update(float(x))
        if det.drift_detected:
            alarms.append(int(now))
    return alarms


def main():
    ckpt = sorted((ROOT / "checkpoints").glob("metr_exp1_best_*.pth"))[-1]
    df = data.load_speed_df()
    speed, tod = df.values, replay.time_of_day(df.index)
    scaler = data.official_scaler()
    _, adj = data.load_adj()
    n_train, n_val, _ = data.split_bounds(len(speed))
    v0 = replay.SEQ - 1 + n_train
    v1 = v0 + n_val - 1 - replay.SEQ  # last 'now' whose targets stay inside validation
    model = gwn.build(DEV, ckpt)

    scen = []
    for seed in SEEDS:
        ev = incidents.schedule(adj, speed, v0 + replay.SEQ, v1, seed=seed, ramp=2)
        series = incidents.inject(speed, ev)
        p = replay.replay(model, series, tod, scaler, v0, v1, device=DEV)["preds"][:, 0]
        y = series[v0 + 1: v1 + 2]  # target of the t+1 forecast issued at v0 + i
        err = np.where(y != 0, np.abs(p - y), np.nan)
        scen.append((seed, ev, err))
        print(f"seed {seed}: {len(ev)} incidents replayed")

    # the signal for step `now` is the error of the forecast issued at now - 1
    nows = np.arange(v0 + 1, v1 + 2)
    norm = residual_stats(model, speed, tod, scaler, n_train, ckpt)
    slots = nows % drift.STEPS_PER_DAY
    rows = []
    nbrs = drift.neighbour_index(adj)
    combos = [(n, h) for n in (False, True) for h in ["mean", "max", "top3", "top5", "top10", "top20"]]
    combos += [(True, h) for h in ["nbrmax", "nbrtop3"]]  # neighbour-confirmed z (EXPERIMENTS E5)
    for normed, how in combos:
        sig = {seed: drift.aggregate(err, how, norm if normed else None, slots, nbrs) for seed, _, err in scen}
        for kind, params in GRID:
            stats = []
            for seed, ev, _ in scen:
                al = run_detector(sig[seed], nows, kind, params)
                stats.append(drift.detection_stats(al, ev, v0 + 1, v1 + 1, TAIL))
            n = sum(s["n_events"] for s in stats)
            det = sum(s["detected"] for s in stats)
            delays = [s["mean_delay"] * s["detected"] for s in stats if s["detected"]]
            rows.append(dict(norm=normed, agg=how, kind=kind, params=json.dumps(params), events=n, detected=det,
                             rate=det / n, mean_delay=sum(delays) / det if det else np.nan,
                             false_per_day=np.mean([s["false_per_day"] for s in stats])))
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "results" / "calibration.csv", index=False, float_format="%.4f")
    ok = res[res.false_per_day <= MAX_FALSE].sort_values(["rate", "mean_delay"], ascending=[False, True])
    print(ok.head(12).to_string(index=False, float_format="%.3f"))
    best = ok.iloc[0]
    choice = dict(norm=bool(best["norm"]), agg=best["agg"], kind=best["kind"], **json.loads(best["params"]),
                  val_rate=float(best["rate"]), val_delay=float(best["mean_delay"]),
                  val_false_per_day=float(best["false_per_day"]), ckpt=str(ckpt))
    (ROOT / "results" / "detector.json").write_text(json.dumps(choice, indent=2))
    print("chosen:", choice)


if __name__ == "__main__":
    main()
