"""Run every variant over identical replays of the test split and write results CSVs.

Each seed fixes one incident schedule; all variants replay exactly the same injected series.
Usage: python scripts/run_experiments.py [--seeds 0 1 2] [--variants static naive ours reset]
Outputs: results/metrics.csv, results/detection.csv, results/replays/*.npz (gitignored).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
# Block idle sleep (also behind Win+L) while this process runs. Windows drops the request when
# the process exits. A closed lid or a manual Sleep still suspends the machine.
if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
sys.path.insert(0, str(ROOT / "src"))
import adapt  # noqa: E402
import data  # noqa: E402
import drift  # noqa: E402
import gwn  # noqa: E402
import incidents  # noqa: E402
import metrics  # noqa: E402
import replay  # noqa: E402

DEV = "cuda:0"
TAIL = 12  # recovery tail after an incident ends, in steps, counted as disrupted


def detector_config():
    """Detector chosen on the validation period by scripts/calibrate_detector.py."""
    c = json.loads((ROOT / "results" / "detector.json").read_text())
    det = {k: v for k, v in c.items() if k not in ("norm", "ckpt") and not k.startswith("val_")}
    if c["norm"]:
        z = np.load(ROOT / "results" / "residual_stats.npz")
        det["norm"] = (z["mu"], z["sd"])
    if det.get("agg", "").startswith("nbr"):
        det["nbrs"] = drift.neighbour_index(data.load_adj()[1])
    return det


def variants(model, tod, scaler, band):
    DET = detector_config()
    return {
        "static": lambda: drift.DetectOnly(**DET),  # (a): never updates, still logs alarms
        "naive": lambda: adapt.NaiveFT(model, tod, scaler, DEV),  # (b)
        "ours": lambda: adapt.HeadAdapt(model, tod, scaler, DEV, beta=1.0, **DET),  # (c)
        "reset": lambda: adapt.HeadAdapt(model, tod, scaler, DEV, beta=0.0, reset=True, band=(*band["det"], 1.0), **DET),
    }


def normal_band(ckpt, speed, tod, scaler, cache=ROOT / "results" / "val_band.json"):
    """Mean/std of the static t+1 error on the clean validation period, in two scales:
    'raw' = network mean error in mph (time to recovery), 'det' = the detector's own signal
    (reset policy). One static replay; both signals are computed from the same forecasts."""
    if cache.exists():
        b = json.loads(cache.read_text())
        if b.get("ckpt") == str(ckpt) and b.get("det_cfg") == (ROOT / "results" / "detector.json").read_text():
            return b
    n_train, n_val, _ = data.split_bounds(len(speed))
    v0 = replay.SEQ - 1 + n_train
    v1 = v0 + n_val - 1 - replay.SEQ
    p = replay.replay(gwn.build(DEV, ckpt), speed, tod, scaler, v0, v1, device=DEV)["preds"][:, 0]
    y = speed[v0 + 1: v1 + 2]
    err = np.where(y != 0, np.abs(p - y), np.nan)
    det = detector_config()
    raw = drift.aggregate(err, "mean")
    sig = drift.aggregate(err, det.get("agg", "mean"), det.get("norm"), np.arange(v0 + 1, v1 + 2) % drift.STEPS_PER_DAY,
                          det.get("nbrs"))
    b = dict(ckpt=str(ckpt), det_cfg=(ROOT / "results" / "detector.json").read_text(),
             raw=(float(np.nanmean(raw)), float(np.nanstd(raw))), det=(float(np.nanmean(sig)), float(np.nanstd(sig))))
    cache.write_text(json.dumps(b))
    return b


def period_masks(events, t0, steps, n_nodes):
    """Boolean (steps, 12, N) masks by target time: disrupted = affected sensor inside
    [onset, end + TAIL]; normal = target time outside every incident window."""
    tt = t0 + np.arange(steps)[:, None] + np.arange(1, replay.SEQ + 1)[None, :]  # target times
    disrupted = np.zeros((steps, replay.SEQ, n_nodes), bool)
    in_any = np.zeros(tt.shape, bool)
    for e in events.itertuples():
        w = (tt >= e.onset) & (tt <= e.end + TAIL)
        in_any |= w
        disrupted[:, :, e.sensors] |= w[:, :, None]
    normal = np.broadcast_to(~in_any[:, :, None], disrupted.shape)
    return normal, disrupted


def time_to_recovery(pred1, true1, events, t0, level, margin, hold=3):
    """Steps after each incident end until the mean t+1 error on its sensors stays within
    level + margin for `hold` consecutive steps. pred1/true1: (steps, N) aligned by target time."""
    out = []
    for e in events.itertuples():
        calm, ttr = 0, np.nan
        for tau in range(e.end + 1, min(e.end + 1 + 288, t0 + len(pred1))):
            i = tau - t0
            y, p = true1[i, e.sensors], pred1[i, e.sensors]
            m = y != 0
            err = np.abs(p[m] - y[m]).mean() if m.any() else np.nan
            calm = calm + 1 if (not np.isnan(err) and err <= level + margin) else 0
            if calm >= hold:
                ttr = tau - hold + 1 - (e.end + 1)
                break
        out.append(ttr)
    return np.array(out, float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--variants", nargs="+", default=["static", "naive", "ours", "reset"])
    ap.add_argument("--ramp", type=int, default=2)
    ap.add_argument("--ckpt", default=None)
    a = ap.parse_args()

    ckpt = a.ckpt or sorted((ROOT / "checkpoints").glob("metr_exp1_best_*.pth"))[-1]
    df = data.load_speed_df()
    speed, tod = df.values, replay.time_of_day(df.index)
    scaler = data.official_scaler()
    _, adj = data.load_adj()
    t0, t_end = data.test_start(len(speed)), len(speed) - 1 - replay.SEQ
    steps = t_end - t0 + 1
    band = normal_band(ckpt, speed, tod, scaler)
    print(f"checkpoint {ckpt}; validation band raw {band['raw']}, detector signal {band['det']}")
    (ROOT / "results" / "replays").mkdir(parents=True, exist_ok=True)

    mrows, drows = [], []
    for seed in a.seeds:
        events = incidents.schedule(adj, speed, t0, t_end, seed=seed, ramp=a.ramp)
        events.to_csv(ROOT / "results" / f"incidents_seed{seed}.csv", index=False)
        series = incidents.inject(speed, events)
        y = replay.targets(series, t0, steps)
        normal, disrupted = period_masks(events, t0, steps, speed.shape[1])
        for name in a.variants:
            torch.manual_seed(seed)
            model = gwn.build(DEV, ckpt)
            policy = variants(model, tod, scaler, band)[name]()
            out = replay.replay(model, series, tod, scaler, t0, t_end, policy, DEV)
            p = out["preds"]
            for period, mask in [("all", None), ("normal", normal), ("disrupted", disrupted)]:
                for h, i in metrics.HORIZONS.items():
                    m = metrics.masked_metrics(p[:, i], y[:, i], extra_mask=None if mask is None else mask[:, i])
                    mrows.append(dict(seed=seed, variant=name, period=period, horizon=h, **m))
            # t+1 forecasts aligned by target time: row j targets t0 + 1 + j
            ttr = time_to_recovery(p[:, 0], y[:, 0], events, t0 + 1, *band["raw"])
            alarms = getattr(policy, "alarms", [])
            st = drift.detection_stats(alarms, events, t0, t_end, TAIL) if hasattr(policy, "alarms") else {}
            upd = getattr(policy, "updates", [])
            drows.append(dict(seed=seed, variant=name, **st, n_alarms=len(alarms),
                              mean_ttr=np.nanmean(ttr) if (~np.isnan(ttr)).any() else np.nan,
                              recovered=int((~np.isnan(ttr)).sum()), n_updates=len(upd),
                              mean_update_ms=1000 * np.mean([u[1] for u in upd]) if upd else 0.0,
                              max_update_ms=1000 * np.max([u[1] for u in upd]) if upd else 0.0,
                              n_resets=len(getattr(policy, "resets", []))))
            np.savez_compressed(ROOT / "results" / "replays" / f"seed{seed}_{name}.npz",
                                preds=p[:, [0, 2, 5, 11]], alarms=np.array(alarms),
                                signal=np.array(getattr(policy, "signal", [])))
            d = mrows[-3]  # disrupted 15min row
            print(f"seed {seed} {name:7s} disrupted 15min MAE {d['mae']:.3f}  alarms {len(alarms)}  updates {len(upd)}")
            pd.DataFrame(mrows).to_csv(ROOT / "results" / "metrics.csv", index=False, float_format="%.4f")
            pd.DataFrame(drows).to_csv(ROOT / "results" / "detection.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
