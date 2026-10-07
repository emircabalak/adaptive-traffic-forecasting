"""Drift detection on the t+1 forecast error, plus detection delay / false alarm accounting.

The signal is built from the labels released at `now` by the replay buffer, using only the
horizon-1 forecast (issued at now-1), because it is the first label to arrive. MAPE is never
used here: it explodes on missing zeros and near-zero speeds.
"""
import numpy as np
from river import drift

from replay import Static

STEPS_PER_DAY = 288


def t1_errors(labels):
    """Per-sensor absolute error of the t+1 forecast released at this step, nan where missing.
    None if no t+1 label arrived."""
    for s, k, pred, true in labels:
        if k == 0:
            return np.where(true != 0, np.abs(pred - true), np.nan)
    return None


def aggregate(err, agg="mean", norm=None, slot=None):
    """One number from per-sensor errors (last axis). err may be (N,) or (steps, N).
    norm = (mu, sd), each (288, N): typical error per time-of-day slot and sensor, taken from the
    training period. With norm, errors become z-scores, so rush hour and noisy sensors stop
    looking like incidents. agg: "mean", "max" or "topK" (mean of the K largest)."""
    err = np.asarray(err, float)
    if norm is not None:
        mu, sd = norm
        err = (err - mu[slot]) / sd[slot]
    with np.errstate(all="ignore"):
        if agg == "mean":
            return np.nanmean(err, -1)
        if agg == "max":
            return np.nanmax(err, -1)
        k = int(agg[3:])
        top = -np.sort(-np.where(np.isnan(err), -np.inf, err), -1)[..., :k]
        return np.nanmean(np.where(np.isinf(top), np.nan, top), -1)


def make_detector(kind="adwin", **params):
    if kind == "adwin":
        return drift.ADWIN(**{"clock": 1, **params})  # default clock=32 checks only every 32 steps (2.7 h)
    if kind == "ph":
        return drift.PageHinkley(**{"mode": "up", **params})
    raise ValueError(kind)


class DetectOnly(Static):
    """Feeds the t+1 error to a detector and logs alarms. No model updates."""

    def __init__(self, kind="adwin", agg="mean", norm=None, **params):
        self.det, self.agg, self.norm = make_detector(kind, **params), agg, norm
        self.signal, self.alarms = [], []

    def step(self, now, labels, past):
        e = t1_errors(labels)
        # the series starts at 00:00 with no gaps, so the slot of time `now` is now % 288
        x = np.nan if e is None else float(aggregate(e, self.agg, self.norm, now % STEPS_PER_DAY))
        self.signal.append((now, x))
        if np.isnan(x):
            return False  # all sensors missing at this step: nothing to learn from
        self.det.update(float(x))
        if self.det.drift_detected:
            self.alarms.append(now)
            return True
        return False


def detection_stats(alarms, events, t_start, t_end, tail=12):
    """events: DataFrame with onset/end. An alarm inside [onset, end + tail] detects that event;
    every other alarm is false. False alarms are normalised by the length of the normal period."""
    alarms = np.asarray(sorted(alarms))
    covered = np.zeros(t_end - t_start + 1, bool)
    delays = []
    for e in events.itertuples():
        lo, hi = e.onset, e.end + tail
        covered[max(lo, t_start) - t_start: min(hi, t_end) - t_start + 1] = True
        hits = alarms[(alarms >= lo) & (alarms <= hi)]
        delays.append(hits[0] - lo if len(hits) else np.nan)
    false = [a for a in alarms if not covered[a - t_start]]
    normal_days = (~covered).sum() / STEPS_PER_DAY
    delays = np.array(delays, float)
    return dict(n_events=len(delays), detected=int((~np.isnan(delays)).sum()),
                mean_delay=np.nanmean(delays) if (~np.isnan(delays)).any() else np.nan,
                median_delay=np.nanmedian(delays) if (~np.isnan(delays)).any() else np.nan,
                false_alarms=len(false), false_per_day=len(false) / normal_days if normal_days else np.nan)


if __name__ == "__main__":
    import pandas as pd

    lab = lambda p, t: [(0, 0, np.array(p, float), np.array(t, float))]
    e = t1_errors(lab([50, 10, 3], [60, 0, 1]))
    assert np.isclose(aggregate(e), 6.0) and np.isclose(aggregate(e, "top1"), 10.0)  # missing ignored
    assert np.isnan(aggregate(t1_errors(lab([1, 2], [0, 0]))))
    mu, sd = np.full((288, 3), 2.0), np.full((288, 3), 4.0)
    assert np.isclose(aggregate(e, "max", (mu, sd), 7), 2.0)  # (10 - 2) / 4

    rng = np.random.default_rng(0)
    x = np.abs(rng.normal(2.3, 0.4, 2000))
    x[1000:1009] += 3.0  # a 9-step jump in error
    for kind, params in [("adwin", {}), ("ph", {"threshold": 5.0})]:
        d = DetectOnly(kind, **params)
        for t, v in enumerate(x):
            d.step(t, lab([60.0 + v], [60.0]), None)  # t+1 error of exactly v
        st = detection_stats(d.alarms, pd.DataFrame([dict(onset=1000, end=1008)]), 0, 1999)
        assert st["detected"] == 1, (kind, d.alarms)
        print(kind, "alarms", d.alarms, "stats", {k: round(v, 3) for k, v in st.items()})
    print("ok")
