"""Drift detection on the t+1 forecast error, plus detection delay / false alarm accounting.

The signal is built from the labels released at `now` by the replay buffer, using only the
horizon-1 forecast (issued at now-1), because it is the first label to arrive. MAPE is never
used here: it explodes on missing zeros and near-zero speeds.
"""
import numpy as np
from river import drift

from replay import Static

STEPS_PER_DAY = 288


def t1_error(labels, agg="mean", topk=5):
    """Masked absolute t+1 error across sensors, aggregated to one number (nan if no valid data)."""
    for s, k, pred, true in labels:
        if k == 0:
            m = true != 0
            if not m.any():
                return np.nan
            e = np.abs(pred[m] - true[m])
            return e.mean() if agg == "mean" else np.sort(e)[-topk:].mean()
    return np.nan


def make_detector(kind="adwin", **params):
    if kind == "adwin":
        return drift.ADWIN(**{"clock": 1, **params})  # default clock=32 checks only every 32 steps (2.7 h)
    if kind == "ph":
        return drift.PageHinkley(**{"mode": "up", **params})
    raise ValueError(kind)


class DetectOnly(Static):
    """Feeds the t+1 error to a detector and logs alarms. No model updates."""

    def __init__(self, kind="adwin", agg="mean", **params):
        self.det, self.agg = make_detector(kind, **params), agg
        self.signal, self.alarms = [], []

    def step(self, now, labels, past):
        x = t1_error(labels, self.agg)
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
    assert np.isclose(t1_error(lab([50, 10, 3], [60, 0, 1])), 6.0)  # missing sensor ignored
    assert np.isnan(t1_error(lab([1, 2], [0, 0])))

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
