"""Disruption scenarios.

Synthetic: k graph-adjacent sensors drop to a fraction of their observed speed for a few steps.
The change is written into the raw series before replay, so it shows up both in later model
inputs and in the targets. Missing readings (0.0) stay 0.0.

Real: non-recurring slowdowns mined from the data by comparing each reading with the median of
the same weekday and time slot across all weeks.
"""
from collections import deque

import numpy as np
import pandas as pd

STEPS_PER_DAY = 288


def neighbours(adj):
    """Undirected neighbour lists from the (directed) thresholded Gaussian adjacency."""
    a = (adj > 0) | (adj.T > 0)
    np.fill_diagonal(a, False)
    return [np.flatnonzero(r) for r in a]


def bfs_cluster(nbrs, seed, k):
    """First k nodes reached by BFS from seed, or None if its component is smaller than k."""
    seen, q = [seed], deque([seed])
    while q and len(seen) < k:
        for v in map(int, nbrs[q.popleft()]):
            if v not in seen:
                seen.append(v)
                q.append(v)
                if len(seen) == k:
                    break
    return seen if len(seen) == k else None


def factor_profile(duration, factor, ramp):
    """Multiplier per step: linear ramp from 1 to factor over `ramp` steps, hold, ramp back.
    The total length is duration + 2 * ramp, so the full-strength part always lasts `duration`."""
    up = np.linspace(1, factor, ramp + 2)[1:-1]
    return np.concatenate([up, np.full(duration, factor), up[::-1]])


def schedule(adj, speed, t_start, t_end, seed, k=5, duration=9, factor=0.15, ramp=0,
             per_day=1, hours=(6, 20), min_valid=0.8):
    """One incident per day (by default) at a random time inside `hours`, on k adjacent sensors
    that have at least `min_valid` non-missing readings during the incident. Deterministic in seed."""
    rng = np.random.default_rng(seed)
    nbrs = neighbours(adj)
    length = duration + 2 * ramp
    rows = []
    first_day = t_start - t_start % STEPS_PER_DAY
    for day in range(first_day, t_end, STEPS_PER_DAY):
        slots = rng.choice(np.arange(hours[0] * 12, hours[1] * 12 - length), per_day, replace=False)
        for slot in sorted(slots):
            onset = day + slot
            if onset < t_start or onset + length > t_end:
                continue
            for _ in range(100):  # resample until a valid cluster turns up
                c = bfs_cluster(nbrs, int(rng.integers(len(nbrs))), k)
                if c and (speed[onset: onset + length, c] != 0).mean() >= min_valid:
                    break
            else:
                continue
            rows.append(dict(onset=onset, end=onset + length - 1, sensors=c,
                             factor=factor, duration=duration, ramp=ramp))
    return pd.DataFrame(rows)


def inject(speed, events):
    """Copy of speed with every event applied. Zeros (missing) stay zero."""
    out = speed.copy()
    for e in events.itertuples():
        f = factor_profile(e.duration, e.factor, e.ramp)
        out[e.onset: e.end + 1, e.sensors] *= f[:, None]
    return out


def mine_slowdowns(df, min_drop=20.0, max_ratio=0.6, min_steps=3, smooth=3):
    """Candidate non-recurring slowdowns: runs of >= min_steps where the (smoothed) speed is at
    least min_drop mph below, and below max_ratio times, the same-weekday same-slot median.
    Uses all weeks for the reference; this is offline labelling for evaluation, never model input."""
    v = df.where(df != 0)
    sm = v.rolling(smooth, center=True, min_periods=2).median()
    key = [df.index.dayofweek, df.index.hour * 12 + df.index.minute // 5]
    # ponytail: median includes the day itself; with ~17 weeks per weekday one outlier barely moves it
    ref = v.groupby(key).transform("median")
    hit = ((ref - sm) >= min_drop) & (sm < max_ratio * ref)
    rows = []
    for j, col in enumerate(df.columns):
        h = hit[col].to_numpy()
        if not h.any():
            continue
        edges = np.flatnonzero(np.diff(np.r_[0, h.astype(int), 0]))
        for s, e in zip(edges[::2], edges[1::2]):
            if e - s >= min_steps:
                rows.append(dict(sensor_idx=j, sensor_id=col, onset=s, end=e - 1,
                                 start_time=df.index[s], min_speed=sm[col].iloc[s:e].min(),
                                 ref_speed=ref[col].iloc[s:e].mean()))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import data

    df = data.load_speed_df()
    speed = df.values
    _, adj = data.load_adj()
    t0, t1 = data.test_start(len(speed)), len(speed) - 13

    ev = schedule(adj, speed, t0, t1, seed=0, ramp=2)
    ev2 = schedule(adj, speed, t0, t1, seed=0, ramp=2)
    assert ev.equals(ev2), "schedule must be deterministic in seed"
    nbrs = neighbours(adj)
    for e in ev.itertuples():
        c = e.sensors
        assert len(set(c)) == 5 and all(any(u in nbrs[v] for u in c if u != v) for v in c)
    hit = inject(speed, ev)
    e = ev.iloc[0]
    mid = e.onset + e.ramp + 4
    s = e.sensors
    ok = speed[mid, s] != 0
    assert np.allclose(hit[mid, s][ok], 0.15 * speed[mid, s][ok])
    assert (hit[speed == 0] == 0).all()
    untouched = np.ones(speed.shape, bool)
    for x in ev.itertuples():
        untouched[x.onset: x.end + 1, x.sensors] = False
    assert np.array_equal(hit[untouched], speed[untouched])
    print(f"ok: {len(ev)} synthetic incidents over the test period, e.g. onset {df.index[e.onset]} sensors {s}")

    real = mine_slowdowns(df)
    real_test = real[real.onset >= t0]
    print(f"mined {len(real)} slowdown candidates ({len(real_test)} in the test period); longest:")
    print(real_test.assign(steps=real_test.end - real_test.onset + 1)
          .sort_values("steps", ascending=False).head(8).to_string(index=False))
