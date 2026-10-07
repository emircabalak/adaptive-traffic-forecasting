"""Streaming replay: walk a speed series one 5-minute step at a time, as if it were live.

At step `now` the model forecasts now+1..now+12 from now-11..now. Each forecast goes into a
pending buffer keyed by its target time and is released only when that time arrives, so the
policy (detector + adaptation) can never see a label before wall-clock time reaches it. The
policy also only gets the series up to `now`.
"""
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

SEQ = 12


def time_of_day(index):
    """Fraction of the day, computed exactly like official generate_training_data.py."""
    v = index.values
    return ((v - v.astype("datetime64[D]")) / np.timedelta64(1, "D")).astype(np.float32)


def make_input(speed, tod, now, scaler):
    """(1, 2, N, 12) model input for the window ending at `now`, as official util/engine build it."""
    mean, std = scaler
    x = (speed[now - SEQ + 1: now + 1] - mean) / std  # zeros normalised too, like official code
    t = np.broadcast_to(tod[now - SEQ + 1: now + 1, None], x.shape)
    return torch.from_numpy(np.stack([x, t]).astype(np.float32))[None].transpose(2, 3)


def forecast(model, x):
    """Model output in normalised units, (12, N). Left pad by one step as engine.eval does."""
    return model(F.pad(x, (1, 0, 0, 0)))[0, :, :, 0]


class Static:
    """Variant (a): never updates. Other policies implement the same `step`."""

    def step(self, now, labels, past):
        """labels: list of (issue_t, k, pred_row, true_row) whose target time == now.
        past: speed[: now + 1]. Called before the forecast at `now`."""


def replay(model, speed, tod, scaler, t_start, t_end, policy=None, device="cuda:0"):
    """Run steps t_start..t_end (inclusive 'now' indices). Returns forecasts in mph,
    shape (steps, 12, N), where row i was issued at t_start + i for targets +1..+12."""
    policy = policy or Static()
    mean, std = scaler
    steps = t_end - t_start + 1
    preds = np.empty((steps, SEQ, speed.shape[1]), np.float32)
    pending = defaultdict(list)  # target_t -> [(issue_t, k)]
    lat_policy, lat_fc = np.empty(steps), np.empty(steps)
    model.eval()
    for i, now in enumerate(range(t_start, t_end + 1)):
        labels = [(s, k, preds[s - t_start, k], speed[now]) for s, k in pending.pop(now, [])]
        t0 = time.perf_counter()
        policy.step(now, labels, speed[: now + 1])
        t1 = time.perf_counter()
        with torch.no_grad():
            out = forecast(model, make_input(speed, tod, now, scaler).to(device))
        preds[i] = out.cpu().numpy() * std + mean
        for k in range(SEQ):
            pending[now + 1 + k].append((now, k))
        lat_policy[i], lat_fc[i] = t1 - t0, time.perf_counter() - t1
    return dict(preds=preds, latency_policy=lat_policy, latency_forecast=lat_fc)


def targets(speed, t_start, steps):
    """Ground truth aligned with replay preds: (steps, 12, N), row i = speed[t_start+i+1 .. +12]."""
    w = np.lib.stride_tricks.sliding_window_view(speed[t_start + 1: t_start + steps + SEQ], SEQ, axis=0)
    return w.transpose(0, 2, 1)


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))
    import data
    import gwn

    dev = "cpu"  # keep the GPU free for training; the check is about plumbing, not weights
    torch.manual_seed(0)
    model = gwn.build(dev)
    df = data.load_speed_df()
    speed, tod = df.values, time_of_day(df.index)  # float64, same as the npz files
    scaler = data.official_scaler()
    t0, n = data.test_start(len(speed)), 40

    # 1. replay forecasts == batch forecasts on the official test npz windows
    te = np.load(data.DATA / "METR-LA" / "test.npz")
    x = te["x"][:n].copy()
    x[..., 0] = (x[..., 0] - scaler[0]) / scaler[1]
    with torch.no_grad():
        batch = model(F.pad(torch.tensor(x, dtype=torch.float32).transpose(1, 3), (1, 0, 0, 0)))
    batch = batch[:, :, :, 0].numpy() * scaler[1] + scaler[0]
    out = replay(model, speed, tod, scaler, t0, t0 + n - 1, device=dev)
    assert np.allclose(out["preds"], batch, atol=1e-3), np.abs(out["preds"] - batch).max()
    assert np.array_equal(targets(speed, t0, n), te["y"][:n, :, :, 0])

    # 2. leak test: labels reach the policy exactly at their target time, never earlier
    class Recorder(Static):
        def __init__(self):
            self.seen = []

        def step(self, now, labels, past):
            assert len(past) == now + 1
            for s, k, p, y in labels:
                assert s + k + 1 == now
                self.seen.append((now, s, k, y.copy()))

    rec_a, rec_b = Recorder(), Recorder()
    replay(model, speed, tod, scaler, t0, t0 + n - 1, rec_a, dev)
    future = t0 + 25
    speed2 = speed.copy()
    speed2[future] += 50.0  # change the world at one future time
    out_b = replay(model, speed2, tod, scaler, t0, t0 + n - 1, rec_b, dev)
    before = lambda rec: [r for r in rec.seen if r[0] < future]
    assert all(a[:3] == b[:3] and np.array_equal(a[3], b[3]) for a, b in zip(before(rec_a), before(rec_b)))
    assert np.array_equal(out["preds"][: future - t0], out_b["preds"][: future - t0])
    assert any(r[0] == future and r[3][0] != speed[future][0] for r in rec_b.seen)
    print(f"ok: replay == batch on {n} test windows, no label before its time; "
          f"forecast latency {1000 * out['latency_forecast'].mean():.1f} ms/step on {dev}")
