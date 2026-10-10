"""Online update policies for the replay loop.

HeadAdapt (variant c, ours): on a detector alarm, a few gradient steps on the output head only
(end_conv_1, end_conv_2). Loss = masked Huber + beta * ||theta_head - theta_head_0||^2, with
gradient clipping. Optionally resets the head to theta_0 once the t+1 error is back in its
normal band for M steps.

NaiveFT (variant b): one gradient step on all weights at every step, no anchor, no detector.

Training windows come only from `past` (the series up to now). A target later than now is
simply not in `past`, so it is padded with 0.0, which every loss treats as missing. The model
stays in eval() mode throughout, so BatchNorm statistics and dropout never change.
"""
import time

import numpy as np
import torch
import torch.nn.functional as F

import data
import gwn
from drift import DetectOnly
from replay import SEQ, Static, make_input


def window_batch(past, tod, scaler, issues, device):
    """Inputs (B, 2, N, 12) and targets (B, 12, N) in mph for windows issued at `issues`.
    Targets after len(past) - 1 (i.e. after now) are 0.0 = masked."""
    xs, ys = [], []
    now = len(past) - 1
    for s in issues:
        xs.append(make_input(past, tod, s, scaler))
        y = np.zeros((SEQ, past.shape[1]))
        known = min(SEQ, now - s)
        y[:known] = past[s + 1: s + 1 + known]
        ys.append(y)
    return torch.cat(xs).to(device), torch.tensor(np.stack(ys), dtype=torch.float32, device=device)


def masked_huber(pred_norm, y, scaler, delta):
    """Huber in normalised units over non-missing targets."""
    mean, std = scaler
    m = y != 0
    if not m.any():
        return pred_norm.sum() * 0
    return F.huber_loss(pred_norm[m], ((y - mean) / std)[m], delta=delta)


def predict_norm(model, x):
    return model(F.pad(x, (1, 0, 0, 0)))[:, :, :, 0]  # (B, 12, N), normalised units


class HeadAdapt(DetectOnly):
    def __init__(self, model, tod, scaler, device="cuda:0", steps=5, lr=1e-3, beta=1.0,
                 huber_delta=1.0, n_windows=6, clip=5.0, reset=False, band=None, reset_after=6,
                 kind="adwin", agg="mean", **det_params):
        super().__init__(kind, agg, **det_params)
        self.model, self.tod, self.scaler, self.device = model, tod, scaler, device
        self.steps, self.lr, self.beta, self.delta = steps, lr, beta, huber_delta
        self.n_windows, self.clip = n_windows, clip
        self.reset, self.band, self.reset_after = reset, band, reset_after  # band = (mean, std, margin)
        for p in model.parameters():
            p.requires_grad_(False)
        self.head = gwn.head_params(model)
        for p in self.head:
            p.requires_grad_(True)
        self.theta0 = [p.detach().clone() for p in self.head]
        self.adapted, self.calm = False, 0
        self.updates, self.resets = [], []  # (now, seconds, final loss), now

    def step(self, now, labels, past):
        alarm = super().step(now, labels, past)
        if alarm:
            self.update(now, past)
        elif self.reset and self.adapted:
            x = self.signal[-1][1]
            mean, std, margin = self.band
            self.calm = self.calm + 1 if (not np.isnan(x) and x <= mean + margin * std) else 0
            if self.calm >= self.reset_after:
                with torch.no_grad():
                    for p, p0 in zip(self.head, self.theta0):
                        p.copy_(p0)
                self.adapted, self.calm = False, 0
                self.resets.append(now)

    def update(self, now, past):
        t0 = time.perf_counter()
        issues = range(now - self.n_windows, now)  # most recent windows with >= 1 known target
        x, y = window_batch(past, self.tod, self.scaler, issues, self.device)
        opt = torch.optim.Adam(self.head, lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            loss = masked_huber(predict_norm(self.model, x), y, self.scaler, self.delta)
            if self.beta:
                loss = loss + self.beta * sum(((p - p0) ** 2).sum() for p, p0 in zip(self.head, self.theta0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.head, self.clip)
            opt.step()
        self.adapted, self.calm = True, 0
        self.updates.append((now, time.perf_counter() - t0, loss.item()))


def last_valid(x, scaler):
    """Most recent non-missing normalised input speed per window and sensor, (B, N), and a mask
    of where one exists. Missing readings are 0 mph, which normalise to -mean/std."""
    z0 = -scaler[0] / scaler[1]
    sp = x[:, 0]  # (B, N, 12)
    ok = (sp - z0).abs() > 1e-6
    idx = torch.where(ok, torch.arange(sp.shape[-1], device=sp.device), -1).amax(-1)  # last valid step
    last = sp.gather(-1, idx.clamp(min=0)[..., None])[..., 0]
    return last, idx >= 0


class NodeAdapt(DetectOnly):
    """Per-sensor output adapter, trained on alarm. Graph WaveNet itself stays frozen.
    y' = y + alpha_v * (last_v - y) + b_v in normalised units; alpha = b = 0 is the static model.
    per_horizon=True gives each horizon its own alpha and b, shape (12, N)."""

    def __init__(self, model, tod, scaler, device="cuda:0", steps=20, lr=0.05, beta=0.01,
                 huber_delta=1.0, n_windows=6, decay=1.0, per_horizon=False, max_horizon=SEQ, local_z=None,
                 active_steps=0, gate=None, anomaly_ratio=None, track_z=3.0, max_track=144, band=None,
                 kind="adwin", agg="mean", **det_params):
        super().__init__(kind, agg, **det_params)
        # anomaly_ratio: after an alarm, keep updating every step for as long as any tracked sensor
        # (recent confirmed z above track_z at the alarm) reads below anomaly_ratio times its usual
        # speed for that time of day (data.speed_profile, training data). Capped at max_track steps.
        self.anomaly_ratio, self.track_z, self.max_track = anomaly_ratio, track_z, max_track
        self.track, self.track_until = None, -1
        self.profile = data.speed_profile() if anomaly_ratio else None
        # local_z: train only sensors whose recent (neighbour-confirmed) z exceeded it, plus neighbours.
        # active_steps: after an alarm, keep updating every step for this many steps (no decay meanwhile).
        self.local_z, self.active_steps, self.active_until = local_z, active_steps, -1
        # gate: during the active phase, only update while the frozen model's mean absolute error
        # (normalised units) on the trained sensors' known targets is at least this large.
        self.gate = gate
        self.model, self.tod, self.scaler, self.device = model, tod, scaler, device
        self.steps, self.lr, self.beta, self.delta = steps, lr, beta, huber_delta
        self.n_windows, self.decay = n_windows, decay
        self.hmask = (torch.arange(SEQ, device=device) < max_horizon).float()[:, None]  # (12, 1)
        for p in model.parameters():
            p.requires_grad_(False)
        shape = (SEQ, model.nodevec1.shape[0]) if per_horizon else (model.nodevec1.shape[0],)
        self.alpha = torch.zeros(shape, device=device, requires_grad=True)
        self.b = torch.zeros(shape, device=device, requires_grad=True)
        self.updates = []

    def _apply(self, base, x):
        last, ok = last_valid(x, self.scaler)
        pull = torch.where(ok[:, None, :], last[:, None, :] - base, torch.zeros_like(base))
        return base + self.hmask * (self.alpha * pull + self.b)

    def adjust(self, out, x):
        return self._apply(out, x)

    def anomalous(self, now, past):
        """True while a tracked sensor is still far below its usual speed."""
        if self.track is None or now > self.track_until:
            return False
        v = past[now, self.track]
        ok = v != 0
        if not ok.any():
            return True  # missing readings: keep the current state
        usual = self.profile[now % self.profile.shape[0], self.track]
        return bool((v[ok] < self.anomaly_ratio * usual[ok]).any())

    def step(self, now, labels, past):
        if super().step(now, labels, past):
            self.active_until = now + self.active_steps
            if self.anomaly_ratio and self.recent:
                hot = np.max(np.nan_to_num(np.stack(self.recent), nan=-np.inf), 0) > self.track_z
                if hot.any():
                    self.track, self.track_until = np.flatnonzero(hot), now + self.max_track
            self.update(now, past)
        elif now <= self.active_until or (self.anomaly_ratio and self.anomalous(now, past)):
            self.update(now, past, gated=True)
        elif self.decay < 1:
            with torch.no_grad():
                self.alpha.mul_(self.decay)
                self.b.mul_(self.decay)

    def sensor_mask(self):
        """(N,) 1/0 mask of sensors to train, or None for all."""
        if self.local_z is None or not self.recent:
            return None
        hot = np.max(np.nan_to_num(np.stack(self.recent), nan=-np.inf), 0) > self.local_z
        if self.nbrs is not None:
            nb = self.nbrs[hot].ravel()
            hot[nb[nb < len(hot)]] = True
        return torch.tensor(hot, dtype=torch.float32, device=self.device)

    def update(self, now, past, gated=False):
        t0 = time.perf_counter()
        x, y = window_batch(past, self.tod, self.scaler, range(now - self.n_windows, now), self.device)
        with torch.no_grad():
            base = predict_norm(self.model, x)
        smask = self.sensor_mask()
        if gated and self.gate is not None:
            m = (y != 0) & (self.hmask > 0) & ((smask > 0) if smask is not None else True)
            yn = (y - self.scaler[0]) / self.scaler[1]
            if not m.any() or (base - yn).abs()[m].mean() < self.gate:
                self.active_until = -1  # the static model is right again: stop, let decay take over
                return
        opt = torch.optim.Adam([self.alpha, self.b], lr=self.lr)
        for _ in range(self.steps):
            opt.zero_grad()
            yh = y * self.hmask  # horizons past max_horizon count as missing
            if smask is not None:
                yh = yh * smask  # sensors outside the flagged area count as missing too
            loss = masked_huber(self._apply(base, x), yh, self.scaler, self.delta)
            loss = loss + self.beta * ((self.alpha ** 2).sum() + (self.b ** 2).sum())
            loss.backward()
            opt.step()
        self.updates.append((now, time.perf_counter() - t0, loss.item()))


class NaiveFT(Static):
    """Variant (b): every step, one Adam step on all weights using the windows whose
    targets became (partly) known, no anchor, no gating."""

    def __init__(self, model, tod, scaler, device="cuda:0", lr=1e-3, huber_delta=1.0, n_windows=1, clip=5.0):
        self.model, self.tod, self.scaler, self.device = model, tod, scaler, device
        self.delta, self.n_windows, self.clip = huber_delta, n_windows, clip
        self.params = [p for p in model.parameters()]
        for p in self.params:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(self.params, lr=lr)  # state persists across steps, as plain online FT would
        self.updates = []

    def step(self, now, labels, past):
        if not labels:
            return
        t0 = time.perf_counter()
        x, y = window_batch(past, self.tod, self.scaler, range(now - self.n_windows, now), self.device)
        self.opt.zero_grad()
        loss = masked_huber(predict_norm(self.model, x), y, self.scaler, self.delta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, self.clip)
        self.opt.step()
        self.updates.append((now, time.perf_counter() - t0, loss.item()))


class Persist(DetectOnly):
    """Rule-based reference, not a parameter update: on selected sensors the forecast is pulled
    towards the last observed speed, y' = y + w * (last - y), for horizons up to max_horizon.
    mode "all": every sensor, always (plain persistence when w = 1).
    mode "oracle": the sensors and times in `windows` [(start, end, sensors)], for diagnostics.
    mode "flagged": after a detector alarm, sensors whose neighbour-confirmed z went above
    flag_z in the last few steps, each for `hold` steps."""

    def __init__(self, model, tod, scaler, device="cuda:0", mode="all", w=1.0, max_horizon=SEQ,
                 windows=(), flag_z=3.0, hold=12, anomaly_ratio=None, band=None, kind="adwin", agg="mean",
                 **det_params):
        super().__init__(kind, agg, **det_params)
        self.scaler, self.device, self.mode, self.w = scaler, device, mode, w
        self.hmask = (torch.arange(SEQ, device=device) < max_horizon).float()[:, None]
        self.windows, self.flag_z, self.hold = list(windows), flag_z, hold
        self.until = np.full(model.nodevec1.shape[0], -1)
        # anomaly_ratio: a flagged sensor is only pulled while it reads below this fraction of its
        # usual speed for the time of day (data.speed_profile), so falsely flagged ones aren't
        self.anomaly_ratio = anomaly_ratio
        self.profile = data.speed_profile() if anomaly_ratio else None
        self.cur = None
        self.now, self.updates = -1, []

    def step(self, now, labels, past):
        self.now, self.cur = now, past[now]
        if super().step(now, labels, past) and self.mode == "flagged" and self.recent:
            hot = np.max(np.nan_to_num(np.stack(self.recent), nan=-np.inf), 0) > self.flag_z
            self.until[hot] = now + self.hold

    def active(self):
        n = len(self.until)
        if self.mode == "all":
            return np.ones(n, bool)
        if self.mode == "oracle":
            a = np.zeros(n, bool)
            for start, end, sensors in self.windows:
                if start <= self.now <= end:
                    a[list(sensors)] = True
            return a
        a = self.until >= self.now
        if self.anomaly_ratio:
            usual = self.profile[self.now % self.profile.shape[0]]
            a &= (self.cur != 0) & (self.cur < self.anomaly_ratio * usual)
        return a

    def adjust(self, out, x):
        a = torch.tensor(self.active(), dtype=torch.float32, device=out.device)
        if not a.any():
            return out
        last, ok = last_valid(x, self.scaler)
        pull = torch.where(ok[:, None, :], last[:, None, :] - out, torch.zeros_like(out))
        return out + self.w * self.hmask * a * pull


# name -> factory(model, tod, scaler, device, band=..., **detector and policy kwargs)
POLICIES = {
    "static": lambda model, tod, scaler, device, band=None, **det: DetectOnly(**det),
    "head": HeadAdapt,
    "node": NodeAdapt,
    "persist": Persist,
    "naive": lambda model, tod, scaler, device, band=None, **det: NaiveFT(model, tod, scaler, device),
}


if __name__ == "__main__":
    import data
    from replay import replay, time_of_day

    dev = "cpu"
    torch.manual_seed(0)
    df = data.load_speed_df()
    speed, tod = df.values, time_of_day(df.index)
    scaler = data.official_scaler()
    t0 = data.test_start(len(speed))

    # window_batch never exposes data after now
    past = speed[: t0 + 1]
    x, y = window_batch(past, tod, scaler, [t0 - 3, t0 - 1], dev)
    assert (y[0, 3:] == 0).all() and (y[1, 1:] == 0).all()
    assert np.allclose(y[0, :3].numpy(), speed[t0 - 2: t0 + 1], atol=1e-3)

    # head-only update: body weights bit-identical, head moved, anchor pulls back
    model = gwn.build(dev)
    body0 = {n: p.detach().clone() for n, p in model.named_parameters() if n.split(".")[0] not in gwn.HEAD}
    pol = HeadAdapt(model, tod, scaler, dev, steps=3, beta=1.0)
    pol.update(t0, past)
    assert all(torch.equal(p, body0[n]) for n, p in model.named_parameters() if n in body0)
    assert any(not torch.equal(p, p0) for p, p0 in zip(pol.head, pol.theta0))
    assert not model.training
    print(f"ok: head update {1000 * pol.updates[-1][1]:.0f} ms on {dev}, body untouched")

    # short end-to-end replay with each policy
    for name, mk in [("naive", lambda m: NaiveFT(m, tod, scaler, dev)),
                     ("ours", lambda m: HeadAdapt(m, tod, scaler, dev, reset=True, band=(2.5, 0.5, 1.0)))]:
        m = gwn.build(dev)
        p = mk(m)
        out = replay(m, speed, tod, scaler, t0, t0 + 30, p, dev)
        print(f"{name}: {len(p.updates)} updates, policy {1000 * out['latency_policy'].mean():.0f} ms/step")
