"""Masked error metrics. Ground truth 0.0 means missing and is excluded everywhere."""
import numpy as np

MAPE_FLOOR = 5.0  # mph; keeps near-zero speeds from blowing MAPE up. Report this value.
HORIZONS = {"15min": 2, "30min": 5, "60min": 11}  # index into the 12 output steps


def masked_metrics(pred, true, floor=MAPE_FLOOR, extra_mask=None):
    """MAE, RMSE, MAPE (%) over positions where true != 0 (and extra_mask, if given)."""
    pred, true = np.asarray(pred, float), np.asarray(true, float)
    m = true != 0.0
    if extra_mask is not None:
        m &= extra_mask
    if not m.any():
        return dict(mae=np.nan, rmse=np.nan, mape=np.nan, n=0)
    e = pred[m] - true[m]
    return dict(mae=np.abs(e).mean(), rmse=np.sqrt((e ** 2).mean()),
                mape=100 * (np.abs(e) / np.maximum(true[m], floor)).mean(), n=int(m.sum()))


def per_horizon(pred, true, **kw):
    """pred/true shaped (samples, 12, nodes). Returns {name: metrics} for 15/30/60 min."""
    return {h: masked_metrics(pred[:, i], true[:, i], **kw) for h, i in HORIZONS.items()}


if __name__ == "__main__":
    t = np.array([[60.0, 0.0, 2.0]])
    r = masked_metrics(np.array([[57.0, 40.0, 4.0]]), t)
    assert r["n"] == 2 and np.isclose(r["mae"], 2.5)  # the 0.0 reading is ignored
    assert np.isclose(r["mape"], 100 * (3 / 60 + 2 / 5) / 2)  # 2 mph uses the 5 mph floor
    assert np.isnan(masked_metrics(np.ones(3), np.zeros(3))["mae"])
    print("ok")
