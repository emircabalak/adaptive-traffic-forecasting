"""METR-LA loading. Raw file is the DCRNN release (pandas 0.15 HDF, unreadable by pandas 3)."""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tables

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW_H5 = DATA / "metr-la.h5"
ADJ_PKL = DATA / "sensor_graph" / "adj_mx.pkl"
SEQ_IN = SEQ_OUT = 12


def load_speed_df(path=RAW_H5):
    """(T, N) DataFrame of mph, 0.0 = missing. Reads the old pandas layout with pytables directly."""
    with tables.open_file(str(path)) as f:
        g = f.root.df
        index = pd.to_datetime(g.axis1[:])
        cols = [c.decode() for c in g.axis0[:]]
        assert [c.decode() for c in g.block0_items[:]] == cols
        return pd.DataFrame(g.block0_values[:], index=index, columns=cols)


def load_adj(path=ADJ_PKL):
    with open(path, "rb") as f:
        sensor_ids, id_to_ind, adj = pickle.load(f, encoding="latin1")
    return list(map(str, sensor_ids)), adj


def split_bounds(T):
    """Window counts exactly as official generate_training_data.py (70/10/20 by round())."""
    n = T - SEQ_IN - SEQ_OUT + 1
    n_test, n_train = round(n * 0.2), round(n * 0.7)
    n_val = n - n_test - n_train
    return n_train, n_val, n_test


def test_start(T):
    """Raw-series index of 'now' for test window 0. Window i: inputs t-11..t, targets t+1..t+12."""
    n_train, n_val, _ = split_bounds(T)
    return SEQ_IN - 1 + n_train + n_val


def official_scaler(npz_dir=DATA / "METR-LA"):
    """Mean/std exactly as official util.load_dataset: train-window inputs, zeros INCLUDED.
    Replay must use this so inputs match what the checkpoint was trained on."""
    x = np.load(Path(npz_dir) / "train.npz")["x"][..., 0]
    return float(x.mean()), float(x.std())


def speed_profile(cache=ROOT / "results" / "speed_profile.npy"):
    """Median speed per (5-minute slot of the day, sensor), (288, N), from the training period
    only. Used to tell whether a sensor is currently far below its usual speed."""
    if Path(cache).exists():
        return np.load(cache)
    df = load_speed_df()
    n_train, _, _ = split_bounds(len(df))
    v = df.values[: n_train + SEQ_IN - 1].astype(float)
    v[v == 0] = np.nan
    slot = np.arange(len(v)) % 288  # series starts at 00:00
    prof = np.stack([np.nanmedian(v[slot == k], 0) for k in range(288)])
    prof = np.where(np.isnan(prof), np.nanmedian(v, 0), prof)
    np.save(cache, prof)
    return prof


def masked(y):
    """Boolean mask of valid (non-missing) readings."""
    return y != 0.0


def convert_for_pandas3(src=RAW_H5, dst=DATA / "metr-la-pd3.h5"):
    """Re-save so the official generate_training_data.py (pd.read_hdf) works with pandas 3."""
    load_speed_df(src).to_hdf(dst, key="df")
    return dst


def check():
    df = load_speed_df()
    v = df.values
    T, N = v.shape
    assert (T, N) == (34272, 207), v.shape
    assert (np.diff(df.index.values) == np.timedelta64(5, "m")).all()
    ids, adj = load_adj()
    assert ids == list(df.columns), "sensor order mismatch between h5 and adj_mx.pkl"
    print(f"{df.index[0]} .. {df.index[-1]}, missing(0.0) frac {(v == 0).mean():.4f}, "
          f"all-missing steps {(v == 0).all(1).sum()}, adj nnz {(adj > 0).sum()}")
    print("split windows (train, val, test):", split_bounds(T), "test t0 index:", test_start(T))

    npz_dir = DATA / "METR-LA"
    if (npz_dir / "test.npz").exists():
        te = np.load(npz_dir / "test.npz")
        t0 = test_start(T)
        for i in (0, 1, 3000, len(te["x"]) - 1):
            t = t0 + i
            assert np.array_equal(te["x"][i, :, :, 0], v[t - 11: t + 1]), i
            assert np.array_equal(te["y"][i, :, :, 0], v[t + 1: t + 13]), i
        tr = np.load(npz_dir / "train.npz")["x"][..., 0]  # same as official_scaler()
        print(f"official scaler mean/std {tr.mean():.4f}/{tr.std():.4f}; "
              f"masked train mean/std {tr[tr != 0].mean():.4f}/{tr[tr != 0].std():.4f}")
        print("test-series reconstruction matches npz windows")


if __name__ == "__main__":
    check()
