"""Static streaming replay of the frozen baseline over the whole test split, scored with our
masked metrics and compared with the paper. Writes results/baseline_eval.csv."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import data  # noqa: E402
import gwn  # noqa: E402
import metrics  # noqa: E402
import replay  # noqa: E402

PAPER = {"15min": (2.69, 5.15, 6.90), "30min": (3.07, 6.22, 8.37), "60min": (3.53, 7.37, 10.01)}

ckpt = sys.argv[1] if len(sys.argv) > 1 else sorted((ROOT / "checkpoints").glob("metr_exp1_best_*.pth"))[-1]
model = gwn.build("cuda:0", ckpt)
df = data.load_speed_df()
speed, tod = df.values, replay.time_of_day(df.index)
t0 = data.test_start(len(speed))
t_end = len(speed) - 1 - replay.SEQ
out = replay.replay(model, speed, tod, data.official_scaler(), t0, t_end)
y = replay.targets(speed, t0, t_end - t0 + 1)

rows = []
for h, m in metrics.per_horizon(out["preds"], y).items():
    p = PAPER[h]
    rows.append(dict(horizon=h, mae=m["mae"], rmse=m["rmse"], mape=m["mape"],
                     paper_mae=p[0], paper_rmse=p[1], paper_mape=p[2]))
res = pd.DataFrame(rows)
res.to_csv(ROOT / "results" / "baseline_eval.csv", index=False, float_format="%.4f")
print(f"checkpoint: {ckpt}\nMAPE floor {metrics.MAPE_FLOOR} mph (paper MAPE has no floor)")
print(res.to_string(index=False, float_format="%.3f"))
print(f"forecast latency {1000 * out['latency_forecast'].mean():.2f} ms/step")
