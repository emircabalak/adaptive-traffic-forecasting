"""Learn the LearnedBlend starting parameters by streaming once through the training period.

The training period has real slowdowns and no synthetic incidents, so theta starts from real
traffic only. Resumable: theta, the optimizer state and the last finished step are saved to
results/blend_theta.json after every chunk; rerunning continues from there.
Usage: python scripts/pretrain_blend.py [--period train|val] [--log results/pretrain_blend.log]
--period val learns on the validation period instead (out of sample for Graph WaveNet, real
traffic only, no synthetic incidents) and writes results/blend_theta_val.json.
Detached: powershell -ExecutionPolicy Bypass -File scripts/start_pretrain_blend.ps1
"""
import json
import os
import sys
from pathlib import Path

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"  # see IDEAS log, forrtl window-CLOSE abort
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_experiments as rx  # noqa: E402  (src imports, blocks idle sleep)
from run_experiments import adapt, data, gwn, replay  # noqa: E402

CHUNK = 3000

if "--log" in sys.argv:
    i = sys.argv.index("--log")
    sys.stdout = sys.stderr = open(Path(sys.argv[i + 1]).resolve(), "a", buffering=1)

import torch  # noqa: E402

period = "train"
if "--period" in sys.argv:
    period = sys.argv[sys.argv.index("--period") + 1]

ckpt = sorted((ROOT / "checkpoints").glob("metr_exp1_best_*.pth"))[-1]
df = data.load_speed_df()
speed, tod = df.values, replay.time_of_day(df.index)
scaler = data.official_scaler()
n_train, n_val, _ = data.split_bounds(len(speed))
if period == "train":
    t0, t1 = replay.SEQ - 1, replay.SEQ - 1 + n_train - 1 - replay.SEQ  # targets stay in the period
    f = adapt.THETA_FILE
else:
    t0 = replay.SEQ - 1 + n_train
    t1 = t0 + n_val - 1 - replay.SEQ
    f = adapt.THETA_FILE.with_name("blend_theta_val.json")
state = json.loads(f.read_text()) if f.exists() else {"done_until": t0 - 1, "theta": None}
model = gwn.build(rx.DEV, ckpt)
pol = adapt.LearnedBlend(model, tod, scaler, rx.DEV, theta=state["theta"], learn=True, **rx.detector_config())
if state.get("opt"):
    opt = state["opt"]
    pol.opt.state[pol.theta] = {"step": torch.tensor(float(opt["step"])),
                                "exp_avg": torch.tensor(opt["m"], device=rx.DEV),
                                "exp_avg_sq": torch.tensor(opt["v"], device=rx.DEV)}

start = state["done_until"] + 1
print(f"pretraining theta on steps {start}..{t1}", flush=True)
while start <= t1:
    end = min(start + CHUNK - 1, t1)
    replay.replay(model, speed, tod, scaler, start, end, pol, rx.DEV)
    st = pol.opt.state[pol.theta]
    losses = [u[2] for u in pol.updates[-500:]]
    state = {"done_until": end, "theta": pol.theta.detach().cpu().tolist(),
             "opt": {"step": float(st["step"]), "m": st["exp_avg"].cpu().tolist(), "v": st["exp_avg_sq"].cpu().tolist()},
             "ckpt": str(ckpt), "recent_loss": sum(losses) / max(len(losses), 1)}
    tmp = f.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, f)
    print(f"done until {end}: theta {[round(t, 3) for t in state['theta']]}, recent loss {state['recent_loss']:.4f}",
          flush=True)
    start = end + 1
print("pretraining finished", flush=True)
