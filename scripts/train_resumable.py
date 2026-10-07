"""Resumable version of the official training loop.

Same loop as third_party/Graph-WaveNet/train.py (upstream engine.trainer, util.load_dataset and
paper hyperparameters), plus a state file written after every epoch: model, optimizer, epoch,
validation history and RNG states. Starting it again continues from the last finished epoch.

If no state file exists but an earlier train.py run left epoch checkpoints, it continues from the
latest of those. Validation losses come from that run's log, and Adam restarts with fresh moments
(the only difference from an uninterrupted run).

Usage (pythonw has no console, so pass --log):
  pythonw -u scripts/train_resumable.py --log results/baseline_train.log
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

os.environ["FOR_DISABLE_CONSOLE_CTRL_HANDLER"] = "1"  # see IDEAS log, forrtl window-CLOSE abort
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=100)
ap.add_argument("--save", default=str(ROOT / "checkpoints" / "metr"))
ap.add_argument("--log", default=None)
ap.add_argument("--prev_log", default=str(ROOT / "results" / "baseline_train.log"),
                help="log of an earlier train.py run, used only when bootstrapping from its checkpoints")
ap.add_argument("--device", default="cuda:0")
ap.add_argument("--print_every", type=int, default=100)
a = ap.parse_args()
if a.log:
    sys.stdout = sys.stderr = open(Path(a.log).resolve(), "a", buffering=1)

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(ROOT / "src"))
import gwn  # noqa: E402

gwn.patch_upstream()
import util  # noqa: E402  (upstream)
from engine import trainer  # noqa: E402  (upstream)

# paper configuration = train.py defaults + README flags
BATCH, LR, DROPOUT, WD, NHID = 64, 0.001, 0.3, 0.0001, 32
save = Path(a.save)
save.parent.mkdir(parents=True, exist_ok=True)
state_file = save.with_name(save.name + "_state.pt")
device = torch.device(a.device)

_, _, adj = util.load_adj(str(ROOT / "data/sensor_graph/adj_mx.pkl"), "doubletransition")
dl = util.load_dataset(str(ROOT / "data/METR-LA"), BATCH, BATCH, BATCH)
scaler = dl["scaler"]
supports = [torch.tensor(i).to(device) for i in adj]
engine = trainer(scaler, 2, 12, 207, NHID, DROPOUT, LR, WD, device, supports, True, True, None)

his_loss, start = [], 1
if state_file.exists():
    st = torch.load(state_file, map_location=device, weights_only=False)
    engine.model.load_state_dict(st["model"])
    engine.optimizer.load_state_dict(st["optimizer"])
    his_loss, start = st["his_loss"], st["epoch"] + 1
    np.random.set_state(st["np_rng"])
    torch.set_rng_state(st["torch_rng"].cpu())
    print(f"resumed from {state_file} after epoch {st['epoch']}", flush=True)
else:
    ckpts = {int(m.group(1)): p for p in save.parent.glob(save.name + "_epoch_*.pth")
             if (m := re.search(r"_epoch_(\d+)_", p.name))}
    if ckpts:
        last = max(ckpts)
        val = {}
        for line in open(a.prev_log, encoding="utf-8", errors="ignore"):
            m = re.match(r"Epoch: (\d+), Train Loss.*Valid Loss: ([\d.]+)", line)
            if m:
                val[int(m.group(1))] = float(m.group(2))
        assert all(e in val for e in range(1, last + 1)), "missing validation losses in prev_log"
        engine.model.load_state_dict(torch.load(ckpts[last], map_location=device))
        his_loss, start = [val[e] for e in range(1, last + 1)], last + 1
        print(f"bootstrapped from {ckpts[last].name} (epoch {last}); Adam state starts fresh", flush=True)

print(f"training epochs {start}..{a.epochs}", flush=True)
for i in range(start, a.epochs + 1):
    train_loss, train_mape, train_rmse = [], [], []
    t1 = time.time()
    dl["train_loader"].shuffle()
    for it, (x, y) in enumerate(dl["train_loader"].get_iterator()):
        trainx = torch.Tensor(x).to(device).transpose(1, 3)
        trainy = torch.Tensor(y).to(device).transpose(1, 3)
        m = engine.train(trainx, trainy[:, 0, :, :])
        train_loss.append(m[0]); train_mape.append(m[1]); train_rmse.append(m[2])
        if it % a.print_every == 0:
            print(f"Iter: {it:03d}, Train Loss: {m[0]:.4f}, Train MAPE: {m[1]:.4f}, Train RMSE: {m[2]:.4f}", flush=True)
    t2 = time.time()
    valid = []
    s1 = time.time()
    for x, y in dl["val_loader"].get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        testy = torch.Tensor(y).to(device).transpose(1, 3)
        valid.append(engine.eval(testx, testy[:, 0, :, :]))
    print(f"Epoch: {i:03d}, Inference Time: {time.time() - s1:.4f} secs")
    vl, vm, vr = np.mean(valid, axis=0)
    his_loss.append(vl)
    print(f"Epoch: {i:03d}, Train Loss: {np.mean(train_loss):.4f}, Train MAPE: {np.mean(train_mape):.4f}, "
          f"Train RMSE: {np.mean(train_rmse):.4f}, Valid Loss: {vl:.4f}, Valid MAPE: {vm:.4f}, "
          f"Valid RMSE: {vr:.4f}, Training Time: {t2 - t1:.4f}/epoch", flush=True)
    torch.save(engine.model.state_dict(), f"{save}_epoch_{i}_{round(vl, 2)}.pth")
    tmp = state_file.with_suffix(".tmp")
    torch.save(dict(model=engine.model.state_dict(), optimizer=engine.optimizer.state_dict(), epoch=i,
                    his_loss=his_loss, np_rng=np.random.get_state(), torch_rng=torch.get_rng_state()), tmp)
    os.replace(tmp, state_file)  # atomic, so a crash mid-save never corrupts the state

# test, as in train.py
bestid = int(np.argmin(his_loss))
best = next(save.parent.glob(f"{save.name}_epoch_{bestid + 1}_*.pth"))
engine.model.load_state_dict(torch.load(best, map_location=device))
engine.model.eval()
realy = torch.Tensor(dl["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]
outputs = []
for x, y in dl["test_loader"].get_iterator():
    with torch.no_grad():
        outputs.append(engine.model(torch.Tensor(x).to(device).transpose(1, 3)).transpose(1, 3).squeeze())
yhat = torch.cat(outputs, dim=0)[: realy.size(0)]
print("Training finished")
print(f"The valid loss on best model is {his_loss[bestid]:.4f} (epoch {bestid + 1})")
amae, amape, armse = [], [], []
for h in range(12):
    m = util.metric(scaler.inverse_transform(yhat[:, :, h]), realy[:, :, h])
    print(f"Evaluate best model on test data for horizon {h + 1:d}, Test MAE: {m[0]:.4f}, Test MAPE: {m[1]:.4f}, Test RMSE: {m[2]:.4f}")
    amae.append(m[0]); amape.append(m[1]); armse.append(m[2])
print(f"On average over 12 horizons, Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}")
torch.save(engine.model.state_dict(), f"{save}_exp1_best_{round(his_loss[bestid], 2)}.pth")
print("Total time spent: done", flush=True)
