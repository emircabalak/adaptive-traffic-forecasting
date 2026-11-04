# Implementation plan

Adaptive traffic speed forecasting under sudden disruptions. RTU group course project, six weeks.

This file is the working plan. Tick boxes as tasks finish and add to the decision log at the
bottom; don't delete old entries. Ideas, tuning notes and stretch work live in [IDEAS.md](IDEAS.md).

We have no results yet. Every number in this repo has to come from a run whose log or CSV we can
point to. An earlier draft of the spec contained made-up numbers, and none of them may come back.

## Where we left off (updated after each work day)

2026-10-29. Two methods evaluated on test: the learned node adapter (best on 45-minute
incidents) and the severe-slowdown rule (best on 2 h incidents). Details at the end of
`docs/EXPERIMENTS.md`, team summary in `docs/TEAM_UPDATE.md`, figures in `results/`.
Open: a hybrid that passes the normal-period constraint, mined real incidents as a scenario
(the best check of whether the rule's thresholds transfer), dashboard.

To continue after a break or a shutdown:
1. Tuning queue: `powershell -ExecutionPolicy Bypass -File scripts\start_tuning.ps1`. It runs
   the configs listed in `results/tune_queue.jsonl` that aren't in `results/tuning.csv` yet,
   resuming per seed. Progress: `results/tune.log`. Safe to run twice.
2. Read the newest outcomes in EXPERIMENTS.md and `results/tuning.csv`, write the next entry
   (hypothesis first), add its configs to the queue, start the queue again.
3. Training (done; only if it ever needs rerunning): `scripts\start_training.ps1`, resumes per epoch.
4. Final step once the method is settled: test-set run with `scripts/run_experiments.py` (add the
   new variant there), then plots and the summary for the team.

---

## 0. Environment

| Item | Choice | Reason |
|---|---|---|
| OS | Native Windows 11 | PyTorch publishes CUDA wheels for Windows, and `torch.cuda.is_available()` returns True on the RTX 4050 Laptop GPU (6 GB). TensorFlow's lack of Windows CUDA doesn't affect us because Graph WaveNet is written in PyTorch. We don't need a VM or WSL. |
| Python | 3.14, venv at `C:\Users\emirc\venvs\adaptive` | The venv sits outside OneDrive so the 1 to 2 GB of packages don't get synced. |
| GPU | Local RTX 4050 for everything. Colab only for multi-seed runs or sweeps. | See section 6. |
| Data | `data/` (gitignored) | `metr-la.h5` from the DCRNN Google Drive folder, sensor graph files from the DCRNN GitHub repo. |
| Model code | `third_party/Graph-WaveNet`, a git submodule at upstream commit `6b162e8` | The spec says to use the authors' code. We import it and never edit it. |

Setup for a teammate:
```
git clone --recurse-submodules <repo>
python -m venv <a folder outside OneDrive>
<venv>/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
<venv>/Scripts/python -m pip install -r requirements.txt
```
Then put `metr-la.h5` in `data/` and run the commands in section 4, week 1.

## 1. Reference numbers

Checked against the paper PDF (Wu et al., IJCAI 2019, arXiv 1906.00121, Table 2), METR-LA,
Graph WaveNet. The numbers in the project brief were from memory and turned out to be slightly off.

| Horizon | MAE | RMSE | MAPE |
|---|---|---|---|
| 15 min (h=3) | 2.69 | 5.15 | 6.90 % |
| 30 min (h=6) | 3.07 | 6.22 | 8.37 % |
| 60 min (h=12) | 3.53 | 7.37 | 10.01 % |

Table 4 of the paper reports 53.68 s per training epoch on METR-LA (on their GPU). The paper
configuration matches the repo README command: `--gcn_bool --adjtype doubletransition --addaptadj
--randomadj`, 100 epochs, batch 64, lr 1e-3, dropout 0.3, weight decay 1e-4, nhid 32.

Our baseline passes if each MAE is within about 0.1 mph of the table. Independent reproductions
usually land around 2.69 to 2.75, 3.07 to 3.15 and 3.5 to 3.6, so anything in that band is fine.
A bigger gap means something in our setup differs and we should find out what.

## 2. What the official code does (and why it matters here)

- The repo doesn't include `adj_mx.pkl`. We took it from DCRNN's `data/sensor_graph/`, along with
  `distances_la_2012.csv`, `graph_sensor_ids.txt` and DCRNN's `gen_adj_mx.py`. Regenerating the
  thresholded Gaussian adjacency from the distances reproduces the shipped pickle (max abs diff
  1.2e-7, 1722 nonzero entries both ways).
- The DCRNN `metr-la.h5` was written by pandas 0.15, and pandas 3 can't read it. `src/data.py`
  reads it through pytables and can write `data/metr-la-pd3.h5`, which the official
  `generate_training_data.py` then reads without changes.
- `generate_training_data.py` builds x from speeds at t-11..t plus a time-of-day channel, and y
  from t+1..t+12. It splits 70/10/20 using `round()`. With 34272 timesteps that gives 34249
  windows: 23974 train, 3425 val, 6850 test. Test window i has its "now" at raw index
  `27410 + i`. `src/data.py` checks that the raw series reproduces the npz windows exactly, and the
  replay engine depends on this.
- `util.load_dataset` z-scores inputs with the train mean and std, but those statistics include
  the 0.0 missing values (mean 54.41, std 19.49, versus 58.58 and 12.82 with zeros masked). Missing
  inputs reach the model as normalised zeros. Labels stay in raw mph, and every loss and metric
  masks `null_val=0.0`. We keep this as is for the baseline so our numbers are comparable with the
  paper. All of our own metrics mask zeros. A variant with clean statistics is listed in IDEAS.
- The official MAPE has no denominator floor. Ours in `src/metrics.py` uses the mask plus a floor.
- `--device` defaults to `cuda:3`, so pass `--device cuda:0`. The directory in `--save` has to exist.
  `scripts/train_baseline.py` sets both, along with the paper flags.
- Upstream builds the gate, residual and skip layers as `nn.Conv1d` with (1, k) kernels and feeds
  them 4D tensors. PyTorch 1.x accepted that and ran a 2D convolution; current PyTorch raises an
  error. `src/gwn.py` swaps each of these layers for an `nn.Conv2d` that shares the same weight
  tensor, with dilation (1, d). The computation matches what PyTorch 1.x did, parameter names
  don't change, and the upstream files stay untouched. `python src/gwn.py` checks the equivalence.
- The model contains BatchNorm layers. During streaming and adaptation the whole model stays in
  `eval()` mode, so BN running statistics and dropout never change. Only head parameters get
  gradients.
- The output head is `end_conv_1` (1x1 conv, 256 to 512 channels) plus `end_conv_2` (1x1 conv,
  512 to 12), about 138k parameters. We decided "head" means both layers. Updating only
  `end_conv_2` is an ablation in IDEAS.

## 3. Repository layout (target)

```
third_party/Graph-WaveNet/   official code, submodule, never edited
data/                         raw and processed data (gitignored)
checkpoints/                  frozen baseline (gitignored, shared through Drive)
src/
  data.py          h5 loading, masking, official scaler, adjacency, test series reconstruction
  gwn.py           builds gwnet from the official code, loads the checkpoint, head parameter helpers
  metrics.py       masked MAE/RMSE/MAPE (MAPE with floor), normal vs disrupted split
  replay.py        streaming loop with the pending-label buffer
  drift.py         river ADWIN / PageHinkley on the t+1 error
  adapt.py         head-only Huber + L2 anchor update, reset-to-initial policy, naive full fine-tuning
  incidents.py     synthetic blockage injection and mining of real slowdowns
scripts/
  run_experiments.py         every variant and scenario, writes results/*.csv
  plots.py
app/dashboard.py   Streamlit (lowest priority)
results/            CSVs and figures (committed)
docs/               PLAN.md, IDEAS.md
```
One module per concern. We'll add class hierarchies only if a second implementation of something
actually shows up.

## 4. Stages and tasks

Everything below belongs to the data pipeline, baseline and online adaptation owner unless a task
says otherwise.

### Week 1: data and baseline reproduction (current)
- [x] Init the repo, add Graph WaveNet as a submodule, create the venv, confirm CUDA works
- [x] Check the paper numbers in the PDF (section 1)
- [x] Download the sensor graph files from DCRNN
- [x] Download `metr-la.h5`. Shape (34272, 207), 2012-03-01 00:00 to 2012-06-27 23:55, 5-minute
      steps with no gaps, 8.11 % of readings are 0.0 (missing), 2148 timesteps where all sensors are missing
- [x] Run the official `generate_training_data.py`, producing `data/METR-LA/{train,val,test}.npz`
- [x] Regenerate the adjacency with `gen_adj_mx.py` and confirm it matches `adj_mx.pkl`
- [x] `src/data.py` with a self-check (`python src/data.py`): sensor order, 5-minute index, split
      sizes, test series reconstruction
- [x] One-epoch smoke run through `scripts/train_baseline.py`: 66.7 s per training epoch, 4.1 s
      validation, about 1.8 GB VRAM, GPU at 86 C. The pipeline runs end to end.
- [x] Full 100-epoch training. Best validation loss 2.7359 at epoch 93, saved as
      `checkpoints/metr_exp1_best_2.74.pth`. Log: `results/baseline_train.log`. Epochs 1 to 35
      ran in upstream train.py, 36 onwards in train_resumable.py (see decision log).
- [x] Evaluate with our masked metrics (`scripts/eval_baseline.py`, `results/baseline_eval.csv`):

      | Horizon | MAE (ours / paper) | RMSE (ours / paper) | MAPE % (ours, 5 mph floor / paper) |
      |---|---|---|---|
      | 15 min | 2.703 / 2.69 | 5.164 / 5.15 | 6.92 / 6.90 |
      | 30 min | 3.095 / 3.07 | 6.239 / 6.22 | 8.49 / 8.37 |
      | 60 min | 3.533 / 3.53 | 7.355 / 7.37 | 10.14 / 10.01 |

      Every MAE is within 0.03 mph of the paper, so the baseline passes. Streaming replay MAE
      equals the official batch test MAE (2.7028 at 15 min), which confirms the replay engine
      on the trained model as well.

Commands:
```
python src/data.py                                   # check raw data, adjacency, splits
python -c "import sys; sys.path.insert(0,'src'); import data; data.convert_for_pandas3()"
python third_party/Graph-WaveNet/generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la-pd3.h5
python src/gwn.py                                    # check the Conv1d patch
python scripts/train_baseline.py                     # official train.py with paper flags; extra flags override
```

### Week 2: streaming replay (stage 3)
- [x] `replay.py` walks the test series one step at a time. At step t it predicts t+1..t+12 from t-11..t.
- [x] Pending buffer: a deque of `(issue_t, horizon_k, target_t, pred[207])`. A label is released
      only once `now >= target_t`, and the code asserts that nothing scores or adapts on a label
      with `target_t > now`.
- [x] Unit check: static replay MAE per horizon equals the batch test MAE (same windows, same model).
- [x] Leak test: change y at a future time and confirm nothing computed before that time changes.
- [x] Measure inference latency per step. `python src/replay.py` runs the equivalence and leak
      tests on a randomly initialised model (40 test windows), so they need no checkpoint.

### Week 3: drift detection (stage 4) and incidents
- [x] `incidents.py` synthetic blockage: k=5 graph-adjacent sensors picked by BFS on the
      adjacency, speed scaled to 0.15x nominal for 9 steps, optional linear ramp. The injection
      modifies the raw series before replay, so it shows up in later inputs and in targets.
      Missing zeros stay zero. Each injection is seeded and logged (onset, end, sensors).
- [x] Incident schedule: a fixed list of (time, sensors) per seed, shared by all variants.
- [~] Real slowdown mining on the METR-LA test period. For each sensor and time of day, compare
      against the median of the same weekday in other weeks, and keep large drops that don't repeat
      in other weeks. Output a CSV of candidate events and plot a few by hand, the way the PeMS04
      example was checked.
      Status: `mine_slowdowns` works but is too permissive (3255 candidates in the test period).
      Tighten min_steps / min_drop and merge events on neighbouring sensors. A clear real event:
      2012-06-18 14:15, sensors 717462, 717466, 717468, 717472 drop from about 62 to about 5 mph.
- [x] `drift.py`: the network-wide signal is the masked mean |error| of the t+1 predictions across
      sensors, fed to river's ADWIN and PageHinkley. The detector never sees MAPE.
- [x] Detection delay (alarm step minus onset step) and false alarms per normal day (alarms outside
      every incident window plus margin, per 288 steps).
- [x] Calibrate detector parameters on the validation period only, then freeze them for test.
      `scripts/calibrate_detector.py`: 3 validation seeds (100 to 102, 35 incidents), grid over
      signal (raw or normalised; mean, max, top3/5/10/20) and detector (ADWIN delta, PageHinkley
      threshold and delta). Rule: highest detection rate with at most 1 false alarm per normal
      day. Chosen: PageHinkley (threshold 50, delta 0.2) on the top-3 mean of per-sensor
      z-scored t+1 errors (z from training-period error stats per sensor and 5-minute slot,
      `results/residual_stats.npz`). Validation: 15 of 35 incidents detected (43 %), mean delay
      6.9 steps, 0.92 false alarms per day. Saved in `results/detector.json`.

### Week 4: online adaptation (stage 5)
- [x] `adapt.py`, our method. On an alarm, take the most recent N windows whose 12-step targets
      are all known (or partially known, with a horizon mask) and run K Adam or SGD steps on the
      head only. Loss = Huber(pred, y, masked) + beta * ||theta_head - theta_head_0||^2, with
      gradient clipping.
- [x] Reset policy: put theta_head_0 back once the t+1 error has stayed inside its normal band for M steps.
- [x] Naive variant (b): update all weights every step, without the anchor or the detector.
- [ ] Time each adaptation. It has to be far below 300 s; on the GPU we expect under a second.
- [ ] Guard: the normal-period error of (c) must not go above static (a). If it does, tune beta, lr and K.

### Week 5: experiments and evaluation
- [x] `run_experiments.py` runs variants (a), (b), (c) on every scenario (synthetic seeds, mined
      real events, no incident) over identical replays and writes `results/*.csv`.
- [x] MAE, RMSE and MAPE at 15, 30 and 60 min, reported separately for normal and disrupted
      periods (disrupted means the incident window on affected sensors plus a recovery tail). Also
      time to recovery, detection delay and false alarms per day.
- [x] At least 3 seeds, reported as mean and std.
- [x] `plots.py`: forecast against actual around incidents, error over time with alarm markers, bar charts.
- [ ] Mined real events as a scenario (needs the mining thresholds tightened first).
- [ ] Make the update rule actually help (IDEAS log 2026-10-08), then rerun.

### Week 6: dashboard and report
- [ ] Streamlit app: choose a scenario, scrub the replay, show forecast against actual for one sensor with alarm markers.
- [ ] Every number in the report comes from a CSV in `results/`.

## 5. Definitions

Fixed here so everyone on the team computes the same thing.

- Masked error: positions where the ground truth is 0.0 are excluded everywhere.
- MAPE: `|y_hat - y| / max(y, 5 mph)` over masked positions. The 5 mph floor is configurable and
  the report states it.
- Normal period: steps outside [onset, end + recovery_tail] of every incident, for every sensor.
  Per-sensor disrupted metrics use the affected sensors inside the window.
- Detection delay: steps from onset to the first alarm inside [onset, end + tail]. No alarm in that
  range counts as a miss.
- False alarm: an alarm outside all incident windows, reported per 288 steps (one day).
- Time to recovery: steps after the incident ends until the rolling t+1 error on affected sensors
  stays within `normal_mean + margin` for M steps. The margin is fixed in advance, for example one
  std of the normal error.
- Normal error band (used by the reset policy and the recovery metric): mean and std of the t+1
  network error over the validation period.

## 6. Compute budget

- Measured on the RTX 4050 Laptop: 66.7 s per training epoch plus 4.1 s validation, so 100 epochs
  take about 2 h (the paper reports 53.7 s/epoch on its GPU). The GPU reached 86 C, so keep the
  laptop plugged in and well ventilated during training.
- A replay is 6850 test steps with one forward pass each (a few ms), so seconds to a few minutes
  per variant. Naive full fine-tuning is the slowest variant.
- Colab helps for running seeds in parallel, for hyperparameter sweeps, or if the laptop
  throttles. The MVP doesn't need it. Colab sessions get killed, so checkpoint every epoch (the
  official code already does).

## 7. Decision log

Append only, newest last.

- 2026-10-07: Native Windows with PyTorch CUDA. No VM or WSL. The venv lives outside OneDrive.
- 2026-10-07: Paper numbers checked against arXiv PDF Table 2 (section 1).
- 2026-10-07: The baseline uses official preprocessing unchanged (statistics include zeros) so it
  stays comparable with the paper. All our metrics and losses mask zeros.
- 2026-10-07: Head means end_conv_1 plus end_conv_2. The model stays in eval() during replay and adaptation.
- 2026-10-07: The drift signal is the network-wide masked mean |error| at horizon t+1.
- 2026-10-07: The raw h5 is read through pytables because pandas 3 can't parse the pandas 0.15
  format. Downloading from Google Drive with Python failed with an SSL error on this machine, while
  curl worked (file id `1pAGRfzMx6K9WWsfDcD1NMbIif0T0saFC`).
- 2026-10-07: `src/gwn.py` replaces upstream Conv1d layers with equivalent Conv2d at construction
  time, so the official code runs on PyTorch 2.11 without edits.
- 2026-10-07: river's ADWIN checks for change only every `clock` samples, default 32 (2.7 h at
  5-minute steps). `drift.make_detector` sets clock=1. PageHinkley runs in mode "up" (we only care
  about error increases).
- 2026-10-07: Adaptation windows are the N most recent issue times (now-N .. now-1) with their
  targets masked where the target time is after now. Targets come only from the series up to now,
  so future data can't get in even by mistake. This lets the incident itself enter the loss
  sooner than "fully known windows only", which would all predate a short incident.
- 2026-10-07: Variants in `run_experiments.py`: static (a), naive (b, all weights, Adam state kept
  across steps, every step), ours (c, head only, beta=1, gated by ADWIN), reset (head only, beta=0,
  head restored once the t+1 error is back in the validation band for 6 steps). Detector and
  update hyperparameters are provisional until calibrated on the validation period.
- 2026-10-07: `scripts/train_resumable.py` runs the same loop as upstream train.py (upstream
  engine, util and model) and saves model, optimizer, epoch, validation history and RNG states
  after every epoch. It can also continue from checkpoints left by a train.py run; in that case
  Adam restarts with fresh moments. Tested: bootstrap from a train.py checkpoint, then resume from
  its own state file. `scripts/start_training.ps1` starts or continues training safely.
- 2026-10-08: Baseline accepted. Test MAE 2.703 / 3.095 / 3.533 at 15 / 30 / 60 min against the
  paper's 2.69 / 3.07 / 3.53. This checkpoint is the frozen baseline for every experiment.
- 2026-10-08: The detector signal is the top-3 mean of per-sensor, per-time-of-day z-scored t+1
  errors (still the masked t+1 absolute error, only normalised). The raw network mean caught
  at most 14 % of validation incidents at <= 1 false alarm per day.
- 2026-10-10: Final method: per-sensor output adapter (src/adapt.py NodeAdapt) with max_horizon 6,
  decay 0.7, 3 training windows, 6 continuous update steps after an alarm, nbrtop3 PageHinkley
  detector at threshold 8. Chosen on validation (EXPERIMENTS E0 to E15), confirmed on test:
  disrupted gain +0.185 mph (45 min) and +0.267 (2 h), normal-period delta +0.0036.
- 2026-10-29: Added the severe-slowdown rule (Persist in src/adapt.py, config "rule-severe") as a
  rule-based alternative to the learned adapter. Test: +0.113 (45 min) and +0.850 (2 h), normal
  delta +0.0036. Reported next to the adapter, not instead of it.
- 2026-10-29: E23/E23b: on real slowdowns (clean data, below 0.3 of the usual speed, any duration)
  Graph WaveNet beats persistence on both validation and test. Persistence-style gains on
  synthetic incidents don't transfer; the real-event scenario becomes the priority.
