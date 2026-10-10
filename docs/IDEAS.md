# Ideas, tuning and troubleshooting log

Companion to [PLAN.md](PLAN.md). The MVP probably won't work well on the first try. The detector
may fire too often, the update may overfit nine steps of incident data, or a head shared by all
sensors may not be able to express a local blockage. This file lists what to try, roughly
cheapest first, and keeps a record of what we tried. Update it whenever something fails or works.

None of these ideas start before the MVP pipeline (PLAN weeks 1 to 5) produces a full results CSV.

---

## 1. Likely failure modes and first fixes

| Symptom | Likely cause | What to try |
|---|---|---|
| Too many false alarms | The network-wide error is noisy (rush hour, missing data) | Normalise the error by the expected error for that time of day (residual divided by the validation std per time slot); raise ADWIN delta or the PH threshold; require two consecutive alarms |
| Alarm comes too late | Five affected sensors out of 207 hardly move the network mean | Use the max or the top-k mean over sensors instead of the mean; per-sensor detection (stretch); error pooled over graph neighbourhoods |
| Normal error gets worse after adapting | The head update overfits or spreads to every sensor | Larger beta, fewer steps, smaller lr; the reset policy; update only `end_conv_2`; restrict the loss to affected sensors |
| Disrupted error barely improves | Nine steps is very little data, and the 1x1 conv head is shared by all nodes, so it can't make a per-node correction | Per-node bias/scale adapter (section 3); more steps; higher lr with a stronger anchor |
| Error drops, but the correction outlives the incident | The head was never reset | Reset policy, or exponential decay back towards theta_0 |
| Adaptation takes longer than 5 minutes | Not running on the GPU, or too many windows | The head has about 138k parameters, so this should take milliseconds; profile it |

## 2. Hyperparameters and how to tune them

Tune only on the validation period with injected incidents, using different seeds from test. We
touch test once per final configuration. Every tuning run gets logged in `results/tuning/*.csv`.

| Group | Parameter | Start | Range |
|---|---|---|---|
| Detector ADWIN | delta | 0.002 | 1e-4 to 0.1 (log) |
| Detector PH | threshold (lambda) | about 1 false alarm per day | tuned |
| Detector PH | delta, alpha | 0.005, 0.9999 | tuned |
| Signal | aggregation | mean | mean, top-5 mean, max, time-of-day normalised |
| Update | steps K | 5 | 1 to 50 |
| Update | lr | 1e-3 | 1e-4 to 1e-2 (log) |
| Update | beta (L2 anchor) | 1.0 | 0, or 1e-3 to 1e2 (log) |
| Update | Huber delta | 1.0 (normalised units) | 0.5 to 5 |
| Update | window count N | 6 | 3 to 24 |
| Update | grad clip | 5 | 1 to 10 |
| Update | optimiser | Adam, fresh state per alarm | Adam or SGD |
| Reset | band margin, M steps | 1 std, 6 | tuned |

K, lr and beta interact, so start with a small grid over those three, then switch to random search
(about 30 runs) or Optuna if the grid turns out too coarse. The objective is disrupted-period MAE
at 15 min, with a hard constraint that normal-period MAE stays at or below the static MAE. Configs
that break the constraint are rejected. A replay takes minutes, so about 100 configs fit into one
night on the laptop or one Colab session.

Baseline training hyperparameters stay at the paper's values, since changing them breaks the
comparison with the paper.

## 3. Better update rules (after the MVP)

1. Head ablations: `end_conv_2` only versus both layers, and bias-only (very cheap and very stable).
2. Per-node adapter: a zero-initialised scale and bias per sensor on the output,
   `y = y_hat * (1 + a_v) + b_v`, and train only those. The update stays local to the blocked
   sensors while the shared head stays frozen. If the shared 1x1 head turns out unable to
   localise, this is the first thing to try.
3. Masked local loss: compute the adaptation loss only on sensors whose error is out of band.
4. Partial targets: use windows whose targets are only partly known, with a horizon mask, so the
   incident itself enters the training set sooner.
5. An EWC (Fisher-weighted) anchor instead of plain L2. The Fisher can come from the train set and
   is cheap to compute for the head.
6. Experience replay: mix a few normal windows from a small buffer into each update batch.
7. Residual correction model: keep Graph WaveNet frozen and fit a tiny online model (ridge, RLS
   or a small MLP) on the residuals of recent steps. This is often hard to beat and trivial to reset.
8. Meta-learned initialisation or learning rate (MAML-style, trained on simulated incidents). This
   is closer to a research project, so only if time is left.

## 4. Better detection

- Normalise residuals by time of day, using the mean and std of validation residuals per 5-minute slot.
- Per-sensor detectors (a stretch goal in the spec). 207 ADWIN instances are cheap; alarms can be
  aggregated over graph neighbours.
- Compare against simple detectors: a fixed threshold on the z-scored residual, and CUSUM. To
  justify the detector choice, sweep the sensitivity and plot detection delay against false alarms.

## 5. Other forecasting models

The spec fixes Graph WaveNet as the backbone, so these are for context or a second backbone only.

| Model | Why it's interesting | Effort |
|---|---|---|
| Graph WaveNet fork by Shleifer et al. (2019) | Same code family, better reported numbers | low |
| DCRNN | The original METR-LA baseline. Being an RNN, it may adapt differently. | medium |
| STGCN / ASTGCN | Lighter. The ASTGCN repo already has PeMS04. | medium |
| MTGNN, AGCRN | Learned graphs, strong benchmark results | medium |
| STAEformer, PDFormer, D2STGNN | Recent top models on METR-LA | medium to high |
| Historical average, last value, seasonal naive | They belong in the report for context and as a sanity check on incident periods, where last value can beat deep models during a blockage. We should do these. | very low |

BasicTS and LibCity bundle many of these models and could provide a second backbone. The baseline
itself has to use the authors' code, as the spec requires.

## 6. Stretch goals from the spec (not in the MVP)

- Error-aware dynamic graph re-weighting: weaken message passing between sensors whose residuals
  disagree, by scaling adaptive adjacency entries with residual similarity. This needs access to
  the inside of `gwnet.forward`, so we'd subclass it rather than edit upstream code.
- Sensor noise and sensor failure scenarios. A failure produces zeros, which must be masked and
  must not trigger alarms.
- Per-sensor detection.
- PEMS-BAY as a second dataset. The same scripts work, and `pems-bay.h5` is in the same DCRNN
  Drive folder (file id `1wD-mHlqAb2mtHOe_68fZvDh1LpDegMMq`).
- The PeMS04 real incident (sensor 104, 25 Jan 2018) as a qualitative figure.

## 7. Compute notes

- The laptop's RTX 4050 (6 GB) is enough for baseline training at batch 64 and for all replay
  experiments. If it runs out of memory, drop to batch 32 (and record that as a deviation) or use
  `torch.autocast` with bf16.
- A free Colab T4 is about as fast as the laptop. It's useful for running seeds in parallel. Keep
  data and checkpoints on Drive and save every epoch.
- Windows, WSL and VMs: PyTorch CUDA works natively, so we don't need any of them. GPU passthrough
  to a VM is hard on consumer Windows and costs speed. WSL2 is only a fallback if some library has
  no Windows wheel.

## 8. Log

Append, newest last. Format: date, what we tried, outcome, next step.

- 2026-10-07: project setup. Details in the PLAN decision log. Next: smoke run, then full baseline training.
- 2026-10-07: gdown (Python requests) could not reach Google Drive here (SSL INVALID_SESSION_ID),
  while curl with `--ssl-no-revoke` worked. If a download fails, try curl before anything else.
- 2026-10-07: official train.py failed on PyTorch 2.11 (Conv1d given 4D input). Fixed from our side in
  `src/gwn.py` without touching upstream. Smoke run: 66.7 s/epoch. Full baseline training started.
- 2026-10-07: the first full training run died silently after epoch 7. A process started from the
  session (even with Start-Process) gets killed when the session closes. Long jobs now start
  through WMI (`Invoke-CimMethod Win32_Process Create`), which detaches them. Training on the
  laptop also stops if the machine sleeps, so keep it plugged in with the lid open.
- 2026-10-07: river ADWIN with its default clock=32 detected a 9-step error jump 23 steps late in
  the synthetic self-check. With clock=1 it took 5 steps. PageHinkley (threshold 5) took 1 step
  but raised 2 false alarms in 2000 steps of noise. Both need calibration on real residuals.
- 2026-10-07: replay, incidents, drift, adapt and the experiment runner are written and pass
  their self-checks. Waiting for the trained checkpoint before any real experiment.
- 2026-10-07: the second training run died at epoch 9 with "forrtl: error (200): program aborting
  due to window-CLOSE event". The Intel Fortran runtime inside numpy/scipy installs a console
  handler and aborts when the hidden console gets a close event. Training now runs under
  pythonw.exe (no console), `scripts/train_baseline.py --log <file>` writes its own log, and
  FOR_DISABLE_CONSOLE_CTRL_HANDLER=1 is set before numpy loads. Launch command:
  `Invoke-CimMethod Win32_Process Create` with
  `pythonw.exe -u scripts\train_baseline.py --log results\baseline_train.log`.
- 2026-10-07: pipeline smoke test with the 1-epoch checkpoint (results/smoke/, not results).
  Full run of 4 variants x 1 seed took 18 min while training shared the GPU. What it showed
  about the method, independent of model quality: the network-wide mean t+1 error with ADWIN
  (delta 0.002) caught only 6 of 24 synthetic incidents and gave about 2.3 false alarms per normal
  day, so the mean over 207 sensors dilutes a 5-sensor blockage too much. Next things to try,
  in order: top-k mean aggregation, time-of-day normalised residuals, PageHinkley with a
  calibrated threshold. Head updates took about 0.2 s on average (max 3.6 s, the first call
  includes CUDA warm-up). Naive full fine-tuning roughly doubled normal-period error, which is
  the forgetting the spec expects.
- 2026-10-07: Win+L only locks the screen; processes keep running. The idle sleep timer still runs
  while locked, though: on this laptop it is 2 h on AC and 30 min on battery. train_resumable.py and
  run_experiments.py now call SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED), which
  blocks idle sleep while they run without touching power settings. Closing the lid or choosing
  Sleep still suspends the machine.
- 2026-10-08: detector calibration on validation (details in PLAN week 3). Best result: 43 % of
  synthetic incidents detected at 0.92 false alarms per day. Diagnosis on seed 100: the t+1 error
  on the blocked sensors is high for only about 3 steps at onset (and 1 to 2 steps when the
  blockage lifts). After that the model sees the low speeds in its input and its 5-minute
  forecast follows them, so the t+1 error falls back to about 2 mph. Meanwhile the normal
  period has heavy tails (99.9th percentile of the top-3 z signal is 35, and incident peaks are
  about 20 to 37). Some "false alarms" are probably real, unlabelled slowdowns in the data.
  Consequences and ideas:
  1. A t+1-only signal caps detection quality. Ablation worth running: aggregate the errors of
     every forecast whose target is `now` (horizons 1 to 12, issued at now-1 .. now-12). All of
     them are released at the same moment, so this is still leak free, and forecasts issued
     before the onset keep missing for up to 12 steps instead of 3. This departs from the
     spec's t+1 rule, so it needs a team decision before it becomes the default.
  2. Evaluate false alarms against the mined real slowdowns too: an alarm on a real event
     shouldn't count as false.
  3. Because the t+1 forecast recovers by itself, the cost of an incident for the static model
     is probably at the 30 and 60 minute horizons. The experiments should show where the error
     actually is before we tune the update rule.
- 2026-10-08: first full experiment. Test split, 3 seeds (0, 1, 2), about 22 synthetic incidents
  per seed (5 adjacent sensors at 0.15x for 9 steps, 2-step ramps), detector from
  results/detector.json, ours = head only with beta 1, lr 1e-3, 5 steps, 6 windows.
  MAE in mph, mean over seeds (std in brackets):

  | Period, horizon | static | naive | ours | reset |
  |---|---|---|---|---|
  | normal 15 min | 2.663 (0.011) | 4.516 (0.414) | 2.692 (0.011) | 2.675 (0.013) |
  | normal 60 min | 3.438 (0.027) | 6.602 (0.263) | 3.439 (0.027) | 3.441 (0.028) |
  | disrupted 15 min | 13.248 (0.305) | 14.731 (0.902) | 13.274 (0.311) | 13.517 (0.329) |
  | disrupted 60 min | 25.040 (0.993) | 30.352 (1.167) | 25.029 (0.994) | 25.030 (1.045) |

  Detection on test: about 11 of 22 incidents per seed, mean delay about 7 steps, about 1.2 false
  alarms per normal day. Head updates took 0.16 s on average (max 0.28 s).

  What this says:
  - Naive full fine-tuning forgets, as expected: normal error rises by about 70 % at 15 min.
  - Ours and reset are within noise of static on disrupted periods, and ours is 0.03 mph worse
    on normal 15 min, which breaks the "no degradation" constraint (small, but real).
  - Why ours can't help with this setup: the alarm comes about 7 steps after onset, which is
    close to the end of a 9-step blockage. The update then teaches the head "speeds are low" just
    as they recover. The head is a 1x1 convolution shared by all 207 sensors, so it can't make a
    local correction for 5 sensors without nudging every other sensor too. Disrupted error is
    dominated by 30 and 60 minute forecasts that cross the onset or the release, and no update
    made after the alarm can fix forecasts issued before it.
  - Figure results/fig_incident_seed0_3.png shows this for one incident.

  Decisions to make next (team), roughly in order of expected value:
  1. Per-node output adapter (section 3, item 2) instead of the shared head. Keeps the update
     local to the sensors that are wrong, which should fix the normal-period degradation.
  2. Earlier, stronger detection: the multi-horizon signal from the previous log entry (needs
     a decision, it changes the spec's t+1 rule), or per-sensor detection (spec stretch goal).
  3. Longer incidents as an extra scenario (for example 2 h). A 45-minute blockage is barely
     longer than detection delay plus one update, so there is little time left to benefit.
  4. Tune beta, lr and K on validation with the normal-period guard. beta = 1 is too weak to keep
     the normal error flat.
- 2026-10-10: tuning and the test run are done; the log moved to docs/EXPERIMENTS.md, which has
  every experiment from E0 on. Ideas still open from this file: mined real incidents as a test
  scenario (mining thresholds need tightening), the multi-horizon detection signal, per-sensor
  detectors, EWC or replay-based anchors, and a residual-correction model as an alternative
  update rule.
