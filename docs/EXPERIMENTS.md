# Experiment log

Every change to the online method is tried here before it is kept. Each entry is written before
the run (hypothesis, change, what counts as success), and the outcome is filled in after.
Changes that improve the objective are kept; the rest are reverted. Later entries build on the
best configuration so far.

## Protocol

- Tuning happens on the validation period only, with synthetic incidents from seeds 100, 101 and
  102 (35 incidents in total). The test split is used once per finalist, through
  `scripts/run_experiments.py`.
- Harness: `python scripts/tune.py '<json config>'`. It replays the validation period for each
  seed, compares against the static model on the same series, and appends a row to
  `results/tuning.csv`.
- Objective: lower disrupted-period MAE, averaged over 15, 30 and 60 min (`dis_avg`).
- Hard constraint: normal-period MAE at 15 min must not exceed static by more than 0.005 mph
  (`norm15_delta <= 0.005`). That margin is about half the seed-to-seed std of the static model.
- Secondary: false alarms per day, mean update time (must stay far below 300 s).

## Baseline for comparison (validation, before any change)

Filled in by E0.

## Entries

### E0: harness and reference numbers
Hypothesis: none, this sets the reference. Run static and the current "ours" (head only, beta 1,
lr 1e-3, 5 steps, 6 windows, calibrated detector) on validation.
Outcome: pending.

### E1: where does the disrupted error come from?
Hypothesis: most of the static model's disrupted error comes from forecasts issued before the
onset or shortly before the release, which no update after the alarm can fix. If so, the
achievable gain is bounded, and we should measure against that bound, not against zero.
Change: diagnostic only (`python scripts/tune.py --diag`, `results/diag_error_split.csv`).
Outcome (static model, 35 validation incidents, incident sensors inside each incident's own
window):

| Horizon | issued before onset | issued during incident | issued after release |
|---|---|---|---|
| 15 min | 29 % of error (MAE 32.5) | 63 % (MAE 16.4) | 8 % (MAE 2.9) |
| 30 min | 45 % (MAE 40.6) | 51 % (MAE 21.0) | 4 % (MAE 3.6) |
| 60 min | 82 % (MAE 43.3) | 18 % (MAE 9.0) | none in window |

Signed error (forecast minus actual) on incident sensors, by issue time relative to onset: while
the blockage holds (offsets 2 to 7), the 15-minute forecast is 6 to 10 mph too high and the
30-minute forecast about 24 mph too high, so the model expects recovery too soon. From offset 8
the release starts to fall inside the horizon and the sign flips (down to -36 mph at 15 min).
Conclusions:
- At 60 min at most 18 % of the error is reachable by any update, at 15 min up to 63 %.
- A "slowdown persists" correction helps early in an incident and hurts near its end. For a
  9-step blockage the useful window is roughly offsets 2 to 7, and the current detector fires at
  about offset 7, so it can't gain anything. Detection has to come earlier, and the correction
  has to be local and short lived.
- Long incidents behave differently: the early-recovery bias lasts the whole incident. Real
  incidents can be long (the mined 2012-06-18 event lasted over 5 hours). Tuning therefore also
  reports a long-incident scenario (2 h hold) next to the spec's 45-minute one.

### E2: per-node output adapter instead of the shared head
Hypothesis: the shared 1x1 head can't correct 5 sensors without moving the other 202, which is
why "ours" degrades the normal period. E1 adds that the useful correction is "the slowdown
persists", which a plain bias learns slowly. So the adapter is, per sensor v and in normalised
units, `y' = y + alpha_v * (last_v - y) + b_v`, where last_v is the most recent non-missing input
speed. alpha pulls the forecast towards persistence, b is a bias. Both start at zero (identity),
are trained on alarm with masked Huber + beta * (alpha^2 + b^2), and all Graph WaveNet weights
stay frozen, so the base forecasts are computed once per update and the steps are cheap.
Variants: E2a no decay; E2b decay 0.9 per step after the update, so the correction fades within
about 30 minutes unless another alarm renews it.
Success: norm15_delta <= 0.005 and dis_avg below static, on the 9-step scenario; the 24-step
scenario reported alongside.
Outcome: pending.

### E3 and later
Planned in order: update hyperparameters (lr, steps, windows, beta, decay back to zero),
restricting the loss to sensors that are out of band, stronger detection signals (per-sensor
detection, the multi-horizon signal), and a longer incident scenario for evaluation.
Each gets its own entry before it runs.
### E2 outcome (filled in after the run)
- E2a (no decay): dis_avg 20.351 vs static 20.125 (gain -0.226), norm15 delta +0.096. Rejected.
  Updating every sensor with no decay leaks into the normal period, worse than the shared head.
- E2b (decay 0.9): dis_avg 20.213 (gain -0.089), norm15 delta +0.0016 (passes). Decay fixes
  the normal period. Disrupted error is still slightly worse than static, which fits E1: the
  alarm fires around offset 7, and the correction then carries into the release.
- Update time dropped from about 220 ms (head) to about 110 ms, because the backbone runs once.
Kept: decay. Not yet better than static.

### E3: oracle alarm timing (diagnostic, not a method)
Hypothesis: with an alarm at onset + 2 (the earliest a t+1 based detector can plausibly fire),
the E2b adapter beats static on disrupted error. If it doesn't even then, the adapter is the
problem; if it does, detection timing is the bottleneck.
Change: `"oracle_offset": k` in tune.py replaces the detector with alarms at onset + k for every
incident and none elsewhere. Run k = 1, 2, 4, 7.
Outcome (E2b adapter, 9-step incidents; static dis15/30/60 = 13.439 / 21.502 / 25.434):

| alarm at | dis15 | dis30 | dis60 | dis_avg gain |
|---|---|---|---|---|
| onset + 1 | 13.257 | 21.375 | 25.947 | -0.068 |
| onset + 2 | 13.333 | 21.474 | 26.026 | -0.153 |
| onset + 4 | 13.465 | 21.819 | 26.271 | -0.393 |
| onset + 7 | 13.737 | 22.334 | 26.189 | -0.629 |

Earlier is clearly better, and with an alarm at onset + 1 or + 2 the 15 and 30 minute errors
drop below static. The 60 minute error always gets worse. alpha and b are shared across
horizons, so a correction learned from short-horizon targets is applied to 60 minute forecasts
that cross the release (E1). Two bottlenecks, then: the adapter shouldn't touch horizons it has
no evidence for, and detection must come early.

### E4: per-horizon adapter
Hypothesis: with alpha and b per (horizon, sensor), a horizon only moves once targets for it
are known, so the 60 minute output stays at the static forecast early in an incident while 15
and 30 minutes get corrected. That should turn the oracle + 1 result positive and make dis60
equal to static.
Change: `per_horizon: true` in NodeAdapt, parameters shaped (12, N). Run with oracle + 1, + 2
and then with the real detector.
Outcome: dis60 now equals static exactly (25.434) as intended, but the 15 minute gain mostly
disappears: oracle + 1 gives dis15/30 13.399 / 21.484 (gain +0.019), oracle + 2 gives +0.017,
the real detector -0.004. Per-horizon parameters get too few known targets each. Positive but
tiny; kept as an option, not the default.

### E4b: shared adapter, applied only up to horizon H
Hypothesis: one alpha and b per sensor learns from all known targets (as in E2b) but is applied
only to horizons 1..H. That keeps E2b's 15 and 30 minute gains and E4's untouched 60 minutes.
Change: `max_horizon: H` in NodeAdapt (outputs past H are left as the static forecast, and the
loss only uses horizons up to H). Oracle + 1, H = 3 and 6.
Outcome: H = 6 gives dis15/30/60 13.257 / 21.375 / 25.434, gain +0.103, the best so far; H = 3
gives +0.052. Both leave the normal period untouched. Kept: shared alpha and b, max_horizon 6.

### E5: neighbour-confirmed detection signal
Hypothesis: a sensor's z-scored t+1 error only counts when a graph neighbour is also high
(`min(z_v, max over neighbours z_u)`). Incidents span adjacent sensors, single-sensor noise
doesn't, so the false-alarm tail should shrink and a lower PageHinkley threshold becomes
possible, which means earlier alarms. Still river PageHinkley/ADWIN on a t+1 error signal.
Change: aggregations "nbrmax" and "nbrtop3" added to `drift.aggregate` and to the calibration
grid. Previous choice saved as `results/detector_E0.json`.
Success: on validation, a higher detection rate or a lower delay than E0's detector at
<= 1 false alarm per day.
Outcome: "nbrtop3" with PageHinkley (threshold 30, delta 0.2) detects 19 of 35 incidents (54 %)
with mean delay 3.7 steps at 0.80 false alarms per day. E0's detector: 15 of 35 (43 %), 6.9
steps, 0.92 per day. Better on all three, kept (`results/detector.json`). At threshold 50 the
same signal gives 46 % with only 0.18 to 0.37 false alarms per day, a useful fallback if false
alarms ever matter more than recall.

### E6: E4b adapter with the E5 detector, scan max_horizon and decay
Hypothesis: with alarms about 3 steps earlier, the E4b adapter keeps a positive gain with the
real detector (E4 with the old detector was at -0.004).
Change: node adapter, max_horizon in {6, 9}, decay in {0.8, 0.9, 0.95}, E5 detector.
Outcome (static with the E5 detector: dis_avg 20.125; detector finds 6.3 of 11.7 incidents per
seed at 0.80 false alarms per day):

| config | dis15 | dis30 | dis_avg gain | norm15 delta |
|---|---|---|---|---|
| H6, decay 0.8 | 13.399 | 21.465 | +0.025 | +0.0010 |
| H6, decay 0.9 | 13.437 | 21.518 | -0.005 | +0.0020 |
| H6, decay 0.95 | 13.501 | 21.587 | -0.049 | +0.0044 |
| H9, decay 0.9 | identical to H6 | | | |

H9 equals H6 because with 6 training windows no target past horizon 6 is ever known. Faster
decay is better, which again says the correction should be short lived. Kept: decay 0.8, the
first configuration with the real detector that beats static while passing the constraint.
The gain is small (+0.025 mph); the oracle bound in E4b was +0.103.

### E6b, E7, E8 (one batch on top of H6, decay 0.8)
- E6b: decay 0.7 and 0.6. Hypothesis: the trend continues.
- E7: local adaptation, `local_z` 3 and 5: only sensors whose neighbour-confirmed z went above
  the threshold in the last 6 steps, plus their neighbours, are trained. Hypothesis: false
  alarms cost less and the normal-period delta goes to about zero.
- E8: `active_steps` 6: after an alarm keep updating every step for 6 steps (no decay meanwhile).
  Hypothesis: helps once the incident has lasted a few steps; may hurt for 9-step incidents near
  the release. Should matter more for the 2 h scenario.
Outcome (static dis_avg 20.125):

| config | dis15 | dis30 | gain | norm15 delta |
|---|---|---|---|---|
| decay 0.7 | 13.393 | 21.449 | +0.033 | +0.0006 |
| decay 0.6 | 13.395 | 21.446 | +0.033 | +0.0004 |
| decay 0.8, local_z 3 | 13.409 | 21.490 | +0.014 | +0.0012 |
| decay 0.8, local_z 5 | 13.408 | 21.447 | +0.028 | +0.0017 |
| decay 0.8, active 6 | 13.338 | 21.550 | +0.017 | +0.0015 |

- Decay plateaus at 0.6 to 0.7. Kept: decay 0.7 (new best, +0.033).
- Local adaptation doesn't help at decay 0.8 (+0.014 and +0.028 vs +0.025). Rejected, the simpler
  global version stays. The adapter already keeps quiet sensors near zero on its own.
- Continuous updates give the best dis15 so far but hurt dis30. Not kept for the 9-step case;
  retested on the 2 h scenario in E11.
- Observation that changes the plan: false alarms now cost almost nothing (norm15 delta below
  0.001 at 0.8 false alarms a day). The detector was chosen under "at most 1 false alarm per
  day", a limit set when false alarms damaged the normal period. Recall and delay are now worth
  more, so the threshold should be tuned end to end on the final objective.

### E9: update hyperparameters (base: H6, decay 0.7)
Hypothesis: the defaults (lr 0.05, 20 steps, 6 windows, beta 0.01) were never tuned; at least
one of them has room.
Change: one at a time: lr 0.02 / 0.1, steps 10 / 40, n_windows 3 / 12, beta 0.001 / 0.1.
Outcome: pending.

### E10: lower detector threshold, tuned end to end
Hypothesis: a lower PageHinkley threshold on the nbrtop3 signal raises recall and cuts delay,
and the extra false alarms barely touch the normal period.
Change: detector override threshold 20, 12, 8 (delta 0.2), base H6 decay 0.7.
Success: higher gain with norm15 delta still <= 0.005.
Outcome: pending.

### E11: 2 hour incidents
Hypothesis: with long incidents the early-recovery bias lasts the whole event, so the adapter
gains much more, and continuous updates (E8) start to pay off.
Change: dur 24; static, H6 decay 0.7, plus active_steps 6 and 12.
Outcome: pending.





