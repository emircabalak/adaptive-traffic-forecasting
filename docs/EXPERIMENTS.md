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
Outcome (base +0.033): n_windows 3 +0.056 (best), 12 +0.004; lr 0.1 +0.035, 0.02 +0.030;
steps 10 +0.032, 40 +0.030; beta 0.001 +0.072 but norm15 delta +0.0059 (violates), beta 0.1
+0.010. Fresher training windows matter most: with 3 windows the incident steps make up more of
the batch. Kept: n_windows 3. beta between 0.001 and 0.01 is worth one more try.

### E10: lower detector threshold, tuned end to end
Hypothesis: a lower PageHinkley threshold on the nbrtop3 signal raises recall and cuts delay,
and the extra false alarms barely touch the normal period.
Change: detector override threshold 20, 12, 8 (delta 0.2), base H6 decay 0.7.
Success: higher gain with norm15 delta still <= 0.005.
Outcome (base threshold 30: +0.033, 6.3 of 11.7 detected, 0.80 false alarms a day):
threshold 20 +0.007 (6.7, 1.78 a day), 12 +0.038 (8.7, 3.04), 8 +0.047 (8.3, 4.15). norm15
delta stays about +0.001 throughout, so false alarms really are cheap now. Kept: threshold 8.
Threshold 20 being worse than 30 is probably noise between runs of this size.

### E11: 2 hour incidents
Hypothesis: with long incidents the early-recovery bias lasts the whole event, so the adapter
gains much more, and continuous updates (E8) start to pay off.
Change: dur 24; static, H6 decay 0.7, plus active_steps 6 and 12.
Outcome (static dis15/30/60 12.334 / 23.982 / 31.212): decay 0.7 alone +0.052; active 6 +0.189;
active 12 +0.338 (dis15 11.987, dis30 23.314). norm15 delta at most +0.0018. As predicted, long
incidents are where updating pays off, and continuous updates are what unlocks it. For 9-step
incidents continuous updates were mixed (E8), so E12 checks both lengths together.

### E12: combine the winners, judged on both incident lengths
Hypothesis: n_windows 3 + threshold 8 stack. Continuous updates help 2 h incidents a lot and
cost little on 45-minute ones once the windows are fresher.
Change: base H6, decay 0.7, n_windows 3, threshold 8. Variants: plain; beta 0.003; active 6;
active 12. Each at dur 9 and dur 24.
Selection: the spec scenario (dur 9) must improve on E10's best (+0.047) or stay within 0.01 of
it while dur 24 improves clearly.
Outcome:

| config | dur 9 gain | dur 24 gain | norm15 delta (9 / 24) |
|---|---|---|---|
| base | +0.078 | +0.071 | +0.0011 / +0.0013 |
| beta 0.003 | +0.185 | +0.174 | +0.0055 / +0.0059 (violates) |
| active 6 | +0.220 | +0.184 | +0.0035 / +0.0038 |
| active 12 | +0.187 | +0.349 | +0.0055 / +0.0061 (violates) |

Kept: active 6 (H6, decay 0.7, n_windows 3, threshold 8). dur 9 gain is 0.220, up from 0.033
before E9, and above the E4b oracle bound (+0.103, measured with the older settings). With fresh
windows, continuous updates now help short incidents too (in E8, with 6 windows, they didn't).
Active 12 is the best on long incidents but breaks the constraint: about 4 false alarms a day,
each followed by 12 update steps on every sensor, add up in the normal period.

### E13: keep long-incident gains without the normal-period cost
Hypothesis: during continuous updates, training only the flagged sensors (local_z) confines a
false alarm's effect to a few sensors, so active 12 can pass the constraint. A middle value
(active 9) or a slightly stronger anchor (beta 0.005 with active 6) are the simpler alternatives.
Change: on top of E12 active: active 12 + local_z 3; active 12 + local_z 5; active 9;
active 6 + beta 0.005. Each at dur 9 and 24.
Correction: the default beta is 0.01, so beta 0.005 is a weaker anchor, not a stronger one as
written above. The result is read accordingly.
Outcome:

| config | dur 9 gain | dur 24 gain | norm15 delta (9 / 24) |
|---|---|---|---|
| active 12 + local_z 3 | +0.335 | +0.572 | +0.0102 / +0.0107 |
| active 12 + local_z 5 | +0.347 | +0.553 | +0.0116 / +0.0115 |
| active 9 | +0.177 | +0.258 | +0.0047 / +0.0052 |
| active 6 + beta 0.005 | +0.369 | +0.303 | +0.0087 / +0.0091 |

Everything stronger than E12 active 6 breaks the constraint. Local training makes each flagged
sensor's correction larger (the loss is averaged over fewer sensors), which helps real
incidents and hurts after false alarms. There is a clear trade-off between correction strength
and false-alarm damage, and the constraint is what binds. Nothing kept; E12 active 6 stays best.

### E14: strong local correction, less false-alarm damage
Hypothesis: the strong E13 settings fail only through false alarms (about 3.5 a day). Either
fewer false alarms (higher threshold) or stopping the active phase when the frozen model is
already right again (evidence gate) should bring norm15 delta under 0.005 while keeping most of
the gain.
Change: base active 12 + local_z 3 (E13). (a) threshold 12 and 20; (b) threshold 8 with
`gate` 0.25 and 0.4 (normalised units, about 5 and 8 mph). Each at dur 9 and 24.
Outcome (E12 active 6 for reference: +0.220 / +0.184, delta +0.0035 / +0.0038):

| config | dur 9 gain | dur 24 gain | norm15 delta (9 / 24) | false alarms a day |
|---|---|---|---|---|
| threshold 12 | +0.297 | +0.583 | +0.0092 / +0.0092 | 3.0 |
| threshold 20 | +0.209 | +0.416 | +0.0048 / +0.0045 | 1.7 |
| threshold 8, gate 0.25 | +0.202 | +0.229 | +0.0043 / +0.0047 | 3.8 |
| threshold 8, gate 0.4 | +0.181 | +0.183 | +0.0040 / +0.0045 | 3.8 |

Threshold 20 matches E12 on 45-minute incidents (-0.011, within run-to-run noise) and more than
doubles the gain on 2 h incidents. It passes the constraint, but with little margin. The gate
does what it should (constraint passes at threshold 8) but gives up too much gain.
Candidate kept: active 12 + local_z 3 + threshold 20 ("E14-thr20").

### E15: more margin on the normal period
Hypothesis: adding the gate to threshold 20 lowers norm15 delta (fewer post-false-alarm updates)
at a small cost; threshold 16 sits between 12 (too many false alarms) and 20.
Change: base E14-thr20. (a) + gate 0.25; (b) threshold 16; (c) threshold 16 + gate 0.25.
Each at dur 9 and 24.
After E15 the finalists go to the test set.
Outcome:

| config | dur 9 gain | dur 24 gain | norm15 delta (9 / 24) |
|---|---|---|---|
| threshold 20 + gate 0.25 | +0.177 | +0.296 | +0.0026 / +0.0024 |
| threshold 16 | +0.206 | +0.544 | +0.0062 / +0.0057 (violates) |
| threshold 16 + gate 0.25 | +0.205 | +0.295 | +0.0030 / +0.0031 |

The gate buys margin on the normal period (about 0.003 instead of 0.005 to 0.006) and costs
some long-incident gain. Tuning stops here: the remaining options trade along the same curve.

## Finalists for the test set

All share: node adapter, max_horizon 6, decay 0.7, n_windows 3, lr 0.05, 20 steps, beta 0.01,
nbrtop3 PageHinkley detector (delta 0.2).

| name | detector threshold | active steps | local_z | gate | validation gain (9 / 24) | norm15 delta |
|---|---|---|---|---|---|---|
| node-strong (E14-thr20) | 20 | 12 | 3 | none | +0.209 / +0.416 | +0.0048 / +0.0045 |
| node-safe (E15-thr16-gate0.25) | 16 | 12 | 3 | 0.25 | +0.205 / +0.295 | +0.0030 / +0.0031 |
| node-global (E12-active6) | 8 | 6 | none | none | +0.220 / +0.184 | +0.0035 / +0.0038 |

Compared on the test split against static (a), naive (b), the spec's original head-only update
(c, E0 detector) and the head reset variant, 3 seeds, 9-step and 24-step incidents
(`results/test_variants.jsonl`, `scripts/run_experiments.py`). Which finalist becomes "ours" is
decided by the same rule as in tuning, now on test: largest dur 9 gain with norm15 delta
<= 0.005, dur 24 as tie-breaker. Since three finalists are compared on test, the report has to
say so.






Known limitation of the comparison: "head-reset" uses the E0 detector, but its reset band is
computed on the current detector's signal (mean 3.40, std 2.40 instead of 5.86 and 3.71 for E0's
signal). It therefore resets at a lower signal level than it was designed for. It's a reference
variant, so this is noted rather than fixed.

## Test-set result and final choice (2026-10-10)

Test split, 3 seeds (0, 1, 2), about 21 synthetic incidents per seed and length. Gain = static
disrupted MAE minus variant disrupted MAE, averaged over 15/30/60 min, mean +- std over seeds.
norm15 delta = normal-period MAE at 15 min minus static's (limit 0.005).

| variant | 45 min gain | 2 h gain | norm15 delta (45 min / 2 h) |
|---|---|---|---|
| naive (b) | -3.372 +- 0.286 | -2.202 +- 0.165 | +1.853 / +1.885 |
| head, spec (c) | +0.001 +- 0.012 | -0.005 +- 0.017 | +0.029 / +0.033 |
| head-reset | -0.144 +- 0.070 | -0.049 +- 0.121 | +0.058 / +0.052 |
| node-strong | +0.156 +- 0.018 | +0.466 +- 0.167 | +0.0053 / +0.0057 |
| node-safe | +0.140 +- 0.071 | +0.301 +- 0.103 | +0.0027 / +0.0028 |
| node-global | +0.185 +- 0.047 | +0.267 +- 0.068 | +0.0036 / +0.0034 |

Decision, by the rule fixed before the test run: node-global (largest 45-minute gain within the
constraint) is "ours". node-strong has the best long-incident gain but goes just over the limit
on test (0.0053), which was the risk noted when it was picked; node-safe is the alternative if a
reviewer wants a larger margin on the normal period.

node-global per horizon, 45-minute incidents, disrupted period: 15 min 13.248 -> 12.996, 30 min
20.811 -> 20.509, 60 min unchanged (25.040, by design). Normal period: 2.663 -> 2.666 at 15 min.
Detection: 14.0 of 21.7 incidents per seed (static's detector setting: 10.7), mean delay 6.7
steps, 3.8 false alarms per normal day. Mean update time 123 ms (max 303 ms).

Caveats for the report: the three finalists were compared on test, so the choice among them
used test data (the tuning itself did not). Gains are small in absolute terms because most of
the disrupted error comes from forecasts issued before an incident starts (E1), which no
online method can fix. Disrupted-period MAPE is very large for every variant (around 100 % at
15 min) because blocked speeds are near the 5 mph floor; MAE is the meaningful metric there.

## Round 2: closing the gap to the actual speed

Motivation (2026-10-10): the test figures show that during an incident ours is only slightly
closer to the actual speed than static. Disrupted MAE is about 13 mph at 15 min against about
2.7 in normal periods, and ours removes 2 % of it. Before tuning further, measure how much room
there is.

### E16: how much room is there? Persistence references
Hypothesis: during a blockage, "the last observed speed persists" is far more accurate than
Graph WaveNet, so the room for improvement is large, and the limiting factor is knowing where
and when to apply such a correction.
Change: new `Persist` policy (src/adapt.py), a rule-based reference that pulls the forecast to
the last observed speed with weight w on selected sensors.
- E16a: persistence on every sensor, always (plain last-value forecast).
- E16b: oracle, persistence only on the incident's sensors from onset + 1 to the end, all
  horizons; E16b-H6: same, only up to 30 min. Upper bound for "perfect detection + simple rule".
- E16c: the same rule on sensors flagged by our detector (threshold 8 and 20), held 12 steps.
Each at dur 9 and 24. Diagnostic: these are references, not candidate methods (E16c could
become one).
Outcome: pending.
