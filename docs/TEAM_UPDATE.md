# Update on the forecasting and online adaptation part

Hi all, here's where my part stands and what changed compared with the original spec. Code and
full notes are in the repo: `docs/PLAN.md` (plan and decisions), `docs/EXPERIMENTS.md` (every
experiment, the hypothesis behind it and the result) and `results/` (CSVs and figures).

## Baseline

Graph WaveNet trained with the authors' code and settings reproduces the paper on METR-LA:
test MAE 2.70 / 3.10 / 3.53 mph at 15 / 30 / 60 min, against 2.69 / 3.07 / 3.53 in the paper.
The authors' code needed one compatibility fix for current PyTorch, done from our side
(`src/gwn.py`); their files are untouched.

## Streaming setup

`src/replay.py` replays the test split step by step. A forecast's ground truth is only released
once its target time has passed, and two automated checks confirm that nothing sees future
data. Synthetic incidents follow the spec (5 adjacent sensors at 0.15x speed for 45 minutes).
I also added 2-hour incidents, because real incidents are often long (the data contains a real
one on 18 June 2012 that lasted over 5 hours).

## What changed from the spec, and why

1. The spec's update (a few gradient steps on the output head) doesn't help. On test it is
   within noise of the static model and slightly worsens the normal period. The head is shared
   by all 207 sensors, so it can't correct 5 of them without moving the rest.
2. Our update is now a small per-sensor adapter on the output: two numbers per sensor that pull
   the forecast towards the latest observed speed and shift it. Graph WaveNet itself stays
   completely frozen. The adapter only touches the 15 and 30 minute outputs; forecasts an hour
   ahead mostly cross the end of the incident, where a correction does more harm than good.
3. Detection still uses river's PageHinkley on the 5-minute-ahead error, as in the spec, but each
   sensor's error is normalised by its usual error at that time of day, and a sensor only counts
   when a neighbouring sensor agrees. This catches more incidents and catches them earlier.
4. After an alarm the adapter keeps updating for 30 minutes on the newest data, then fades back
   to zero within about half an hour.

## Results on the test split (3 seeds)

Disrupted-period MAE averaged over 15/30/60 min, improvement over the static model:

| variant | 45 min incidents | 2 h incidents | normal period, 15 min |
|---|---|---|---|
| naive fine-tuning (b) | 3.37 mph worse | 2.20 mph worse | 1.85 mph worse |
| spec's head-only update (c) | no change | no change | 0.03 mph worse |
| ours (node adapter) | 0.19 mph better | 0.27 mph better | 0.004 mph worse (within our 0.005 limit) |

At 15 minutes during 45-minute incidents, ours goes from 13.25 to 13.00 mph MAE. Each update
takes about 0.1 s, far below the 5-minute budget. Naive fine-tuning shows the forgetting the spec
expected.

The gains are modest, and there's a measured reason: for 45-minute incidents, most of the error
comes from forecasts made before the incident started, which no online method can fix. At 60
minutes that's 82 % of the error. Long incidents leave more room, which is why the gain is
larger there.

Figures: `results/fig_mae_bars_dur9.png`, `results/fig_mae_bars_dur24.png`, and two example
incidents (`results/fig_incident_seed0_dur9_6.png`, `..._dur24_12.png`; these are best cases,
picked to show the mechanism).

## Still open

- Real (mined) incidents as a test scenario; the mining works but needs tighter thresholds.
- The dashboard (lowest priority in the spec).

Questions welcome. If anyone wants to rerun something, every long job can be stopped and
resumed; the commands are at the top of `docs/PLAN.md`.
