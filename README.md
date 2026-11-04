# Adaptive traffic speed forecasting under sudden disruptions

Course project, Adaptive Data Processing (DE0297), Riga Technical University.

A Graph WaveNet forecaster trained on METR-LA is run as if it were live, one 5-minute step at a
time. A drift detector watches its own forecast errors, and when something like an accident
throws the forecasts off, a small online update corrects them while the disruption is still
going on, without retraining and without making normal periods worse.

## Results (test split, 3 seeds)

The frozen baseline reproduces the paper: test MAE 2.70 / 3.10 / 3.53 mph at 15 / 30 / 60 min
(Wu et al. report 2.69 / 3.07 / 3.53).

Improvement in disrupted-period MAE over the static model, averaged over 15/30/60 min, on
synthetic blockages of 5 adjacent sensors:

| method | 45 min blockages | 2 h blockages | normal period, 15 min |
|---|---|---|---|
| naive fine-tuning of all weights | 3.37 mph worse | 2.20 mph worse | 1.85 mph worse |
| gradient steps on the output head | no change | no change | 0.03 mph worse |
| per-sensor output adapter (ours) | 0.19 mph better | 0.27 mph better | 0.004 mph worse |
| severe-slowdown rule | 0.11 mph better | 0.85 mph better | 0.004 mph worse |

Most of the remaining error comes from forecasts issued before an incident starts or that cross
the moment it clears, which no online method can know about. On real slowdowns in the data the
static model already beats a "the slowdown persists" correction, so the rule's gains are specific
to the synthetic incidents. Details: `docs/EXPERIMENTS.md`.

## Layout

```
src/        data loading, Graph WaveNet wrapper, streaming replay, drift detection,
            online update policies, incident injection and mining, metrics
scripts/    training, tuning on validation, test experiments, plots
docs/       PLAN.md (plan and decisions), EXPERIMENTS.md (every experiment), IDEAS.md
results/    CSVs and figures
third_party/Graph-WaveNet   the authors' code, as a submodule, unmodified
```

## Setup

```
git clone --recurse-submodules https://github.com/emircabalak/adaptive-traffic-forecasting
python -m venv .venv
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

Data: download `metr-la.h5` from the DCRNN Google Drive folder (linked in the Graph WaveNet
README) into `data/`, and `adj_mx.pkl` plus `distances_la_2012.csv` from the DCRNN repository
into `data/sensor_graph/`. Then:

```
python src/data.py
python -c "import sys; sys.path.insert(0,'src'); import data; data.convert_for_pandas3()"
python third_party/Graph-WaveNet/generate_training_data.py --output_dir=data/METR-LA --traffic_df_filename=data/metr-la-pd3.h5
python scripts/train_resumable.py          # about 2 h on a laptop GPU, resumes after interruptions
python scripts/eval_baseline.py
python scripts/run_experiments.py          # all variants in results/test_variants.jsonl
python scripts/plots.py
```

The `scripts/start_*.ps1` files start the long jobs in the background on Windows; each one can
be run again after a shutdown and continues where it stopped.

## References

- Wu et al., Graph WaveNet for Deep Spatial-Temporal Graph Modeling, IJCAI 2019.
- Li et al., Diffusion Convolutional Recurrent Neural Network, ICLR 2018 (METR-LA dataset).
- river (online machine learning in Python) for ADWIN and Page-Hinkley.
