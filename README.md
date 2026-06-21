# EEG Emotional Word Association

Neurotech pipeline for OpenNeuro dataset `ds007955`: EEG and autonomic responses during an emotional word association task.

The first modeling target is `NewWord`: whether a participant entered an original association (`1`) or selected a suggested word (`0`). Each event becomes one machine-learning row using response-locked EEG and autonomic features from the window before the response.

## Setup

```bash
datalad install -s https://github.com/OpenNeuroDatasets/ds007955.git data/ds007955
cd data/ds007955
datalad get sub-*/eeg/*_eeg.set sourcedata/physio/*.tsv
cd ../..

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## First Smoke Test

```bash
.venv/bin/python scripts/run_pipeline.py --dataset data/ds007955 --subjects sub-01 --skip-model
```

This writes a one-subject feature table to `outputs/features_with_labels.csv`.

## Pipeline Scope

1. Loads BIDS-style EEG/event files and Empatica physiology files.
2. Preprocesses EEG with average reference, 60 Hz notch filtering, and 1-40 Hz band-pass filtering.
3. Builds response-locked epochs from `-3.0` to `0.0` seconds before each word event.
4. Extracts EEG features: per-channel bandpower, relative bandpower, asymmetry, Hjorth parameters, entropy, and artifact summaries.
5. Extracts physiology features from EDA, BVP, IBI, and temperature when available.
6. Concatenates EEG, autonomic, behavioral/context, subject, and label columns into one feature table.
7. Ranks features and reports the top 10.
8. Trains a scikit-learn `GradientBoostingClassifier` with subject-group cross-validation when feasible.

Pipeline outputs are written to `outputs/`.
