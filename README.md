# EEG Emotional Word Association

Neurotech project using OpenNeuro dataset `ds007955`: EEG and autonomic responses during an emotional word association task.

The first modeling target is `NewWord`: whether a participant entered an original association (`1`) or selected a suggested word (`0`). Each event becomes one machine-learning row using response-locked EEG and autonomic features from the window before the response.

## Current Findings

The clearest technical result is that multimodal physiology improved prediction compared with EEG alone:

```text
EEG-only full valid-epoch model: balanced accuracy 0.532, ROC AUC 0.505
EEG + physiology full valid-epoch model: balanced accuracy 0.631, ROC AUC 0.707
```

The strongest neuroscience interpretation is exploratory:

```text
Original word associations and semantic transition structure show suggestive EEG/autonomic differences, especially around central theta/alpha activity and autonomic timing, but the statistical tests do not survive FDR correction in this small dataset.
```

So the main takeaway is not "this proves the neuroscience." It is more like: the full EEG + physiology pipeline works, multimodal signals helped prediction, and the neuroscience effects are interesting but still exploratory.

## Setup

```bash
datalad install -s https://github.com/OpenNeuroDatasets/ds007955.git data/ds007955
cd data/ds007955
datalad get sub-*/eeg/*_eeg.set sourcedata/physio/*.tsv
cd ../..

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Step 1 Smoke Test

```bash
.venv/bin/python scripts/run_pipeline.py --dataset data/ds007955 --subjects sub-01 --skip-model
```

This writes a one-subject feature table to `outputs/features_with_labels.csv`.

## Step 2 EEG Preprocessing

```bash
.venv/bin/python scripts/preprocess_eeg.py --dataset data/ds007955 --subjects sub-01 --make-plots
```

For the full EEG preprocessing pass:

```bash
.venv/bin/python scripts/preprocess_eeg.py --dataset data/ds007955 --make-plots
```

This writes:

1. `outputs/eeg_preprocessing/eeg_epoch_features.csv`
2. `outputs/eeg_preprocessing/eeg_channel_summary.csv`
3. `outputs/eeg_preprocessing/eeg_preprocessing_qc.csv`
4. `outputs/eeg_preprocessing/eeg_preprocessing_qc.json`
5. PSD comparison plots in `outputs/eeg_preprocessing/plots/`

Preprocessing choices and QC notes are in `docs/eeg_preprocessing.md`.

## Step 3 EEG Feature Ranking

```bash
.venv/bin/python scripts/rank_eeg_features.py
```

This ranks the EEG features from Step 2 and writes:

1. `outputs/feature_ranking/ranked_eeg_features.csv`
2. `outputs/feature_ranking/top_10_eeg_features.csv`
3. `outputs/feature_ranking/feature_ranking_report.md`
4. `outputs/feature_ranking/feature_ranking_summary.json`

Feature ranking notes are in `docs/eeg_feature_ranking.md`.

The selected Step 3 feature set is stored in `config/selected_eeg_features.txt`.

## Step 4 EEG Model Training

```bash
.venv/bin/python scripts/train_model.py
```

This combines the selected EEG features with the `new_word` label and trains three scikit-learn classifiers: gradient boosting, logistic regression, and random forest.

It writes:

1. `outputs/model_training/modeling_dataset.csv`
2. `outputs/model_training/model_metrics.csv`
3. `outputs/model_training/model_predictions.csv`
4. `outputs/model_training/feature_importance.csv`
5. `outputs/model_training/model_training_report.md`
6. `outputs/model_training/model_training_summary.json`

Modeling notes are in `docs/model_training.md`.

## Step 5 Multimodal Model Training

```bash
.venv/bin/python scripts/train_multimodal_model.py --dataset data/ds007955
```

This concatenates the selected EEG features, pre-response physiology summaries, metadata, and labels into one modeling table. It trains the same model set used in Step 4.

It writes:

1. `outputs/multimodal_model/multimodal_modeling_dataset.csv`
2. `outputs/multimodal_model/multimodal_model_metrics.csv`
3. `outputs/multimodal_model/multimodal_model_predictions.csv`
4. `outputs/multimodal_model/multimodal_feature_importance.csv`
5. `outputs/multimodal_model/multimodal_model_report.md`
6. `outputs/multimodal_model/multimodal_model_summary.json`

Multimodal modeling notes are in `docs/multimodal_model_training.md`.

## Step 6 Hypothesis Testing

```bash
.venv/bin/python scripts/test_hypothesis.py
```

This compares selected suggested words against original associations using subject-level condition contrasts. The question is whether the selected EEG and physiology features differ before the response.

It writes:

1. `outputs/hypothesis_testing/condition_counts_by_subject.csv`
2. `outputs/hypothesis_testing/subject_level_condition_tests.csv`
3. `outputs/hypothesis_testing/event_level_condition_tests.csv`
4. `outputs/hypothesis_testing/hypothesis_test_report.md`
5. `outputs/hypothesis_testing/hypothesis_test_summary.json`

The hypothesis and stats notes are in `docs/hypothesis_testing.md`.

## Step 7 Semantic Transition Analysis

```bash
.venv/bin/python scripts/test_semantic_transitions.py
```

This compares high-coherence and low-coherence emotional word transitions using the dataset's semantic similarity scores. It only uses selected-word events with nonzero semantic similarity, so the contrast is about associative structure rather than response mode.

It writes:

1. `outputs/semantic_transition_analysis/semantic_transition_dataset.csv`
2. `outputs/semantic_transition_analysis/semantic_condition_counts_by_subject.csv`
3. `outputs/semantic_transition_analysis/semantic_subject_level_tests.csv`
4. `outputs/semantic_transition_analysis/semantic_event_level_tests.csv`
5. `outputs/semantic_transition_analysis/semantic_continuous_correlations.csv`
6. `outputs/semantic_transition_analysis/semantic_transition_report.md`
7. `outputs/semantic_transition_analysis/semantic_transition_summary.json`

The semantic transition notes are in `docs/semantic_transition_analysis.md`.

## Pipeline Scope

1. Loads BIDS-style EEG/event files and Empatica physiology files.
2. Preprocesses EEG with average reference, 60 Hz notch filtering, and 1-40 Hz band-pass filtering.
3. Builds response-locked epochs from `-3.0` to `0.0` seconds before each word event.
4. Extracts EEG features: per-channel bandpower, relative bandpower, asymmetry, Hjorth parameters, entropy, and artifact summaries.
5. Extracts physiology features from EDA, BVP, IBI, and temperature when available.
6. Concatenates EEG, autonomic, behavioral/context, subject, and label columns into one feature table.
7. Ranks features and reports the top 10.
8. Trains scikit-learn classifiers with subject-group cross-validation when feasible.
9. Tests condition-level and semantic-transition hypotheses.

Pipeline outputs are written to `outputs/`.
