# Multimodal Model Training

The multimodal workflow is implemented in `scripts/train_multimodal_model.py`.

## Input

The script combines:

1. Selected EEG features from `config/selected_eeg_features.txt`
2. EEG epoch rows from `outputs/eeg_preprocessing/eeg_epoch_features.csv`
3. Empatica E4 physiology files from `sourcedata/physio`
4. The `new_word` label

Physiology is summarized in the same response-locked window used for EEG:

```text
-3.0 s to 0.0 s
```

## Feature Table

The combined dataset is written to:

```text
outputs/multimodal_model/multimodal_modeling_dataset.csv
```

The table keeps word text and semantic correlation as metadata. They are not used as model inputs because they are derived from the behavioral response itself.

## Physiology Features

For EDA, BVP, IBI, and skin temperature, the workflow extracts:

1. availability flag
2. sample count
3. mean
4. standard deviation
5. minimum
6. maximum
7. range
8. slope

Participant 7 has no physiology files, so physiology values for that participant are marked unavailable and handled by model imputation.

## Models

The script evaluates:

1. Gradient boosting
2. Logistic regression
3. Random forest

Subject-group cross-validation is used when possible.

## Current Results

Using all valid epochs, the best model was random forest:

```text
rows used: 141
features used: 42
balanced accuracy: 0.631
ROC AUC: 0.707
```

Using only valid epochs that were not artifact-flagged, the best model was logistic regression:

```text
rows used: 32
features used: 42
balanced accuracy: 0.683
ROC AUC: 0.806
```

The full valid-epoch multimodal model performed better than the EEG-only full valid-epoch model. The cleaner subset still performs well, but the sample size is much smaller.

## Outputs

The workflow writes:

1. `outputs/multimodal_model/multimodal_modeling_dataset.csv`
2. `outputs/multimodal_model/multimodal_model_metrics.csv`
3. `outputs/multimodal_model/multimodal_model_predictions.csv`
4. `outputs/multimodal_model/multimodal_feature_importance.csv`
5. `outputs/multimodal_model/multimodal_model_report.md`
6. `outputs/multimodal_model/multimodal_model_summary.json`
