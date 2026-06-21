# Model Training

The modeling workflow is implemented in `scripts/train_model.py`.

## Input

The script combines:

1. EEG epoch features from `outputs/eeg_preprocessing/eeg_epoch_features.csv`
2. Selected Step 3 features from `config/selected_eeg_features.txt`
3. Labels from the `new_word` column

The output table used for modeling is written to:

```text
outputs/model_training/modeling_dataset.csv
```

## Target

The prediction target is:

```text
new_word
```

Class meaning:

```text
0 = selected a suggested word
1 = entered an original association
```

## Models

The main model is `GradientBoostingClassifier`.

Two baselines are also evaluated:

1. Logistic regression
2. Random forest

The comparison is useful because the dataset is small. If gradient boosting performs well but simpler models fail, that may indicate overfitting rather than a stable EEG signal.

## Validation

The script uses stratified subject-group cross-validation when possible. This keeps events from the same participant grouped during validation, which is stricter than randomly splitting individual events.

The main metric is balanced accuracy because the two classes are not perfectly even.

## Current Results

Using all valid epochs, the best model was random forest:

```text
rows used: 141
best balanced accuracy: 0.532
best ROC AUC: 0.505
```

Using only valid epochs that were not artifact-flagged, the best model was logistic regression:

```text
rows used: 32
best balanced accuracy: 0.710
best ROC AUC: 0.893
```

The cleaner subset result is stronger, but it uses far fewer trials. This suggests that artifact handling is important and should be treated as a modeling sensitivity issue rather than an afterthought.

## Outputs

The workflow writes:

1. `outputs/model_training/modeling_dataset.csv`
2. `outputs/model_training/model_metrics.csv`
3. `outputs/model_training/model_predictions.csv`
4. `outputs/model_training/feature_importance.csv`
5. `outputs/model_training/model_training_report.md`
6. `outputs/model_training/model_training_summary.json`
