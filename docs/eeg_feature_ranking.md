# EEG Feature Ranking

The feature ranking workflow is implemented in `scripts/rank_eeg_features.py`.

## Input

The script reads:

```text
outputs/eeg_preprocessing/eeg_epoch_features.csv
```

By default, all valid epochs are used. Artifact-flagged epochs can be excluded for a sensitivity check with:

```bash
.venv/bin/python scripts/rank_eeg_features.py --exclude-artifact-flagged
```

## Target

The ranking target is `new_word`:

```text
0 = selected a suggested word
1 = entered an original association
```

## Ranking Method

Each numeric EEG-derived feature is evaluated as a single-feature predictor. Behavioral context columns, labels, identifiers, and artifact flags are not ranked as candidate EEG features. The final screening score combines:

1. Cross-validated balanced accuracy from a single-feature logistic regression.
2. Cross-validated AUC from the same single-feature model.
3. Mutual information with the target.
4. Absolute Cohen's d between the two target classes.
5. Absolute point-biserial correlation with the target.

Subject-aware cross-validation is used when possible so that ranking is less dependent on repeated events from the same participant.

## Outputs

The workflow writes:

1. `outputs/feature_ranking/ranked_eeg_features.csv`
2. `outputs/feature_ranking/top_10_eeg_features.csv`
3. `outputs/feature_ranking/feature_ranking_report.md`
4. `outputs/feature_ranking/feature_ranking_summary.json`

The top-10 list is a screening result. Final model performance should be evaluated separately in the model-training step.

## Selected Top 10

These are the selected EEG features for the next modeling step:

1. `eeg_C3_theta_power_log10`
2. `eeg_C3_delta_power_log10`
3. `eeg_C3_alpha_power_log10`
4. `eeg_FP1_delta_power_log10`
5. `eeg_FP1_delta_relative_power`
6. `eeg_FP1_alpha_relative_power`
7. `eeg_asymmetry_C3_C4_beta`
8. `eeg_T5_delta_relative_power`
9. `eeg_FP1_entropy`
10. `eeg_C3_hjorth_activity`

This set keeps the ranking grounded in interpretable EEG properties: scalp location, frequency-band activity, asymmetry, and signal complexity.
