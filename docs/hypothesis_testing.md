# Hypothesis Testing

The hypothesis-testing script is `scripts/test_hypothesis.py`.

## Comparison

This dataset does not have a separate external control group, so the comparison is defined inside the task:

```text
selected suggested word = comparison condition
original association = experimental condition
```

## Question

The basic question:

```text
Original associations differ from selected suggested words in pre-response EEG/autonomic state.
```

This uses the same response-locked window as preprocessing and modeling:

```text
-3.0 s to 0.0 s
```

## Features Tested

The test uses:

1. The selected top 10 EEG features from `config/selected_eeg_features.txt`
2. Physiology features from the multimodal modeling table

Word text and semantic correlation are kept out of the neurophysiology feature set.

## Statistical Approach

The main analysis is subject-level:

1. Compute each subject's mean value for selected-word events.
2. Compute each subject's mean value for original-association events.
3. Test the subject-level difference: original minus selected.

For each feature, the script reports:

1. mean condition difference
2. paired t-test p-value
3. Wilcoxon p-value
4. bootstrap 95% confidence interval
5. paired effect size
6. FDR-corrected p-value

Event-level Welch tests are also saved as a secondary check.

## Current Results

The current run used:

```text
events used: 141
subjects used: 9
selected-word events: 61
original-association events: 80
features tested: 42
```

The strongest subject-level effects were:

```text
eeg_C3_theta_power_log10       p = 0.132, FDR p = 0.495, effect size = 0.602
physio_ibi_slope               p = 0.032, FDR p = 0.495, effect size = 1.906
eeg_C3_alpha_power_log10       p = 0.077, FDR p = 0.495, effect size = 0.732
eeg_FP1_delta_power_log10      p = 0.132, FDR p = 0.495, effect size = 0.604
physio_bvp_mean                p = 0.071, FDR p = 0.495, effect size = -0.828
physio_eda_std                 p = 0.091, FDR p = 0.495, effect size = 0.762
```

No feature survived FDR correction at the subject level. So this is not a confirmed effect. The more careful read is that there may be EEG/autonomic differences between original associations and selected suggested words, but this dataset is too small/noisy to make that claim strongly.

## Outputs

The script writes:

1. `outputs/hypothesis_testing/condition_counts_by_subject.csv`
2. `outputs/hypothesis_testing/subject_level_condition_tests.csv`
3. `outputs/hypothesis_testing/event_level_condition_tests.csv`
4. `outputs/hypothesis_testing/hypothesis_test_report.md`
5. `outputs/hypothesis_testing/hypothesis_test_summary.json`
