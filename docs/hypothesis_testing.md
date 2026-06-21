# Hypothesis Testing

The hypothesis-testing workflow is implemented in `scripts/test_hypothesis.py`.

## Comparison

The dataset does not include a separate external control group. The comparison is therefore defined within the task:

```text
selected suggested word = comparison condition
original association = experimental condition
```

## Hypothesis

Primary hypothesis:

```text
Original associations differ from selected suggested words in pre-response EEG/autonomic state.
```

The analysis uses the same response-locked window as preprocessing and modeling:

```text
-3.0 s to 0.0 s
```

## Features Tested

The test includes:

1. The selected top 10 EEG features from `config/selected_eeg_features.txt`
2. Physiology features from the multimodal modeling table

Word text and semantic correlation are not treated as neurophysiology features.

## Statistical Approach

The primary analysis is subject-level:

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

Event-level Welch tests are also written as a secondary descriptive check.

## Outputs

The workflow writes:

1. `outputs/hypothesis_testing/condition_counts_by_subject.csv`
2. `outputs/hypothesis_testing/subject_level_condition_tests.csv`
3. `outputs/hypothesis_testing/event_level_condition_tests.csv`
4. `outputs/hypothesis_testing/hypothesis_test_report.md`
5. `outputs/hypothesis_testing/hypothesis_test_summary.json`
