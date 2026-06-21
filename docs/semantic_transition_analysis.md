# Semantic Transition Analysis

The semantic transition script is `scripts/test_semantic_transitions.py`.

## Comparison

This analysis compares selected-word events with different semantic transition strengths:

```text
high semantic coherence = control condition
low semantic coherence = divergent transition condition
```

The semantic transition score comes from the dataset's `Correlation` column, which measures similarity between consecutive words.

## Why This Contrast

The original-vs-selected comparison is useful for prediction, but it mixes response mode with semantic similarity because original words often have `Correlation = 0`.

This analysis only uses selected-word events with nonzero semantic similarity. That makes the comparison more specifically about emotional word association structure:

```text
Do semantically close emotional associations have a different pre-response brain/autonomic profile than semantically distant emotional associations?
```

## Question

Main question:

```text
Lower-coherence emotional word transitions differ from higher-coherence transitions in pre-response EEG/autonomic state.
```

Possible interpretation:

```text
Low-coherence transitions may reflect broader associative search, greater cognitive-affective control, or a shift away from a tightly coherent emotional chain.
```

## Condition Definition

For each subject, selected-word events with nonzero semantic similarity are split by that subject's median `Correlation` value:

```text
Correlation <= subject median = low semantic coherence
Correlation > subject median = high semantic coherence
```

The split is done within each subject so one person's similarity range is not treated as interchangeable with another person's range.

## Statistical Approach

The main test is subject-level:

1. Average each feature within each condition for each subject.
2. Compute the paired difference: low coherence minus high coherence.
3. Test whether the subject-level difference differs from zero.
4. Correct p-values across features using Benjamini-Hochberg FDR.

Event-level Welch tests are saved as secondary checks.

The script also tests continuous semantic similarity. For each subject, it correlates each feature with `Correlation`, then checks whether the subject-level correlation is consistently different from zero.

## Current Results

The current semantic-transition run used:

```text
events used: 60
subjects used: 7
high-coherence events: 23
low-coherence events: 37
features tested: 42
```

No feature survived FDR correction in the high-vs-low coherence comparison.

The strongest continuous semantic-similarity trend was:

```text
eeg_C3_theta_power_log10
mean within-subject r = 0.774
FDR p = 0.603
```

This points toward a possible relationship between semantic coherence and central theta activity, but it is still exploratory and does not survive correction.

## Outputs

The script writes:

1. `outputs/semantic_transition_analysis/semantic_transition_dataset.csv`
2. `outputs/semantic_transition_analysis/semantic_condition_counts_by_subject.csv`
3. `outputs/semantic_transition_analysis/semantic_subject_level_tests.csv`
4. `outputs/semantic_transition_analysis/semantic_event_level_tests.csv`
5. `outputs/semantic_transition_analysis/semantic_continuous_correlations.csv`
6. `outputs/semantic_transition_analysis/semantic_transition_report.md`
7. `outputs/semantic_transition_analysis/semantic_transition_summary.json`
