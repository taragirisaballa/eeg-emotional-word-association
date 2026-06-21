# EEG Emotional Word Association

Analysis of EEG, autonomic physiology, and behavioral responses during an emotional word association task from OpenNeuro dataset `ds007955`.

Participants responded to fear-related word prompts either by selecting a suggested word or entering an original association. The project looks at whether pre-response brain and physiology signals carry information about that response, and whether semantic movement through emotional associations shows measurable EEG/autonomic structure.

## Main Result

The strongest result is predictive rather than confirmatory:

```text
EEG-only model:          balanced accuracy 0.532, ROC AUC 0.505
EEG + physiology model:  balanced accuracy 0.631, ROC AUC 0.707
```

Adding autonomic physiology improved prediction over EEG alone. That suggests the emotional word association task is better captured as a multimodal brain-body signal than as an EEG-only problem.

The semantic analysis is the more task-specific neuroscience angle. Higher semantic coherence showed an exploratory positive relationship with C3 theta power:

```text
mean within-subject r = 0.774
FDR p = 0.603
```

That points toward a possible link between coherent emotional associations and central theta activity, but it does not survive correction.

The hypothesis tests were more cautious. Original associations and semantic transition structure showed some suggestive EEG/autonomic patterns, especially around central theta/alpha activity and autonomic timing, but none of the tested features survived FDR correction. I would treat those results as exploratory, not definitive.

## Data

The dataset includes:

1. EEG from an OpenBCI Cyton headset
2. Empatica E4 physiology: EDA, BVP, IBI, and skin temperature
3. Word-response events
4. Semantic similarity scores between consecutive words

EEG was recorded at 250 Hz from eight channels. The `.set` recordings expose channel labels as:

```text
FP1, FP2, C3, C4, T5, T6, O1, O2
```

Participant 7 does not have physiology files, so those physiology features are marked unavailable and handled by imputation during modeling.

## Analysis

The analysis uses response-locked windows from `-3.0` to `0.0` seconds before each word response.

The main processing path:

1. Load EEG, physiology, and event files.
2. Filter EEG with average reference, 60 Hz notch filtering, and 1-40 Hz band-pass filtering.
3. Extract EEG features: bandpower, relative bandpower, asymmetry, Hjorth features, entropy, and amplitude summaries.
4. Rank EEG features and keep a selected top-10 feature set.
5. Build EEG-only and EEG + physiology modeling tables.
6. Train scikit-learn classifiers with subject-group cross-validation.
7. Test exploratory condition contrasts:
   selected words vs original associations
   high-coherence vs low-coherence semantic transitions

The selected EEG features are stored in:

```text
config/selected_eeg_features.txt
```

## Models

The modeling scripts compare:

1. Gradient boosting
2. Logistic regression
3. Random forest

For the full valid-epoch multimodal table, random forest performed best:

```text
balanced accuracy: 0.631
ROC AUC: 0.707
```

On the cleaner non-artifact subset, logistic regression performed best, but that subset only had 32 rows, so I treat it as a sensitivity check rather than the headline result.

## Hypothesis Tests

Two condition analyses are included.

Original associations vs selected suggested words:

```text
events used: 141
subjects used: 9
selected-word events: 61
original-association events: 80
```

The strongest effects involved central theta/alpha EEG power, left-frontal delta power, and autonomic measures, but none survived FDR correction.

Semantic transition analysis:

```text
events used: 60
subjects used: 7
high-coherence events: 23
low-coherence events: 37
```

The strongest trend was a positive within-subject relationship between semantic similarity and C3 theta power:

```text
mean within-subject r = 0.774
FDR p = 0.603
```

That is interesting, but not statistically strong enough to claim as a confirmed effect.

## Setup

```bash
datalad install -s https://github.com/OpenNeuroDatasets/ds007955.git data/ds007955
cd data/ds007955
datalad get sub-*/eeg/*_eeg.set sourcedata/physio/*.tsv
cd ../..

python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Running The Analysis

Preprocess EEG:

```bash
.venv/bin/python scripts/preprocess_eeg.py --dataset data/ds007955 --make-plots
```

Rank EEG features:

```bash
.venv/bin/python scripts/rank_eeg_features.py
```

Train the EEG-only model:

```bash
.venv/bin/python scripts/train_model.py
```

Train the multimodal model:

```bash
.venv/bin/python scripts/train_multimodal_model.py --dataset data/ds007955
```

Run the original-vs-selected condition test:

```bash
.venv/bin/python scripts/test_hypothesis.py
```

Run the semantic transition analysis:

```bash
.venv/bin/python scripts/test_semantic_transitions.py
```

## Repository Layout

```text
config/
  selected_eeg_features.txt

docs/
  eeg_preprocessing.md
  eeg_feature_ranking.md
  model_training.md
  multimodal_model_training.md
  hypothesis_testing.md
  semantic_transition_analysis.md

scripts/
  preprocess_eeg.py
  rank_eeg_features.py
  train_model.py
  train_multimodal_model.py
  test_hypothesis.py
  test_semantic_transitions.py
```

Generated outputs are written to `outputs/` and are not tracked in git.
