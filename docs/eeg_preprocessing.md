# EEG Preprocessing Notes

The preprocessing workflow is implemented in `scripts/preprocess_eeg.py`.

## Inputs

The workflow reads each subject's EEGLAB `.set` recording and matching BIDS event table from OpenNeuro dataset `ds007955`.

Each event is treated as a response-locked trial. The default epoch is the three seconds before the word response:

```text
-3.0 s to 0.0 s
```

## Filtering

The current preprocessing sequence is:

1. Load the EEGLAB recording with MNE.
2. Set all loaded channels to EEG type.
3. Attach the standard 10-20 montage when labels are recognized.
4. Apply average reference.
5. Apply a 60 Hz notch filter.
6. Apply a 1-40 Hz band-pass filter.

The notch filter targets line noise. The 1-40 Hz band-pass keeps the usual low-frequency EEG bands used in this first feature set while reducing slow drift and high-frequency contamination.

## Channel Labels

The `.set` recordings expose these labels:

```text
FP1, FP2, C3, C4, T5, T6, O1, O2
```

The BIDS sidecar lists:

```text
Fp1, Fp2, F3, F4, P7, P8, O1, O2
```

The preprocessing script preserves the labels from the recording instead of renaming channels silently. The sidecar labels and a boolean match check are written to `eeg_preprocessing_qc.csv` and `eeg_preprocessing_qc.json`.

## Artifact Flags

Epochs are not removed during this step. They are flagged.

The default artifact rule is:

```text
flag an epoch if more than 5% of samples exceed 100 uV in absolute amplitude
```

This is intentionally conservative. Later modeling can compare results with and without flagged epochs.

## Features

For each valid response-locked epoch, the workflow extracts:

1. Absolute and relative bandpower for delta, theta, alpha, beta, and low gamma.
2. Per-channel mean, standard deviation, peak-to-peak amplitude, skew, kurtosis, and entropy.
3. Hjorth activity, mobility, and complexity.
4. Hemispheric asymmetry features for channel pairs present in the recording.
5. Epoch-level RMS, peak-to-peak amplitude, and artifact fraction.

## Outputs

The workflow writes:

1. `outputs/eeg_preprocessing/eeg_epoch_features.csv`
2. `outputs/eeg_preprocessing/eeg_channel_summary.csv`
3. `outputs/eeg_preprocessing/eeg_preprocessing_qc.csv`
4. `outputs/eeg_preprocessing/eeg_preprocessing_qc.json`
5. Raw-vs-filtered PSD plots when `--make-plots` is used.
