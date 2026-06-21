#!/usr/bin/env python3
"""Preprocess EEG recordings and extract response-locked features."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy, kurtosis, skew


BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}
ASYMMETRY_PAIRS = [
    ("FP1", "FP2"),
    ("Fp1", "Fp2"),
    ("F3", "F4"),
    ("C3", "C4"),
    ("P7", "P8"),
    ("T5", "T6"),
    ("O1", "O2"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/ds007955"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eeg_preprocessing"))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--tmin", type=float, default=-3.0)
    parser.add_argument("--tmax", type=float, default=0.0)
    parser.add_argument("--artifact-threshold-uv", type=float, default=100.0)
    parser.add_argument("--make-plots", action="store_true")
    return parser.parse_args()


def discover_subjects(dataset: Path, requested: list[str] | None) -> list[str]:
    available = sorted(path.name for path in dataset.glob("sub-*") if (path / "eeg").exists())
    if requested is None:
        return available
    missing = sorted(set(requested) - set(available))
    if missing:
        raise FileNotFoundError(f"Requested subjects not found: {', '.join(missing)}")
    return requested


def event_path(dataset: Path, subject: str) -> Path:
    return dataset / subject / "eeg" / f"{subject}_task-wordassociation_events.tsv"


def eeg_path(dataset: Path, subject: str) -> Path:
    return dataset / subject / "eeg" / f"{subject}_task-wordassociation_eeg.set"


def channels_path(dataset: Path, subject: str) -> Path:
    return dataset / subject / "eeg" / f"{subject}_task-wordassociation_channels.tsv"


def load_events(dataset: Path, subject: str) -> pd.DataFrame:
    events = pd.read_csv(event_path(dataset, subject), sep="\t")
    events = events[events["NewWord"].notna()].copy()
    events["NewWord"] = events["NewWord"].astype(int)
    return events.reset_index(drop=True)


def load_sidecar_channels(dataset: Path, subject: str) -> list[str]:
    path = channels_path(dataset, subject)
    if not path.exists():
        return []
    channels = pd.read_csv(path, sep="\t")
    return channels["name"].astype(str).tolist()


def load_raw(set_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage("standard_1020", on_missing="ignore")
    return raw


def preprocess_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    clean = raw.copy()
    clean.set_eeg_reference("average", projection=False, verbose="ERROR")
    clean.notch_filter(freqs=[60], verbose="ERROR")
    clean.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
    return clean


def bandpower(values: np.ndarray, sfreq: float) -> dict[str, float]:
    nperseg = min(len(values), int(sfreq * 2))
    freqs, psd = welch(values, fs=sfreq, nperseg=max(nperseg, 8))
    total_mask = (freqs >= 1.0) & (freqs <= 40.0)
    total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask])) + 1e-18

    features: dict[str, float] = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs < high)
        power = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else np.nan
        features[f"{band}_power_log10"] = math.log10(power + 1e-18)
        features[f"{band}_relative_power"] = power / total_power
    return features


def hjorth(values: np.ndarray) -> tuple[float, float, float]:
    diff1 = np.diff(values)
    diff2 = np.diff(diff1)
    var0 = np.var(values) + 1e-18
    var1 = np.var(diff1) + 1e-18
    var2 = np.var(diff2) + 1e-18
    activity = var0
    mobility = math.sqrt(var1 / var0)
    complexity = math.sqrt(var2 / var1) / mobility
    return activity, mobility, complexity


def shannon_entropy(values: np.ndarray) -> float:
    counts, _ = np.histogram(values, bins=20)
    probabilities = counts[counts > 0] / max(counts.sum(), 1)
    return float(entropy(probabilities))


def epoch_bounds(onset: float, sfreq: float, tmin: float, tmax: float) -> tuple[int, int]:
    start = int(round((onset + tmin) * sfreq))
    stop = int(round((onset + tmax) * sfreq))
    return start, stop


def extract_epoch_features(
    raw: mne.io.BaseRaw,
    events: pd.DataFrame,
    subject: str,
    tmin: float,
    tmax: float,
    artifact_threshold_uv: float,
    sidecar_channels: list[str],
) -> tuple[pd.DataFrame, dict[str, object]]:
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data(picks=raw.ch_names)
    threshold_volts = artifact_threshold_uv * 1e-6

    rows: list[dict[str, object]] = []
    valid_epochs = 0
    invalid_epochs = 0
    artifact_epochs = 0

    for event_index, event in events.iterrows():
        start, stop = epoch_bounds(float(event["onset"]), sfreq, tmin, tmax)
        row: dict[str, object] = {
            "subject": subject,
            "event_index": int(event_index),
            "onset": float(event["onset"]),
            "word": event.get("Word"),
            "correlation": float(event["Correlation"]) if pd.notna(event["Correlation"]) else np.nan,
            "new_word": int(event["NewWord"]),
        }

        if start < 0 or stop <= start or stop > data.shape[1]:
            row["epoch_valid"] = 0
            row["epoch_artifact_flag"] = 0
            row["epoch_invalid_reason"] = "outside_recording_bounds"
            rows.append(row)
            invalid_epochs += 1
            continue

        epoch = data[:, start:stop]
        artifact_fraction = float(np.mean(np.abs(epoch) > threshold_volts))
        artifact_flag = int(artifact_fraction > 0.05)
        valid_epochs += 1
        artifact_epochs += artifact_flag

        row["epoch_valid"] = 1
        row["epoch_artifact_flag"] = artifact_flag
        row["epoch_invalid_reason"] = ""
        row["epoch_artifact_fraction"] = artifact_fraction
        row["epoch_global_rms_uv"] = float(np.sqrt(np.mean(epoch**2)) * 1e6)
        row["epoch_global_peak_to_peak_uv"] = float(np.ptp(epoch) * 1e6)

        for channel_index, channel in enumerate(raw.ch_names):
            values = epoch[channel_index]
            prefix = f"eeg_{channel}"
            for feature_name, value in bandpower(values, sfreq).items():
                row[f"{prefix}_{feature_name}"] = value

            row[f"{prefix}_mean_uv"] = float(np.mean(values) * 1e6)
            row[f"{prefix}_std_uv"] = float(np.std(values) * 1e6)
            row[f"{prefix}_peak_to_peak_uv"] = float(np.ptp(values) * 1e6)
            row[f"{prefix}_skew"] = float(skew(values, bias=False))
            row[f"{prefix}_kurtosis"] = float(kurtosis(values, bias=False))
            row[f"{prefix}_entropy"] = shannon_entropy(values)
            activity, mobility, complexity = hjorth(values)
            row[f"{prefix}_hjorth_activity"] = activity
            row[f"{prefix}_hjorth_mobility"] = mobility
            row[f"{prefix}_hjorth_complexity"] = complexity

        for left, right in ASYMMETRY_PAIRS:
            if left in raw.ch_names and right in raw.ch_names:
                for band in ("theta", "alpha", "beta"):
                    right_key = f"eeg_{right}_{band}_power_log10"
                    left_key = f"eeg_{left}_{band}_power_log10"
                    row[f"eeg_asymmetry_{left}_{right}_{band}"] = row[right_key] - row[left_key]

        rows.append(row)

    qc = {
        "subject": subject,
        "sampling_frequency_hz": sfreq,
        "duration_seconds": float(raw.times[-1]) if len(raw.times) else 0.0,
        "recording_channels": raw.ch_names,
        "sidecar_channels": sidecar_channels,
        "channel_labels_match_sidecar": raw.ch_names == sidecar_channels,
        "n_channels": len(raw.ch_names),
        "n_events": int(len(events)),
        "n_valid_epochs": int(valid_epochs),
        "n_invalid_epochs": int(invalid_epochs),
        "n_artifact_flagged_valid_epochs": int(artifact_epochs),
        "artifact_threshold_uv": float(artifact_threshold_uv),
        "epoch_window_seconds": [float(tmin), float(tmax)],
    }
    return pd.DataFrame(rows), qc


def channel_summary(raw: mne.io.BaseRaw, subject: str) -> pd.DataFrame:
    data = raw.get_data(picks=raw.ch_names)
    rows = []
    for index, channel in enumerate(raw.ch_names):
        values = data[index]
        rows.append(
            {
                "subject": subject,
                "channel": channel,
                "mean_uv": float(np.mean(values) * 1e6),
                "std_uv": float(np.std(values) * 1e6),
                "peak_to_peak_uv": float(np.ptp(values) * 1e6),
                "rms_uv": float(np.sqrt(np.mean(values**2)) * 1e6),
            }
        )
    return pd.DataFrame(rows)


def plot_psd_comparison(raw: mne.io.BaseRaw, clean: mne.io.BaseRaw, subject: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    raw.compute_psd(fmax=70, verbose="ERROR").plot(axes=axes[0], show=False, average=True)
    clean.compute_psd(fmax=70, verbose="ERROR").plot(axes=axes[1], show=False, average=True)
    axes[0].set_title(f"{subject} raw PSD")
    axes[1].set_title(f"{subject} filtered PSD")
    fig.tight_layout()
    fig.savefig(output_dir / f"{subject}_psd_raw_vs_filtered.png", dpi=150)
    plt.close(fig)


def process_subject(
    dataset: Path,
    subject: str,
    output_dir: Path,
    tmin: float,
    tmax: float,
    artifact_threshold_uv: float,
    make_plots: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    events = load_events(dataset, subject)
    sidecar_channels = load_sidecar_channels(dataset, subject)
    raw = load_raw(eeg_path(dataset, subject))
    clean = preprocess_raw(raw)
    features, qc = extract_epoch_features(clean, events, subject, tmin, tmax, artifact_threshold_uv, sidecar_channels)
    channels = channel_summary(clean, subject)

    if make_plots:
        plot_psd_comparison(raw, clean, subject, output_dir / "plots")

    return features, channels, qc


def main() -> None:
    args = parse_args()
    subjects = discover_subjects(args.dataset, args.subjects)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_features = []
    all_channels = []
    qc_rows = []

    for subject in subjects:
        print(f"Processing {subject}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            features, channels, qc = process_subject(
                args.dataset,
                subject,
                args.output_dir,
                args.tmin,
                args.tmax,
                args.artifact_threshold_uv,
                args.make_plots,
            )
        all_features.append(features)
        all_channels.append(channels)
        qc_rows.append(qc)

    feature_table = pd.concat(all_features, ignore_index=True)
    channel_table = pd.concat(all_channels, ignore_index=True)
    qc_table = pd.DataFrame(qc_rows)

    feature_table.to_csv(args.output_dir / "eeg_epoch_features.csv", index=False)
    channel_table.to_csv(args.output_dir / "eeg_channel_summary.csv", index=False)
    qc_table.to_csv(args.output_dir / "eeg_preprocessing_qc.csv", index=False)
    with (args.output_dir / "eeg_preprocessing_qc.json").open("w") as f:
        json.dump(qc_rows, f, indent=2)

    print(f"\nSubjects processed: {len(subjects)}")
    print(f"Feature rows: {len(feature_table)}")
    print(f"Valid epochs: {int(feature_table['epoch_valid'].sum())}")
    print(f"Flagged epochs: {int(feature_table['epoch_artifact_flag'].sum())}")
    print(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
