#!/usr/bin/env python3
"""Preprocess EEG/physiology features and train a NewWord classifier."""

from __future__ import annotations

import argparse
import json
import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache") / "matplotlib"))

import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import entropy, skew, kurtosis
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


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
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--subjects", nargs="*", default=None)
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--tmin", type=float, default=-3.0)
    parser.add_argument("--tmax", type=float, default=0.0)
    parser.add_argument("--min-classes-per-fold", type=int, default=2)
    return parser.parse_args()


def load_raw_eeg(set_path: Path) -> mne.io.BaseRaw:
    raw = mne.io.read_raw_eeglab(set_path, preload=True, verbose="ERROR")
    raw.set_channel_types({ch: "eeg" for ch in raw.ch_names})
    raw.set_montage("standard_1020", on_missing="ignore")
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")

    # Denoising path: remove line noise before the task-relevant 1-40 Hz band.
    raw.notch_filter(freqs=[60], verbose="ERROR")
    raw.filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
    return raw


def bandpower(values: np.ndarray, sfreq: float) -> dict[str, float]:
    nperseg = min(len(values), int(sfreq * 2))
    freqs, psd = welch(values, fs=sfreq, nperseg=max(nperseg, 8))
    out: dict[str, float] = {}
    total_mask = (freqs >= 1.0) & (freqs <= 40.0)
    total_power = float(np.trapezoid(psd[total_mask], freqs[total_mask])) + 1e-12
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        power = float(np.trapezoid(psd[mask], freqs[mask])) if mask.any() else np.nan
        out[f"{name}_abs"] = math.log10(power + 1e-12)
        out[f"{name}_rel"] = power / total_power
    return out


def hjorth(values: np.ndarray) -> tuple[float, float, float]:
    diff1 = np.diff(values)
    diff2 = np.diff(diff1)
    var0 = np.var(values) + 1e-12
    var1 = np.var(diff1) + 1e-12
    var2 = np.var(diff2) + 1e-12
    activity = var0
    mobility = math.sqrt(var1 / var0)
    complexity = math.sqrt(var2 / var1) / mobility
    return activity, mobility, complexity


def shannon_entropy(values: np.ndarray, bins: int = 20) -> float:
    counts, _ = np.histogram(values, bins=bins)
    probs = counts[counts > 0] / max(counts.sum(), 1)
    return float(entropy(probs))


def extract_eeg_features(raw: mne.io.BaseRaw, events: pd.DataFrame, tmin: float, tmax: float) -> pd.DataFrame:
    sfreq = float(raw.info["sfreq"])
    data = raw.get_data(picks=raw.ch_names)
    rows: list[dict[str, float]] = []

    for _, event in events.iterrows():
        onset = float(event["onset"])
        start = int(round((onset + tmin) * sfreq))
        stop = int(round((onset + tmax) * sfreq))
        row: dict[str, float] = {}

        if start < 0 or stop <= start or stop > data.shape[1]:
            row["eeg_valid"] = 0.0
            rows.append(row)
            continue

        epoch = data[:, start:stop]
        row["eeg_valid"] = 1.0
        row["eeg_artifact_fraction_abs100uv"] = float(np.mean(np.abs(epoch) > 100e-6))
        row["eeg_global_rms"] = float(np.sqrt(np.mean(epoch**2)))

        for channel_index, channel in enumerate(raw.ch_names):
            values = epoch[channel_index]
            channel_prefix = f"eeg_{channel}"
            for feature, value in bandpower(values, sfreq).items():
                row[f"{channel_prefix}_{feature}"] = value

            row[f"{channel_prefix}_mean"] = float(np.mean(values))
            row[f"{channel_prefix}_std"] = float(np.std(values))
            row[f"{channel_prefix}_ptp"] = float(np.ptp(values))
            row[f"{channel_prefix}_skew"] = float(skew(values, bias=False))
            row[f"{channel_prefix}_kurtosis"] = float(kurtosis(values, bias=False))
            row[f"{channel_prefix}_entropy"] = shannon_entropy(values)
            activity, mobility, complexity = hjorth(values)
            row[f"{channel_prefix}_hjorth_activity"] = activity
            row[f"{channel_prefix}_hjorth_mobility"] = mobility
            row[f"{channel_prefix}_hjorth_complexity"] = complexity

        for left, right in ASYMMETRY_PAIRS:
            if left in raw.ch_names and right in raw.ch_names:
                for band in ("alpha", "beta", "theta"):
                    l_key = f"eeg_{left}_{band}_abs"
                    r_key = f"eeg_{right}_{band}_abs"
                    row[f"eeg_asym_{left}_{right}_{band}"] = row.get(r_key, np.nan) - row.get(l_key, np.nan)

        rows.append(row)

    return pd.DataFrame(rows)


def read_physio_signal(dataset: Path, subject: str, signal: str) -> pd.DataFrame | None:
    path = dataset / "sourcedata" / "physio" / f"{subject}_task-wordassociation_{signal}.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def summarize_signal(signal_df: pd.DataFrame | None, onset: float, tmin: float, tmax: float, value_col: str, prefix: str) -> dict[str, float]:
    if signal_df is None or value_col not in signal_df:
        return {f"{prefix}_available": 0.0}

    mask = (signal_df["onset"] >= onset + tmin) & (signal_df["onset"] <= onset + tmax)
    values = signal_df.loc[mask, value_col].dropna().to_numpy(dtype=float)
    out = {f"{prefix}_available": 1.0, f"{prefix}_n": float(len(values))}
    if len(values) == 0:
        return out

    out.update(
        {
            f"{prefix}_mean": float(np.mean(values)),
            f"{prefix}_std": float(np.std(values)),
            f"{prefix}_min": float(np.min(values)),
            f"{prefix}_max": float(np.max(values)),
            f"{prefix}_range": float(np.ptp(values)),
            f"{prefix}_slope": float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0,
        }
    )
    return out


def extract_physio_features(dataset: Path, subject: str, events: pd.DataFrame, tmin: float, tmax: float) -> pd.DataFrame:
    signals = {
        "eda": ("EDA", read_physio_signal(dataset, subject, "eda")),
        "bvp": ("BVP", read_physio_signal(dataset, subject, "bvp")),
        "temp": ("TEMP", read_physio_signal(dataset, subject, "temp")),
        "ibi": ("IBI", read_physio_signal(dataset, subject, "ibi")),
    }

    rows = []
    for _, event in events.iterrows():
        onset = float(event["onset"])
        row = {}
        for prefix, (column, signal_df) in signals.items():
            row.update(summarize_signal(signal_df, onset, tmin, tmax, column, f"physio_{prefix}"))
        rows.append(row)
    return pd.DataFrame(rows)


def load_subject_table(dataset: Path, subject: str, tmin: float, tmax: float) -> pd.DataFrame:
    eeg_dir = dataset / subject / "eeg"
    set_path = eeg_dir / f"{subject}_task-wordassociation_eeg.set"
    events_path = eeg_dir / f"{subject}_task-wordassociation_events.tsv"
    events = pd.read_csv(events_path, sep="\t")
    events = events[events["NewWord"].notna()].reset_index(drop=True)

    raw = load_raw_eeg(set_path)
    eeg_features = extract_eeg_features(raw, events, tmin=tmin, tmax=tmax)
    physio_features = extract_physio_features(dataset, subject, events, tmin=tmin, tmax=tmax)

    table = pd.concat(
        [
            events[["onset", "duration", "Correlation", "NewWord"]].reset_index(drop=True),
            eeg_features.reset_index(drop=True),
            physio_features.reset_index(drop=True),
        ],
        axis=1,
    )
    table.insert(0, "subject", subject)
    table["label"] = table["NewWord"].astype(int)
    return table


def choose_cv(y: pd.Series, groups: pd.Series, min_classes_per_fold: int):
    group_counts = groups.nunique()
    class_counts = y.value_counts()
    if group_counts >= 3 and class_counts.min() >= min_classes_per_fold:
        return GroupKFold(n_splits=min(5, group_counts)), groups
    n_splits = int(min(5, class_counts.min()))
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42), None
    return None, None


def train_and_report(table: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    drop_cols = {"subject", "label", "NewWord", "Word", "trial_type"}
    feature_cols = [col for col in table.columns if col not in drop_cols]
    X = table[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = table["label"].astype(int)
    groups = table["subject"]

    selector = SelectKBest(mutual_info_classif, k=min(10, X.shape[1]))
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("select", selector),
            ("classifier", GradientBoostingClassifier(random_state=42)),
        ]
    )

    cv, cv_groups = choose_cv(y, groups, min_classes_per_fold=2)
    report: dict[str, object] = {
        "n_trials": int(len(table)),
        "n_features_before_selection": int(X.shape[1]),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "subjects": sorted(groups.unique().tolist()),
    }

    if cv is None:
        model.fit(X, y)
        y_pred = model.predict(X)
        y_score = model.predict_proba(X)[:, 1] if len(np.unique(y)) == 2 else None
        report["validation"] = "resubstitution_only_too_few_minority_trials"
    else:
        y_pred = cross_val_predict(model, X, y, cv=cv, groups=cv_groups, method="predict")
        y_score = cross_val_predict(model, X, y, cv=cv, groups=cv_groups, method="predict_proba")[:, 1]
        report["validation"] = type(cv).__name__

    report["accuracy"] = float(accuracy_score(y, y_pred))
    report["balanced_accuracy"] = float(balanced_accuracy_score(y, y_pred))
    report["f1"] = float(f1_score(y, y_pred, zero_division=0))
    if y_score is not None and len(np.unique(y)) == 2:
        report["roc_auc"] = float(roc_auc_score(y, y_score))
    report["classification_report"] = classification_report(y, y_pred, zero_division=0, output_dict=True)

    model.fit(X, y)
    selected_mask = model.named_steps["select"].get_support()
    selected_features = [col for col, keep in zip(feature_cols, selected_mask) if keep]
    importances = model.named_steps["classifier"].feature_importances_
    top_features = sorted(zip(selected_features, importances), key=lambda item: item[1], reverse=True)[:10]
    report["top_10_features"] = [{"feature": name, "importance": float(score)} for name, score in top_features]

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report["top_10_features"]).to_csv(output_dir / "top_10_features.csv", index=False)
    with (output_dir / "model_report.json").open("w") as f:
        json.dump(report, f, indent=2)
    return report


def main() -> None:
    args = parse_args()
    available_subjects = sorted(path.name for path in args.dataset.glob("sub-*") if (path / "eeg").exists())
    subjects = args.subjects if args.subjects else available_subjects
    missing = sorted(set(subjects) - set(available_subjects))
    if missing:
        raise FileNotFoundError(f"Requested subjects not found: {', '.join(missing)}")
    if not subjects:
        raise FileNotFoundError(f"No subjects found under {args.dataset}")

    tables = []
    for subject in subjects:
        print(f"Processing {subject}...")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tables.append(load_subject_table(args.dataset, subject, args.tmin, args.tmax))

    feature_table = pd.concat(tables, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_table.to_csv(args.output_dir / "features_with_labels.csv", index=False)

    if args.skip_model:
        print(f"\nWrote {len(feature_table)} rows to {args.output_dir / 'features_with_labels.csv'}")
        return

    report = train_and_report(feature_table, args.output_dir)
    print("\nModel report")
    print(f"Trials: {report['n_trials']}")
    print(f"Class counts: {report['class_counts']}")
    print(f"Validation: {report['validation']}")
    print(f"Balanced accuracy: {report['balanced_accuracy']:.3f}")
    print(f"F1: {report['f1']:.3f}")
    if "roc_auc" in report:
        print(f"ROC AUC: {report['roc_auc']:.3f}")
    print("\nTop 10 features")
    for i, item in enumerate(report["top_10_features"], start=1):
        print(f"{i:2d}. {item['feature']} ({item['importance']:.4f})")


if __name__ == "__main__":
    main()
