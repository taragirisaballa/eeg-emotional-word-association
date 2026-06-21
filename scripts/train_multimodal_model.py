#!/usr/bin/env python3
"""Train models using selected EEG features and autonomic physiology features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PHYSIO_SIGNALS = {
    "eda": "EDA",
    "bvp": "BVP",
    "ibi": "IBI",
    "temp": "TEMP",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/ds007955"))
    parser.add_argument("--eeg-features", type=Path, default=Path("outputs/eeg_preprocessing/eeg_epoch_features.csv"))
    parser.add_argument("--selected-eeg-features", type=Path, default=Path("config/selected_eeg_features.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/multimodal_model"))
    parser.add_argument("--tmin", type=float, default=-3.0)
    parser.add_argument("--tmax", type=float, default=0.0)
    parser.add_argument("--exclude-artifact-flagged", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_selected_features(path: Path) -> list[str]:
    features = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not features:
        raise ValueError(f"No selected EEG features found in {path}")
    return features


def read_signal(dataset: Path, subject: str, signal: str) -> pd.DataFrame | None:
    path = dataset / "sourcedata" / "physio" / f"{subject}_task-wordassociation_{signal}.tsv"
    if not path.exists():
        return None
    return pd.read_csv(path, sep="\t")


def summarize_signal(signal_df: pd.DataFrame | None, onset: float, tmin: float, tmax: float, value_column: str, prefix: str) -> dict[str, float]:
    if signal_df is None or value_column not in signal_df.columns:
        return {
            f"{prefix}_available": 0.0,
            f"{prefix}_n": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_slope": np.nan,
        }

    window = signal_df[(signal_df["onset"] >= onset + tmin) & (signal_df["onset"] <= onset + tmax)]
    values = window[value_column].dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return {
            f"{prefix}_available": 1.0,
            f"{prefix}_n": 0.0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_range": np.nan,
            f"{prefix}_slope": np.nan,
        }

    return {
        f"{prefix}_available": 1.0,
        f"{prefix}_n": float(len(values)),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_slope": float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0.0,
    }


def extract_subject_physio(dataset: Path, subject: str, subject_events: pd.DataFrame, tmin: float, tmax: float) -> pd.DataFrame:
    signals = {name: read_signal(dataset, subject, name) for name in PHYSIO_SIGNALS}
    rows = []
    for _, event in subject_events.iterrows():
        row = {
            "subject": subject,
            "event_index": int(event["event_index"]),
        }
        for signal_name, value_column in PHYSIO_SIGNALS.items():
            row.update(
                summarize_signal(
                    signals[signal_name],
                    float(event["onset"]),
                    tmin,
                    tmax,
                    value_column,
                    f"physio_{signal_name}",
                )
            )
        rows.append(row)
    return pd.DataFrame(rows)


def build_multimodal_table(
    dataset: Path,
    eeg_features_path: Path,
    selected_eeg_features: list[str],
    tmin: float,
    tmax: float,
    exclude_artifact_flagged: bool,
) -> tuple[pd.DataFrame, list[str]]:
    eeg = pd.read_csv(eeg_features_path)
    eeg = eeg[eeg["epoch_valid"] == 1].copy()
    if exclude_artifact_flagged:
        eeg = eeg[eeg["epoch_artifact_flag"] == 0].copy()

    missing = [feature for feature in selected_eeg_features if feature not in eeg.columns]
    if missing:
        raise KeyError(f"Selected EEG features missing from table: {', '.join(missing)}")

    physio_tables = []
    for subject, subject_events in eeg.groupby("subject", sort=True):
        physio_tables.append(extract_subject_physio(dataset, subject, subject_events, tmin, tmax))
    physio = pd.concat(physio_tables, ignore_index=True)

    metadata = [
        "subject",
        "event_index",
        "onset",
        "word",
        "correlation",
        "new_word",
        "epoch_artifact_flag",
        "epoch_artifact_fraction",
    ]
    table = eeg[metadata + selected_eeg_features].merge(physio, on=["subject", "event_index"], how="left")
    physio_features = [column for column in table.columns if column.startswith("physio_")]
    model_features = selected_eeg_features + physio_features

    if table["new_word"].nunique() != 2:
        raise ValueError("Modeling table must contain both NewWord classes after filtering.")
    return table.reset_index(drop=True), model_features


def choose_cv(labels: np.ndarray, groups: np.ndarray, random_state: int):
    class_counts = pd.Series(labels).value_counts()
    group_count = len(np.unique(groups))
    n_splits = int(min(5, class_counts.min(), group_count))
    if n_splits >= 2:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state), groups

    n_splits = int(min(5, class_counts.min()))
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), None

    raise ValueError("Not enough samples per class for cross-validation.")


def build_models(random_state: int) -> dict[str, Pipeline]:
    return {
        "gradient_boosting": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", GradientBoostingClassifier(random_state=random_state)),
            ]
        ),
        "logistic_regression": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=random_state)),
            ]
        ),
    }


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: np.ndarray, groups: np.ndarray, random_state: int) -> dict[str, object]:
    cv, cv_groups = choose_cv(y, groups, random_state)
    predictions = cross_val_predict(model, X, y, cv=cv, groups=cv_groups, method="predict")
    probabilities = cross_val_predict(model, X, y, cv=cv, groups=cv_groups, method="predict_proba")[:, 1]
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "f1": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions, zero_division=0, output_dict=True),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def feature_importance(model: Pipeline, X: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    model.fit(X, y)
    classifier = model.named_steps["classifier"]
    if hasattr(classifier, "feature_importances_"):
        importance = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importance = np.abs(classifier.coef_[0])
    else:
        importance = np.full(X.shape[1], np.nan)
    return pd.DataFrame({"feature": X.columns, "importance": importance}).sort_values("importance", ascending=False)


def write_report(
    results: pd.DataFrame,
    output_dir: Path,
    model_features: list[str],
    row_count: int,
    subject_count: int,
    class_counts: dict[str, int],
    best_model: str,
) -> None:
    lines = [
        "# Multimodal Model Training",
        "",
        "Target: `new_word`.",
        "",
        f"Rows used: {row_count}",
        f"Subjects used: {subject_count}",
        f"Class counts: {class_counts}",
        f"Model features: {len(model_features)}",
        "",
        "## Cross-Validated Results",
        "",
        "| model | accuracy | balanced accuracy | F1 | ROC AUC |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        lines.append(f"| {row.model} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.f1:.3f} | {row.roc_auc:.3f} |")
    lines.extend(
        [
            "",
            f"Best model by balanced accuracy: `{best_model}`.",
            "",
            "The feature table combines selected EEG features, pre-response physiology summaries, metadata, and labels. Word text and semantic correlation are retained as metadata, not model inputs.",
        ]
    )
    (output_dir / "multimodal_model_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_eeg_features = read_selected_features(args.selected_eeg_features)
    table, model_features = build_multimodal_table(
        args.dataset,
        args.eeg_features,
        selected_eeg_features,
        args.tmin,
        args.tmax,
        args.exclude_artifact_flagged,
    )
    table.to_csv(args.output_dir / "multimodal_modeling_dataset.csv", index=False)

    X = table[model_features].apply(pd.to_numeric, errors="coerce")
    y = table["new_word"].astype(int).to_numpy()
    groups = table["subject"].astype(str).to_numpy()
    models = build_models(args.random_state)

    result_rows = []
    prediction_table = table[["subject", "event_index", "onset", "word", "correlation", "new_word", "epoch_artifact_flag"]].copy()
    reports: dict[str, object] = {}

    for name, model in models.items():
        metrics = evaluate_model(model, X, y, groups, args.random_state)
        result_rows.append(
            {
                "model": name,
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
            }
        )
        prediction_table[f"{name}_prediction"] = metrics["predictions"]
        prediction_table[f"{name}_probability_new_word"] = metrics["probabilities"]
        reports[name] = {
            key: value
            for key, value in metrics.items()
            if key not in {"predictions", "probabilities"}
        }

    results = pd.DataFrame(result_rows).sort_values("balanced_accuracy", ascending=False)
    best_model = str(results.iloc[0]["model"])
    importance = feature_importance(models[best_model], X, y)

    results.to_csv(args.output_dir / "multimodal_model_metrics.csv", index=False)
    prediction_table.to_csv(args.output_dir / "multimodal_model_predictions.csv", index=False)
    importance.to_csv(args.output_dir / "multimodal_feature_importance.csv", index=False)

    summary = {
        "dataset": str(args.dataset),
        "eeg_features_input": str(args.eeg_features),
        "selected_eeg_features_input": str(args.selected_eeg_features),
        "artifact_flagged_epochs_excluded": bool(args.exclude_artifact_flagged),
        "rows_used": int(len(table)),
        "subjects_used": int(table["subject"].nunique()),
        "class_counts": {str(k): int(v) for k, v in table["new_word"].value_counts().sort_index().items()},
        "model_features": model_features,
        "best_model": best_model,
        "results": reports,
    }
    with (args.output_dir / "multimodal_model_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    write_report(results, args.output_dir, model_features, len(table), table["subject"].nunique(), summary["class_counts"], best_model)

    print(f"Rows used: {len(table)}")
    print(f"Subjects used: {table['subject'].nunique()}")
    print(f"Model features: {len(model_features)}")
    print(f"Best model: {best_model}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
