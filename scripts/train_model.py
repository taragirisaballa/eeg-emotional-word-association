#!/usr/bin/env python3
"""Train a classifier using the selected EEG features."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/eeg_preprocessing/eeg_epoch_features.csv"))
    parser.add_argument("--selected-features", type=Path, default=Path("config/selected_eeg_features.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/model_training"))
    parser.add_argument("--exclude-artifact-flagged", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_selected_features(path: Path) -> list[str]:
    features = [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]
    if not features:
        raise ValueError(f"No selected features found in {path}")
    return features


def load_modeling_table(features_path: Path, selected_features: list[str], exclude_artifact_flagged: bool) -> pd.DataFrame:
    df = pd.read_csv(features_path)
    df = df[df["epoch_valid"] == 1].copy()
    if exclude_artifact_flagged:
        df = df[df["epoch_artifact_flag"] == 0].copy()

    missing = [feature for feature in selected_features if feature not in df.columns]
    if missing:
        raise KeyError(f"Selected features missing from table: {', '.join(missing)}")
    if df["new_word"].nunique() != 2:
        raise ValueError("Modeling table must contain both NewWord classes after filtering.")

    metadata = ["subject", "event_index", "onset", "word", "new_word", "epoch_artifact_flag"]
    return df[metadata + selected_features].reset_index(drop=True)


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


def write_report(results: pd.DataFrame, output_dir: Path, selected_features: list[str], row_count: int, class_counts: dict[str, int]) -> None:
    best = results.sort_values("balanced_accuracy", ascending=False).iloc[0]
    lines = [
        "# Model Training",
        "",
        "Target: `new_word`.",
        "",
        f"Rows used: {row_count}",
        f"Class counts: {class_counts}",
        "",
        "Selected EEG features:",
        "",
    ]
    lines.extend(f"{index}. `{feature}`" for index, feature in enumerate(selected_features, start=1))
    lines.extend(
        [
            "",
            "## Cross-Validated Results",
            "",
            "| model | accuracy | balanced accuracy | F1 | ROC AUC |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in results.itertuples(index=False):
        lines.append(f"| {row.model} | {row.accuracy:.3f} | {row.balanced_accuracy:.3f} | {row.f1:.3f} | {row.roc_auc:.3f} |")
    lines.extend(
        [
            "",
            f"Best model by balanced accuracy: `{best.model}`.",
            "",
            "Balanced accuracy is emphasized because the label classes are not perfectly even.",
        ]
    )
    (output_dir / "model_training_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_features = read_selected_features(args.selected_features)
    table = load_modeling_table(args.features, selected_features, args.exclude_artifact_flagged)
    table.to_csv(args.output_dir / "modeling_dataset.csv", index=False)

    X = table[selected_features].apply(pd.to_numeric, errors="coerce")
    y = table["new_word"].astype(int).to_numpy()
    groups = table["subject"].astype(str).to_numpy()
    models = build_models(args.random_state)

    result_rows = []
    prediction_table = table[["subject", "event_index", "onset", "word", "new_word", "epoch_artifact_flag"]].copy()
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
    best_model_name = results.iloc[0]["model"]
    importance = feature_importance(models[best_model_name], X, y)

    results.to_csv(args.output_dir / "model_metrics.csv", index=False)
    prediction_table.to_csv(args.output_dir / "model_predictions.csv", index=False)
    importance.to_csv(args.output_dir / "feature_importance.csv", index=False)

    summary = {
        "features_input": str(args.features),
        "selected_features_input": str(args.selected_features),
        "artifact_flagged_epochs_excluded": bool(args.exclude_artifact_flagged),
        "rows_used": int(len(table)),
        "subjects_used": int(table["subject"].nunique()),
        "class_counts": {str(k): int(v) for k, v in table["new_word"].value_counts().sort_index().items()},
        "best_model": str(best_model_name),
        "results": reports,
    }
    with (args.output_dir / "model_training_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    write_report(results, args.output_dir, selected_features, len(table), summary["class_counts"])

    print(f"Rows used: {len(table)}")
    print(f"Subjects used: {table['subject'].nunique()}")
    print(f"Best model: {best_model_name}")
    print()
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
