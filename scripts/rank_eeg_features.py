#!/usr/bin/env python3
"""Rank EEG features for the NewWord prediction target."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


NON_FEATURE_COLUMNS = {
    "subject",
    "event_index",
    "onset",
    "word",
    "new_word",
    "epoch_valid",
    "epoch_invalid_reason",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("outputs/eeg_preprocessing/eeg_epoch_features.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/feature_ranking"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--exclude-artifact-flagged", action="store_true")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_feature_table(path: Path, exclude_artifact_flagged: bool) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[df["epoch_valid"] == 1].copy()
    if exclude_artifact_flagged:
        df = df[df["epoch_artifact_flag"] == 0].copy()
    if df["new_word"].nunique() != 2:
        raise ValueError("Feature table must contain both NewWord classes after filtering.")
    return df.reset_index(drop=True)


def candidate_feature_columns(df: pd.DataFrame) -> list[str]:
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    features = []
    for col in numeric:
        if col in NON_FEATURE_COLUMNS:
            continue
        if col.startswith("eeg_") or col.startswith("epoch_global_"):
            features.append(col)
    return features


def absolute_cohens_d(values: np.ndarray, labels: np.ndarray) -> float:
    group0 = values[labels == 0]
    group1 = values[labels == 1]
    if len(group0) < 2 or len(group1) < 2:
        return np.nan
    pooled = math.sqrt(((len(group0) - 1) * np.nanvar(group0) + (len(group1) - 1) * np.nanvar(group1)) / (len(group0) + len(group1) - 2))
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float(abs((np.nanmean(group1) - np.nanmean(group0)) / pooled))


def point_biserial_abs(values: np.ndarray, labels: np.ndarray) -> float:
    mask = np.isfinite(values)
    if mask.sum() < 3 or len(np.unique(labels[mask])) != 2:
        return np.nan
    if np.nanstd(values[mask]) == 0:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coefficient, _ = pointbiserialr(labels[mask], values[mask])
    return float(abs(coefficient)) if np.isfinite(coefficient) else np.nan


def choose_cv(labels: np.ndarray, groups: np.ndarray, random_state: int):
    class_counts = pd.Series(labels).value_counts()
    n_splits = int(min(5, class_counts.min(), len(np.unique(groups))))
    if n_splits >= 2 and len(np.unique(groups)) >= n_splits:
        return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state), groups

    n_splits = int(min(5, class_counts.min()))
    if n_splits >= 2:
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state), None

    raise ValueError("Not enough samples per class to rank features with cross-validation.")


def single_feature_cv(feature: pd.Series, labels: np.ndarray, groups: np.ndarray, random_state: int) -> tuple[float, float]:
    X = feature.to_numpy(dtype=float).reshape(-1, 1)
    cv, cv_groups = choose_cv(labels, groups, random_state)
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=random_state)),
        ]
    )

    predictions = np.zeros_like(labels, dtype=int)
    probabilities = np.full(labels.shape, np.nan, dtype=float)
    splitter = cv.split(X, labels, groups=cv_groups) if cv_groups is not None else cv.split(X, labels)

    for train_index, test_index in splitter:
        if len(np.unique(labels[train_index])) < 2:
            continue
        model.fit(X[train_index], labels[train_index])
        predictions[test_index] = model.predict(X[test_index])
        probabilities[test_index] = model.predict_proba(X[test_index])[:, 1]

    balanced_accuracy = float(balanced_accuracy_score(labels, predictions))
    if np.isfinite(probabilities).all() and len(np.unique(labels)) == 2:
        auc = float(roc_auc_score(labels, probabilities))
    else:
        auc = np.nan
    return balanced_accuracy, auc


def minmax(values: pd.Series) -> pd.Series:
    clean = values.replace([np.inf, -np.inf], np.nan)
    if clean.notna().sum() == 0:
        return pd.Series(np.zeros(len(values)), index=values.index)
    filled = clean.fillna(clean.median())
    scaler = MinMaxScaler()
    return pd.Series(scaler.fit_transform(filled.to_numpy().reshape(-1, 1)).ravel(), index=values.index)


def rank_features(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    labels = df["new_word"].astype(int).to_numpy()
    groups = df["subject"].astype(str).to_numpy()
    feature_columns = candidate_feature_columns(df)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    imputed = SimpleImputer(strategy="median").fit_transform(X)
    mutual_information = mutual_info_classif(imputed, labels, random_state=random_state)

    rows = []
    for index, feature in enumerate(feature_columns):
        values = X[feature].to_numpy(dtype=float)
        balanced_accuracy, auc = single_feature_cv(X[feature], labels, groups, random_state)
        rows.append(
            {
                "feature": feature,
                "coverage": float(np.isfinite(values).mean()),
                "mutual_information": float(mutual_information[index]),
                "abs_cohens_d": absolute_cohens_d(values, labels),
                "abs_point_biserial_r": point_biserial_abs(values, labels),
                "single_feature_balanced_accuracy": balanced_accuracy,
                "single_feature_auc": auc,
            }
        )

    ranked = pd.DataFrame(rows)
    ranked["score"] = (
        0.40 * minmax(ranked["single_feature_balanced_accuracy"])
        + 0.25 * minmax(ranked["single_feature_auc"])
        + 0.20 * minmax(ranked["mutual_information"])
        + 0.10 * minmax(ranked["abs_cohens_d"])
        + 0.05 * minmax(ranked["abs_point_biserial_r"])
    )
    return ranked.sort_values(["score", "single_feature_balanced_accuracy", "mutual_information"], ascending=False).reset_index(drop=True)


def write_markdown_report(ranked: pd.DataFrame, df: pd.DataFrame, output_dir: Path, top_k: int, exclude_artifact_flagged: bool) -> None:
    top = ranked.head(top_k)
    lines = [
        "# EEG Feature Ranking",
        "",
        "Target: `new_word`.",
        "",
        f"Rows used: {len(df)}",
        f"Subjects used: {df['subject'].nunique()}",
        f"Artifact-flagged epochs excluded: {exclude_artifact_flagged}",
        "",
        "The score combines single-feature cross-validated balanced accuracy, single-feature AUC, mutual information, Cohen's d, and point-biserial correlation.",
        "",
        "## Top Features",
        "",
        "| rank | feature | score | balanced accuracy | AUC | mutual information | Cohen's d |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        lines.append(
            f"| {rank} | `{row.feature}` | {row.score:.3f} | {row.single_feature_balanced_accuracy:.3f} | "
            f"{row.single_feature_auc:.3f} | {row.mutual_information:.3f} | {row.abs_cohens_d:.3f} |"
        )
    lines.append("")
    lines.append("These rankings are a screening step. Final model performance should be estimated in the later modeling workflow.")
    (output_dir / "feature_ranking_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_feature_table(args.features, args.exclude_artifact_flagged)
    ranked = rank_features(df, args.random_state)
    top = ranked.head(args.top_k)

    ranked.to_csv(args.output_dir / "ranked_eeg_features.csv", index=False)
    top.to_csv(args.output_dir / "top_10_eeg_features.csv", index=False)
    write_markdown_report(ranked, df, args.output_dir, args.top_k, args.exclude_artifact_flagged)

    summary = {
        "features_input": str(args.features),
        "rows_used": int(len(df)),
        "subjects_used": int(df["subject"].nunique()),
        "artifact_flagged_epochs_excluded": bool(args.exclude_artifact_flagged),
        "candidate_features": int(len(ranked)),
        "top_features": top["feature"].tolist(),
    }
    with (args.output_dir / "feature_ranking_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Rows used: {summary['rows_used']}")
    print(f"Candidate features: {summary['candidate_features']}")
    print("\nTop 10 EEG features")
    for rank, row in enumerate(top.itertuples(index=False), start=1):
        print(f"{rank:2d}. {row.feature} | score={row.score:.3f} | bal_acc={row.single_feature_balanced_accuracy:.3f} | auc={row.single_feature_auc:.3f}")


if __name__ == "__main__":
    main()
