#!/usr/bin/env python3
"""Test condition differences between selected and original word responses."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_1samp, ttest_ind, wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-table", type=Path, default=Path("outputs/multimodal_model/multimodal_modeling_dataset.csv"))
    parser.add_argument("--selected-eeg-features", type=Path, default=Path("config/selected_eeg_features.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/hypothesis_testing"))
    parser.add_argument("--min-condition-count", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_selected_features(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def cohens_d_independent(group0: pd.Series, group1: pd.Series) -> float:
    x0 = group0.dropna().to_numpy(dtype=float)
    x1 = group1.dropna().to_numpy(dtype=float)
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    pooled = math.sqrt(((len(x0) - 1) * np.var(x0, ddof=1) + (len(x1) - 1) * np.var(x1, ddof=1)) / (len(x0) + len(x1) - 2))
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def paired_effect_size(differences: pd.Series) -> float:
    values = differences.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.mean(values) / sd)


def bootstrap_ci(values: pd.Series, random_state: int = 42, n_bootstrap: int = 5000) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(n_bootstrap):
        sample = rng.choice(clean, size=len(clean), replace=True)
        estimates.append(np.mean(sample))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def condition_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["subject", "new_word"])
        .size()
        .unstack(fill_value=0)
        .rename(columns={0: "selected_count", 1: "original_count"})
        .reset_index()
    )


def subject_level_contrasts(df: pd.DataFrame, features: list[str], min_condition_count: int) -> pd.DataFrame:
    counts = condition_counts(df)
    eligible_subjects = counts[
        (counts.get("selected_count", 0) >= min_condition_count)
        & (counts.get("original_count", 0) >= min_condition_count)
    ]["subject"].tolist()
    eligible = df[df["subject"].isin(eligible_subjects)].copy()

    rows = []
    for feature in features:
        subject_means = (
            eligible.groupby(["subject", "new_word"])[feature]
            .mean()
            .unstack()
            .rename(columns={0: "selected_mean", 1: "original_mean"})
        )
        if "selected_mean" not in subject_means or "original_mean" not in subject_means:
            continue
        subject_means = subject_means.dropna(subset=["selected_mean", "original_mean"])
        subject_means["difference_original_minus_selected"] = subject_means["original_mean"] - subject_means["selected_mean"]

        differences = subject_means["difference_original_minus_selected"]
        if len(differences) >= 2:
            t_result = ttest_1samp(differences, popmean=0.0, nan_policy="omit")
            try:
                if np.nanstd(differences.to_numpy(dtype=float)) == 0:
                    wilcoxon_p = np.nan
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        w_result = wilcoxon(differences)
                    wilcoxon_p = float(w_result.pvalue)
            except ValueError:
                wilcoxon_p = np.nan
            ci_low, ci_high = bootstrap_ci(differences)
        else:
            t_result = None
            wilcoxon_p = np.nan
            ci_low, ci_high = np.nan, np.nan

        rows.append(
            {
                "feature": feature,
                "n_subjects": int(len(subject_means)),
                "selected_subject_mean": float(subject_means["selected_mean"].mean()),
                "original_subject_mean": float(subject_means["original_mean"].mean()),
                "mean_difference_original_minus_selected": float(differences.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_t": float(t_result.statistic) if t_result is not None else np.nan,
                "paired_t_p": float(t_result.pvalue) if t_result is not None else np.nan,
                "wilcoxon_p": wilcoxon_p,
                "paired_effect_size_dz": paired_effect_size(differences),
            }
        )
    return pd.DataFrame(rows)


def event_level_contrasts(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        selected = df.loc[df["new_word"] == 0, feature]
        original = df.loc[df["new_word"] == 1, feature]
        t_result = ttest_ind(original, selected, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "feature": feature,
                "selected_n": int(selected.notna().sum()),
                "original_n": int(original.notna().sum()),
                "selected_mean": float(selected.mean()),
                "original_mean": float(original.mean()),
                "mean_difference_original_minus_selected": float(original.mean() - selected.mean()),
                "welch_t": float(t_result.statistic),
                "welch_p": float(t_result.pvalue),
                "cohens_d": cohens_d_independent(selected, original),
            }
        )
    return pd.DataFrame(rows)


def add_fdr(results: pd.DataFrame, p_column: str, alpha: float) -> pd.DataFrame:
    out = results.copy()
    valid = out[p_column].notna()
    out[f"{p_column}_fdr"] = np.nan
    out[f"{p_column}_significant_fdr_{alpha}"] = False
    if valid.any():
        p_values = out.loc[valid, p_column].to_numpy(dtype=float)
        order = np.argsort(p_values)
        ordered = p_values[order]
        n_tests = len(ordered)
        adjusted = np.empty(n_tests)
        running_min = 1.0
        for index in range(n_tests - 1, -1, -1):
            rank = index + 1
            running_min = min(running_min, ordered[index] * n_tests / rank)
            adjusted[index] = running_min
        corrected = np.empty(n_tests)
        corrected[order] = np.minimum(adjusted, 1.0)
        out.loc[valid, f"{p_column}_fdr"] = corrected
        out.loc[valid, f"{p_column}_significant_fdr_{alpha}"] = corrected <= alpha
    return out


def plain_feature_name(feature: str) -> str:
    parts = feature.replace("eeg_", "").split("_")
    if "asymmetry" in parts:
        return feature.replace("eeg_asymmetry_", "left-right asymmetry: ").replace("_", " ")
    return feature.replace("eeg_", "").replace("_", " ")


def write_report(
    output_dir: Path,
    subject_results: pd.DataFrame,
    event_results: pd.DataFrame,
    df: pd.DataFrame,
    alpha: float,
) -> None:
    ranked = subject_results.sort_values("paired_t_p_fdr", na_position="last")
    top_rows = ranked.head(10)
    significant = subject_results[subject_results[f"paired_t_p_significant_fdr_{alpha}"]]

    lines = [
        "# Hypothesis Test: Original Associations vs Selected Words",
        "",
        "Comparison:",
        "",
        "```text",
        "selected suggested word = comparison condition",
        "original association = experimental condition",
        "```",
        "",
        "Primary hypothesis:",
        "",
        "```text",
        "Original associations differ from selected suggested words in pre-response EEG/autonomic state.",
        "```",
        "",
        f"Events used: {len(df)}",
        f"Subjects used: {df['subject'].nunique()}",
        f"Selected-word events: {int((df['new_word'] == 0).sum())}",
        f"Original-association events: {int((df['new_word'] == 1).sum())}",
        "",
        "Primary tests use subject-level condition means. Event-level tests are included as a secondary descriptive check.",
        "",
        "## Subject-Level Results",
        "",
        "| feature | plain meaning | subjects | mean difference | p | FDR p | effect size |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {plain_feature_name(row.feature)} | {row.n_subjects} | "
            f"{row.mean_difference_original_minus_selected:.4f} | {row.paired_t_p:.4f} | "
            f"{row.paired_t_p_fdr:.4f} | {row.paired_effect_size_dz:.3f} |"
        )

    lines.extend(["", "## Conclusion", ""])
    if len(significant) == 0:
        lines.append(
            "No selected feature survived FDR correction at the subject level. The current dataset shows suggestive condition differences, but not enough evidence for a strong statistical conclusion after multiple-comparison correction."
        )
    else:
        lines.append(
            f"{len(significant)} feature(s) survived FDR correction at the subject level. These features provide the strongest current evidence for a difference between original associations and selected suggested words."
        )
    lines.extend(
        [
            "",
            "The strongest subject-level effects should be interpreted cautiously because this is a small dataset with few events per participant and many artifact-flagged EEG epochs.",
        ]
    )
    (output_dir / "hypothesis_test_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_eeg = read_selected_features(args.selected_eeg_features)
    table = pd.read_csv(args.modeling_table)
    table = table[table["new_word"].isin([0, 1])].copy()

    physio_features = [column for column in table.columns if column.startswith("physio_")]
    test_features = selected_eeg + physio_features
    test_features = [feature for feature in test_features if feature in table.columns]

    subject_results = subject_level_contrasts(table, test_features, args.min_condition_count)
    subject_results = add_fdr(subject_results, "paired_t_p", args.alpha)
    event_results = event_level_contrasts(table, test_features)
    event_results = add_fdr(event_results, "welch_p", args.alpha)

    counts = condition_counts(table)
    counts.to_csv(args.output_dir / "condition_counts_by_subject.csv", index=False)
    subject_results.to_csv(args.output_dir / "subject_level_condition_tests.csv", index=False)
    event_results.to_csv(args.output_dir / "event_level_condition_tests.csv", index=False)

    write_report(args.output_dir, subject_results, event_results, table, args.alpha)

    summary = {
        "modeling_table": str(args.modeling_table),
        "events_used": int(len(table)),
        "subjects_used": int(table["subject"].nunique()),
        "selected_word_events": int((table["new_word"] == 0).sum()),
        "original_association_events": int((table["new_word"] == 1).sum()),
        "features_tested": int(len(test_features)),
        "subject_level_fdr_significant_features": subject_results.loc[
            subject_results[f"paired_t_p_significant_fdr_{args.alpha}"], "feature"
        ].tolist(),
    }
    with (args.output_dir / "hypothesis_test_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Events used: {summary['events_used']}")
    print(f"Subjects used: {summary['subjects_used']}")
    print(f"Features tested: {summary['features_tested']}")
    print("\nTop subject-level results")
    print(
        subject_results.sort_values("paired_t_p_fdr", na_position="last")
        .head(10)[
            [
                "feature",
                "n_subjects",
                "mean_difference_original_minus_selected",
                "paired_t_p",
                "paired_t_p_fdr",
                "paired_effect_size_dz",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
