#!/usr/bin/env python3
"""Compare high- and low-coherence emotional word transitions."""

from __future__ import annotations

import argparse
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, ttest_1samp, ttest_ind, wilcoxon


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modeling-table", type=Path, default=Path("outputs/multimodal_model/multimodal_modeling_dataset.csv"))
    parser.add_argument("--selected-eeg-features", type=Path, default=Path("config/selected_eeg_features.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/semantic_transition_analysis"))
    parser.add_argument("--min-condition-count", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def read_selected_features(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


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


def paired_effect_size(differences: pd.Series) -> float:
    values = differences.dropna().to_numpy(dtype=float)
    if len(values) < 2:
        return np.nan
    sd = np.std(values, ddof=1)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(np.mean(values) / sd)


def cohens_d_independent(group0: pd.Series, group1: pd.Series) -> float:
    x0 = group0.dropna().to_numpy(dtype=float)
    x1 = group1.dropna().to_numpy(dtype=float)
    if len(x0) < 2 or len(x1) < 2:
        return np.nan
    pooled = math.sqrt(((len(x0) - 1) * np.var(x0, ddof=1) + (len(x1) - 1) * np.var(x1, ddof=1)) / (len(x0) + len(x1) - 2))
    if pooled == 0 or np.isnan(pooled):
        return 0.0
    return float((np.mean(x1) - np.mean(x0)) / pooled)


def bootstrap_ci(values: pd.Series, random_state: int = 42, n_bootstrap: int = 5000) -> tuple[float, float]:
    clean = values.dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(random_state)
    estimates = []
    for _ in range(n_bootstrap):
        estimates.append(np.mean(rng.choice(clean, size=len(clean), replace=True)))
    low, high = np.percentile(estimates, [2.5, 97.5])
    return float(low), float(high)


def assign_subject_median_conditions(df: pd.DataFrame) -> pd.DataFrame:
    selected = df[(df["new_word"] == 0) & (df["correlation"] > 0)].copy()
    selected["semantic_condition"] = pd.Series(pd.NA, index=selected.index, dtype="string")
    selected["subject_correlation_median"] = np.nan

    for subject, subject_df in selected.groupby("subject"):
        if subject_df["correlation"].nunique() < 2:
            continue
        median = subject_df["correlation"].median()
        selected.loc[subject_df.index, "subject_correlation_median"] = median
        selected.loc[subject_df.index, "semantic_condition"] = np.where(
            subject_df["correlation"] <= median,
            "low_coherence",
            "high_coherence",
        )
    return selected.dropna(subset=["semantic_condition"]).reset_index(drop=True)


def condition_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["subject", "semantic_condition"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )


def subject_level_tests(df: pd.DataFrame, features: list[str], min_condition_count: int) -> pd.DataFrame:
    counts = condition_counts(df)
    for column in ["low_coherence", "high_coherence"]:
        if column not in counts.columns:
            counts[column] = 0
    eligible_subjects = counts[
        (counts["low_coherence"] >= min_condition_count)
        & (counts["high_coherence"] >= min_condition_count)
    ]["subject"].tolist()
    eligible = df[df["subject"].isin(eligible_subjects)].copy()

    rows = []
    for feature in features:
        subject_means = (
            eligible.groupby(["subject", "semantic_condition"])[feature]
            .mean()
            .unstack()
        )
        if "low_coherence" not in subject_means or "high_coherence" not in subject_means:
            continue
        subject_means = subject_means.dropna(subset=["low_coherence", "high_coherence"])
        differences = subject_means["low_coherence"] - subject_means["high_coherence"]

        if len(differences) >= 2:
            t_result = ttest_1samp(differences, popmean=0.0, nan_policy="omit")
            try:
                if np.nanstd(differences.to_numpy(dtype=float)) == 0:
                    wilcoxon_p = np.nan
                else:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        wilcoxon_p = float(wilcoxon(differences).pvalue)
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
                "high_coherence_subject_mean": float(subject_means["high_coherence"].mean()),
                "low_coherence_subject_mean": float(subject_means["low_coherence"].mean()),
                "mean_difference_low_minus_high": float(differences.mean()),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "paired_t": float(t_result.statistic) if t_result is not None else np.nan,
                "paired_t_p": float(t_result.pvalue) if t_result is not None else np.nan,
                "wilcoxon_p": wilcoxon_p,
                "paired_effect_size_dz": paired_effect_size(differences),
            }
        )
    return pd.DataFrame(rows)


def event_level_tests(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        high = df.loc[df["semantic_condition"] == "high_coherence", feature]
        low = df.loc[df["semantic_condition"] == "low_coherence", feature]
        t_result = ttest_ind(low, high, equal_var=False, nan_policy="omit")
        rows.append(
            {
                "feature": feature,
                "high_coherence_n": int(high.notna().sum()),
                "low_coherence_n": int(low.notna().sum()),
                "high_coherence_mean": float(high.mean()),
                "low_coherence_mean": float(low.mean()),
                "mean_difference_low_minus_high": float(low.mean() - high.mean()),
                "welch_t": float(t_result.statistic),
                "welch_p": float(t_result.pvalue),
                "cohens_d": cohens_d_independent(high, low),
            }
        )
    return pd.DataFrame(rows)


def continuous_subject_correlations(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        fisher_z_values = []
        event_counts = []
        for subject, subject_df in df.groupby("subject"):
            x = pd.to_numeric(subject_df["correlation"], errors="coerce")
            y = pd.to_numeric(subject_df[feature], errors="coerce")
            valid = x.notna() & y.notna()
            if valid.sum() < 3 or x[valid].nunique() < 2 or y[valid].nunique() < 2:
                continue
            r_value = pearsonr(x[valid], y[valid]).statistic
            if np.isfinite(r_value) and abs(r_value) < 1:
                fisher_z_values.append(np.arctanh(r_value))
                event_counts.append(int(valid.sum()))

        if len(fisher_z_values) < 2:
            rows.append(
                {
                    "feature": feature,
                    "n_subjects": len(fisher_z_values),
                    "mean_within_subject_r": np.nan,
                    "mean_events_per_subject": np.nan,
                    "one_sample_t": np.nan,
                    "one_sample_p": np.nan,
                }
            )
            continue

        z_values = pd.Series(fisher_z_values)
        t_result = ttest_1samp(z_values, popmean=0.0, nan_policy="omit")
        rows.append(
            {
                "feature": feature,
                "n_subjects": len(fisher_z_values),
                "mean_within_subject_r": float(np.tanh(z_values.mean())),
                "mean_events_per_subject": float(np.mean(event_counts)),
                "one_sample_t": float(t_result.statistic),
                "one_sample_p": float(t_result.pvalue),
            }
        )
    return pd.DataFrame(rows)


def plain_feature_name(feature: str) -> str:
    if feature.startswith("eeg_asymmetry_"):
        return feature.replace("eeg_asymmetry_", "left-right asymmetry: ").replace("_", " ")
    if feature.startswith("eeg_"):
        return feature.replace("eeg_", "").replace("_", " ")
    if feature.startswith("physio_"):
        return feature.replace("physio_", "").replace("_", " ")
    return feature.replace("_", " ")


def write_report(
    output_dir: Path,
    df: pd.DataFrame,
    subject_results: pd.DataFrame,
    continuous_results: pd.DataFrame,
    alpha: float,
) -> None:
    ranked = subject_results.sort_values("paired_t_p_fdr", na_position="last")
    significant = subject_results[subject_results[f"paired_t_p_significant_fdr_{alpha}"]]
    top_rows = ranked.head(10)
    top_continuous = continuous_results.sort_values("one_sample_p_fdr", na_position="last").head(8)

    lines = [
        "# Semantic Transition Analysis",
        "",
        "Comparison:",
        "",
        "```text",
        "high semantic coherence = control condition",
        "low semantic coherence = divergent transition condition",
        "```",
        "",
        "Hypothesis:",
        "",
        "```text",
        "Lower-coherence emotional word transitions differ from higher-coherence transitions in pre-response EEG/autonomic state.",
        "```",
        "",
        f"Events used: {len(df)}",
        f"Subjects used: {df['subject'].nunique()}",
        f"High-coherence events: {int((df['semantic_condition'] == 'high_coherence').sum())}",
        f"Low-coherence events: {int((df['semantic_condition'] == 'low_coherence').sum())}",
        "",
        "Only selected-word events with nonzero semantic similarity are used, so this analysis is not confounded with original-word generation.",
        "",
        "## Subject-Level Results",
        "",
        "| feature | plain meaning | subjects | low-high difference | p | FDR p | effect size |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {plain_feature_name(row.feature)} | {row.n_subjects} | "
            f"{row.mean_difference_low_minus_high:.4f} | {row.paired_t_p:.4f} | "
            f"{row.paired_t_p_fdr:.4f} | {row.paired_effect_size_dz:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Continuous Similarity Results",
            "",
            "| feature | plain meaning | subjects | mean within-subject r | p | FDR p |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in top_continuous.itertuples(index=False):
        lines.append(
            f"| `{row.feature}` | {plain_feature_name(row.feature)} | {row.n_subjects} | "
            f"{row.mean_within_subject_r:.3f} | {row.one_sample_p:.4f} | {row.one_sample_p_fdr:.4f} |"
        )

    lines.extend(["", "## Conclusion", ""])
    if len(significant) == 0:
        lines.append(
            "No feature survived FDR correction at the subject level. The current analysis does not support a strong corrected statistical claim about semantic coherence."
        )
    else:
        feature_list = ", ".join(f"`{feature}`" for feature in significant["feature"].tolist())
        lines.append(
            f"{len(significant)} feature(s) survived FDR correction at the subject level: {feature_list}."
        )
        lines.append(
            "These features provide the strongest current evidence that low-coherence emotional word transitions differ from high-coherence transitions."
        )
    lines.append("")
    lines.append(
        "This analysis is still exploratory because the sample is small, but it is closer to the core neuroscience question: how semantic movement through emotional associations relates to brain and autonomic state."
    )
    (output_dir / "semantic_transition_report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    selected_eeg = read_selected_features(args.selected_eeg_features)
    table = pd.read_csv(args.modeling_table)
    semantic = assign_subject_median_conditions(table)
    physio_features = [column for column in semantic.columns if column.startswith("physio_")]
    test_features = [feature for feature in selected_eeg + physio_features if feature in semantic.columns]

    counts = condition_counts(semantic)
    subject_results = add_fdr(subject_level_tests(semantic, test_features, args.min_condition_count), "paired_t_p", args.alpha)
    event_results = add_fdr(event_level_tests(semantic, test_features), "welch_p", args.alpha)
    continuous_results = add_fdr(continuous_subject_correlations(semantic, test_features), "one_sample_p", args.alpha)

    semantic.to_csv(args.output_dir / "semantic_transition_dataset.csv", index=False)
    counts.to_csv(args.output_dir / "semantic_condition_counts_by_subject.csv", index=False)
    subject_results.to_csv(args.output_dir / "semantic_subject_level_tests.csv", index=False)
    event_results.to_csv(args.output_dir / "semantic_event_level_tests.csv", index=False)
    continuous_results.to_csv(args.output_dir / "semantic_continuous_correlations.csv", index=False)
    write_report(args.output_dir, semantic, subject_results, continuous_results, args.alpha)

    summary = {
        "modeling_table": str(args.modeling_table),
        "events_used": int(len(semantic)),
        "subjects_used": int(semantic["subject"].nunique()),
        "high_coherence_events": int((semantic["semantic_condition"] == "high_coherence").sum()),
        "low_coherence_events": int((semantic["semantic_condition"] == "low_coherence").sum()),
        "features_tested": int(len(test_features)),
        "subject_level_fdr_significant_features": subject_results.loc[
            subject_results[f"paired_t_p_significant_fdr_{args.alpha}"], "feature"
        ].tolist(),
        "continuous_fdr_significant_features": continuous_results.loc[
            continuous_results[f"one_sample_p_significant_fdr_{args.alpha}"], "feature"
        ].tolist(),
    }
    with (args.output_dir / "semantic_transition_summary.json").open("w") as f:
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
                "mean_difference_low_minus_high",
                "paired_t_p",
                "paired_t_p_fdr",
                "paired_effect_size_dz",
            ]
        ]
        .to_string(index=False)
    )
    print("\nTop continuous semantic-similarity results")
    print(
        continuous_results.sort_values("one_sample_p_fdr", na_position="last")
        .head(10)[
            [
                "feature",
                "n_subjects",
                "mean_within_subject_r",
                "one_sample_p",
                "one_sample_p_fdr",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
