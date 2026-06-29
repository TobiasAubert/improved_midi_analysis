import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import zscore
from scipy.stats import shapiro
from scipy.stats import mannwhitneyu
from scipy.stats import ttest_ind
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
import math
from scipy.stats import wilcoxon
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multicomp import pairwise_tukeyhsd
import re
from pathlib import Path
from typing import Any


class FingerDataAnalyser:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.tests: list[dict[str, Any]] = []

    def refomrat_df(self, in_df: pd.DataFrame) -> pd.DataFrame:
        df = in_df.copy()

        if "participant_id" not in df.columns or "test" not in df.columns:
            raise ValueError("Input DataFrame must contain 'participant_id' and 'test' columns")

        if "ratio" not in df.columns and {"correct_sequences", "keystrokes"}.issubset(df.columns):
            df["ratio"] = df["correct_sequences"] / df["keystrokes"]

        value_columns = [
            column
            for column in ["correct_sequences", "keystrokes", "ratio"]
            if column in df.columns
        ]

        wide_df = df.pivot(index="participant_id", columns="test", values=value_columns)
        wide_df.columns = [f"{test}_{column}" for column, test in wide_df.columns]
        return wide_df.reset_index()

    def add_test_result(self, **result: Any) -> None:
        self.tests.append(result)

    def write_to_md(self, filename: str = "fingerdex.md") -> None:
        md_path = self.output_dir / filename

        lines = ["# Statistical Tests", ""]

        if not self.tests:
            lines.append("No statistical tests were recorded.")
            self._write_markdown(md_path, lines)
            return

        columns = list(self.tests[0].keys())
        rows = [[self._format_md_value(result.get(column)) for column in columns] for result in self.tests]
        lines.extend(self._markdown_table(columns, rows))
        self._write_markdown(md_path, lines)

    @staticmethod
    def _format_md_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return lines

    @staticmethod
    def _write_markdown(md_path: Path, lines: list[str]) -> None:
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _score_summary_table(df_numeric: pd.DataFrame) -> pd.DataFrame:
        if df_numeric.empty:
            raise ValueError("No numeric columns were found for analysis")

        n = len(df_numeric)
        dfree = n - 1
        confidence = 95
        alpha = (100 - confidence) / 100

        mean = df_numeric.mean()
        std = df_numeric.std()
        se = std / np.sqrt(n)
        t_critical = t.ppf(1 - alpha / 2, dfree)
        ci_lower = mean - t_critical * se
        ci_upper = mean + t_critical * se

        df_summary = pd.DataFrame(
            {
                "Mean": mean,
                "Std Dev": std,
                "Std Error": se,
                "CI Lower (95%)": ci_lower,
                "CI Upper (95%)": ci_upper,
                "T Critical": t_critical,
            }
        )

        for col in df_numeric.columns:
            _, p = shapiro(df_numeric[col])
            df_summary.loc[col, "Normality"] = p

        return df_summary

    @staticmethod
    def check_normality_shapiro(df: pd.DataFrame, group_name: str = "") -> pd.DataFrame:
        rows = []
        df_numeric = df.select_dtypes(include=np.number)
        for col in df_numeric.columns:
            stat, p = shapiro(df_numeric[col])
            rows.append(
                {
                    "Group": group_name,
                    "Column": col,
                    "W": stat,
                    "p_value": p,
                    "Normal": p >= 0.05,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def remove_outliers_z(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        df_numeric = df.select_dtypes(include=np.number)
        if df_numeric.empty:
            return df.copy()

        z_scores = np.abs(zscore(df_numeric, nan_policy="omit"))
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        mask = (z_scores < threshold).all(axis=1)
        return df.loc[mask].copy()

    @staticmethod
    def grubbs_test(values: pd.Series, alpha: float = 0.05):
        values = values.dropna()
        n = len(values)
        if n < 3:
            return None

        mean_y = np.mean(values)
        std_y = np.std(values, ddof=1)
        if std_y == 0:
            return None

        abs_diffs = np.abs(values - mean_y)
        max_dev_idx = abs_diffs.idxmax()
        g_calculated = abs_diffs[max_dev_idx] / std_y

        t_crit = t.ppf(1 - alpha / (2 * n), n - 2)
        g_critical = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit**2 / (n - 2 + t_crit**2))

        if g_calculated > g_critical:
            return max_dev_idx
        return None

    def replace_outliers_with_nan(
        self,
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        alpha: float = 0.05,
        verbose: bool = False,
    ) -> pd.DataFrame:
        df_clean = raw_df.copy()

        if "Participant_ID" in df_clean.columns:
            df_clean = df_clean.set_index("Participant_ID")

        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        for col in numeric_cols:
            if col not in summary_df.index:
                continue

            values = df_clean[col].dropna()
            if len(values) < 3:
                continue

            p_normal = summary_df.loc[col, "Normality"]
            is_normal = p_normal > 0.05

            if is_normal:
                outlier_idx = self.grubbs_test(values, alpha=alpha)
                if outlier_idx is not None:
                    df_clean.loc[outlier_idx, col] = np.nan
                    if verbose:
                        print(f"[Grubbs] Outlier in '{col}' at participant {outlier_idx} set to NaN")
            else:
                q1 = values.quantile(0.25)
                q3 = values.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_mask = (values < lower) | (values > upper)
                for idx in values.index[outlier_mask]:
                    df_clean.loc[idx, col] = np.nan
                    if verbose:
                        print(f"[IQR] Outlier in '{col}' at participant {idx} set to NaN")

        if "Participant_ID" in raw_df.columns:
            df_clean = df_clean.reset_index()

        return df_clean

    def compare_groups_ttest(
        self,
        group1: pd.Series,
        group2: pd.Series,
        group1_name: str = "Group1",
        group2_name: str = "Group2",
    ) -> pd.DataFrame:
        group1 = group1.dropna()
        group2 = group2.dropna()

        _, p_norm1 = shapiro(group1)
        _, p_norm2 = shapiro(group2)

        mean1 = group1.mean()
        mean2 = group2.mean()
        sd1 = group1.std()
        sd2 = group2.std()

        if p_norm1 >= 0.05 and p_norm2 >= 0.05:
            t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
            test_name = "Welch's t-test"
            pooled_sd = np.sqrt((sd1**2 + sd2**2) / 2)
            effect_size = (mean1 - mean2) / pooled_sd if pooled_sd != 0 else np.nan

            if np.isnan(effect_size):
                effect_label = "N/A"
            elif abs(effect_size) < 0.2:
                effect_label = "Negligible"
            elif abs(effect_size) < 0.5:
                effect_label = "Small"
            elif abs(effect_size) < 0.8:
                effect_label = "Medium"
            else:
                effect_label = "Large"
        else:
            t_stat, p_value = mannwhitneyu(group1, group2, alternative="two-sided")
            test_name = "Mann-Whitney U"
            effect_size = np.nan
            effect_label = "N/A"

        return pd.DataFrame(
            [
                {
                    "Test": test_name,
                    f"Mean {group1_name}": mean1,
                    f"Mean {group2_name}": mean2,
                    "Statistic": t_stat,
                    "p-value": p_value,
                    "Significant (p < 0.05)": p_value < 0.05,
                    "Effect Size": effect_size,
                    "Effect Size Label": effect_label,
                }
            ]
        )

    def analyze_scores(self, df: pd.DataFrame, filename: str = "score_summary.md") -> None:
        df_wide = self.refomrat_df(df)
        df_numeric = df_wide.select_dtypes(include=np.number)
        df_summary = self._score_summary_table(df_numeric)

        md_path = self.output_dir / filename
        lines = ["# Score Summary", ""]
        rows = [
            [
                self._format_md_value(column),
                self._format_md_value(row.get("Mean")),
                self._format_md_value(row.get("Std Dev")),
                self._format_md_value(row.get("Std Error")),
                self._format_md_value(row.get("CI Lower (95%)")),
                self._format_md_value(row.get("CI Upper (95%)")),
                self._format_md_value(row.get("T Critical")),
                self._format_md_value(row.get("Normality")),
            ]
            for column, row in df_summary.iterrows()
        ]
        lines.extend(
            self._markdown_table(
                ["Column", "Mean", "Std Dev", "Std Error", "CI Lower (95%)", "CI Upper (95%)", "T Critical", "Normality"],
                rows,
            )
        )
        self._write_markdown(md_path, lines)

    def analyze_by_timepoint(self, df: pd.DataFrame, filename: str = "timepoint_summary.md") -> Path:
        """Write a Markdown summary comparing pretest and posttest averages and their statistical test."""
        df_wide = self.refomrat_df(df)

        pretest_cols = [
            col
            for col in df_wide.columns
            if col.startswith("FT1_") or col.startswith("FT2_")
        ]
        posttest_cols = [
            col
            for col in df_wide.columns
            if col.startswith("FT3_") or col.startswith("FT4_")
        ]

        if not pretest_cols or not posttest_cols:
            raise ValueError("Could not find pretest and posttest columns in the DataFrame")

        df_analysis = pd.DataFrame(
            {
                "Participant_ID": df_wide["participant_id"],
                "Pretest_Avg": df_wide[pretest_cols].mean(axis=1),
                "Posttest_Avg": df_wide[posttest_cols].mean(axis=1),
            }
        )

        pre_df = pd.DataFrame({"Score": df_analysis["Pretest_Avg"]})
        post_df = pd.DataFrame({"Score": df_analysis["Posttest_Avg"]})

        pre_summary = self._score_summary_table(pre_df)
        post_summary = self._score_summary_table(post_df)
        comparison = self.compare_groups_ttest(df_analysis["Pretest_Avg"], df_analysis["Posttest_Avg"], "Pretest", "Posttest")

        md_path = self.output_dir / filename
        lines = ["# Timepoint Summary", "", "## Per Participant", ""]
        participant_rows = [
            [
                self._format_md_value(row["Participant_ID"]),
                self._format_md_value(row["Pretest_Avg"]),
                self._format_md_value(row["Posttest_Avg"]),
            ]
            for _, row in df_analysis.iterrows()
        ]
        lines.extend(self._markdown_table(["Participant_ID", "Pretest_Avg", "Posttest_Avg"], participant_rows))

        lines.extend(["", "## Overview", ""])
        overview_rows = []
        for group_name, summary_df in (("Pretest", pre_summary), ("Posttest", post_summary)):
            row = summary_df.loc["Score"]
            overview_rows.append(
                [
                    self._format_md_value(group_name),
                    self._format_md_value(row.get("Mean")),
                    self._format_md_value(row.get("Std Dev")),
                    self._format_md_value(row.get("Std Error")),
                    self._format_md_value(row.get("CI Lower (95%)")),
                    self._format_md_value(row.get("CI Upper (95%)")),
                    self._format_md_value(row.get("T Critical")),
                    self._format_md_value(row.get("Normality")),
                ]
            )
        lines.extend(
            self._markdown_table(
                ["Group", "Mean", "Std Dev", "Std Error", "CI Lower (95%)", "CI Upper (95%)", "T Critical", "Normality"],
                overview_rows,
            )
        )

        lines.extend([
            "",
            "## Comparison",
            "",
            *self._markdown_table(
                ["Test", "Mean Pretest", "Mean Posttest", "Statistic", "p-value", "Significant", "Effect Size", "Effect Size Label"],
                [
                    [
                        self._format_md_value(row.get("Test")),
                        self._format_md_value(row.get("Mean Pretest")),
                        self._format_md_value(row.get("Mean Posttest")),
                        self._format_md_value(row.get("Statistic")),
                        self._format_md_value(row.get("p-value")),
                        self._format_md_value(row.get("Significant (p < 0.05)")),
                        self._format_md_value(row.get("Effect Size")),
                        self._format_md_value(row.get("Effect Size Label")),
                    ]
                    for _, row in comparison.iterrows()
                ],
            ),
        ])
        self._write_markdown(md_path, lines)
        return md_path

    def run_analysis_pipeline(self, df: pd.DataFrame) -> dict[str, Path]:
        """Run the full fingertest analysis pipeline and write all Markdown reports."""
        score_summary_path = self.output_dir / "score_summary.md"
        timepoint_summary_path = self.output_dir / "timepoint_summary.md"

        self.analyze_scores(df, filename=score_summary_path.name)
        self.analyze_by_timepoint(df, filename=timepoint_summary_path.name)

        return {
            "score_summary": score_summary_path,
            "timepoint_summary": timepoint_summary_path,
        }


class StateDataAnalyser:
    def __init__(self, output_dir: str | Path) -> None:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _format_md_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    @staticmethod
    def _markdown_table(headers: list[str], rows: list[list[Any]]) -> list[str]:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(value) for value in row) + " |")
        return lines

    @staticmethod
    def _write_markdown(md_path: Path, lines: list[str]) -> None:
        md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _score_summary_table(df_numeric: pd.DataFrame) -> pd.DataFrame:
        if df_numeric.empty:
            raise ValueError("No numeric columns were found for analysis")

        n = len(df_numeric)
        dfree = n - 1
        confidence = 95
        alpha = (100 - confidence) / 100

        mean = df_numeric.mean()
        median = df_numeric.median()
        std = df_numeric.std()
        se = std / np.sqrt(n)
        t_critical = t.ppf(1 - alpha / 2, dfree)
        q1 = df_numeric.quantile(0.25)
        q3 = df_numeric.quantile(0.75)
        iqr = q3 - q1

        df_summary = pd.DataFrame(
            {
                "N": n,
                "Median": median,
                "Q1": q1,
                "Q3": q3,
                "IQR": iqr,
                "Mean": mean,
                "Std Dev": std,
                "Std Error": se,
                "CI Lower (95%)": mean - t_critical * se,
                "CI Upper (95%)": mean + t_critical * se,
            }
        )

        for col in df_numeric.columns:
            if df_numeric[col].dropna().shape[0] >= 3:
                _, p = shapiro(df_numeric[col].dropna())
                df_summary.loc[col, "Shapiro p"] = p
            else:
                df_summary.loc[col, "Shapiro p"] = np.nan

        return df_summary

    @staticmethod
    def check_normality_shapiro(df: pd.DataFrame, group_name: str = "") -> pd.DataFrame:
        rows = []
        df_numeric = df.select_dtypes(include=np.number)
        for col in df_numeric.columns:
            stat, p = shapiro(df_numeric[col].dropna())
            rows.append(
                {
                    "Group": group_name,
                    "Column": col,
                    "W": stat,
                    "p_value": p,
                    "Normal": p >= 0.05,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def remove_outliers_z(df: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
        df_numeric = df.select_dtypes(include=np.number)
        if df_numeric.empty:
            return df.copy()

        z_scores = np.abs(zscore(df_numeric, nan_policy="omit"))
        z_scores = np.nan_to_num(z_scores, nan=0.0)
        mask = (z_scores < threshold).all(axis=1)
        return df.loc[mask].copy()

    @staticmethod
    def _flatten_transitions(df: pd.DataFrame) -> pd.DataFrame:
        required_cols = {"participant_id", "test", "transitions"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        rows: list[dict[str, Any]] = []
        for _, entry in df.iterrows():
            transitions = entry.get("transitions") or []
            for transition in transitions:
                if not isinstance(transition, dict):
                    continue
                row = {
                    "Participant_ID": entry.get("participant_id"),
                    "Test": entry.get("test"),
                }
                row.update(transition)
                rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def _build_wide_df(df_transitions: pd.DataFrame, allowed_tests: list[str]) -> pd.DataFrame:
        if df_transitions.empty:
            raise ValueError("No transition rows were available for analysis")

        df = df_transitions.copy()
        df["Test"] = df["Test"].astype(str).str.upper().str.strip()
        df = df[df["Test"].isin(allowed_tests)].copy()
        if df.empty:
            raise ValueError(f"No rows found for tests: {allowed_tests}")

        df["onset_to_onset"] = pd.to_numeric(df["onset_to_onset"], errors="coerce")
        df = df.dropna(subset=["onset_to_onset"])
        if df.empty:
            raise ValueError("No valid onset_to_onset values found for analysis")

        df_wide = (
            df.pivot_table(
                index="Participant_ID",
                columns="Test",
                values="onset_to_onset",
                aggfunc="median",
            )
            .reindex(columns=allowed_tests)
            .reset_index()
        )
        return df_wide

    @staticmethod
    def _run_log_rm_anova(df_wide: pd.DataFrame, level_cols: list[str], subject_label: str):
        df_level = df_wide[["Participant_ID", *level_cols]].copy()

        if (df_level[level_cols] < 0).any().any():
            raise ValueError("Negative transition times found; log-transform is not possible.")

        df_complete = df_level.dropna(subset=level_cols).copy()
        if df_complete.empty:
            raise ValueError("No complete cases available for repeated-measures ANOVA.")

        df_complete[level_cols] = np.log1p(df_complete[level_cols])

        df_long = df_complete.melt(
            id_vars="Participant_ID",
            value_vars=level_cols,
            var_name=subject_label,
            value_name="log_onset_to_onset",
        )

        anova_model = AnovaRM(
            data=df_long,
            depvar="log_onset_to_onset",
            subject="Participant_ID",
            within=[subject_label],
        )
        anova_result = anova_model.fit()

        anova_table = anova_result.anova_table.reset_index().rename(columns={"index": "Effect"})
        anova_table.insert(1, "N subjects", df_complete["Participant_ID"].nunique())

        return df_complete, df_long, anova_table

    @staticmethod
    def _run_wilcoxon(df_wide: pd.DataFrame, first_col: str, second_col: str) -> pd.DataFrame:
        paired = df_wide[[first_col, second_col]].dropna()
        if paired.empty:
            raise ValueError("No complete paired cases available for Wilcoxon test.")

        first = paired[first_col]
        second = paired[second_col]
        stat, p_value = wilcoxon(first, second, zero_method="wilcox")

        return pd.DataFrame(
            [
                {
                    "Test": "Wilcoxon signed-rank",
                    "N pairs": len(paired),
                    f"{first_col} median": first.median(),
                    f"{second_col} median": second.median(),
                    "Statistic": stat,
                    "p-value": p_value,
                    "Significant (p < 0.05)": p_value < 0.05,
                }
            ]
        )

    def analyze_transition_pipeline(self, df: pd.DataFrame, filename: str = "state_transition_summary.md") -> Path:
        """Run the useful state-transition tests and write a Markdown report."""
        df_long = self._flatten_transitions(df)

        block_tests = [f"B{i}" for i in range(1, 9)]
        prepost_tests = ["PRE", "POST"]

        block_df = self._build_wide_df(df_long, block_tests)
        prepost_df = self._build_wide_df(df_long, prepost_tests)

        block_summary = self._score_summary_table(block_df.drop(columns=["Participant_ID"], errors="ignore"))
        prepost_summary = self._score_summary_table(prepost_df.drop(columns=["Participant_ID"], errors="ignore"))

        block_complete, block_long, block_anova = self._run_log_rm_anova(block_df, block_tests, "Block")
        prepost_complete, prepost_long, prepost_anova = self._run_log_rm_anova(prepost_df, prepost_tests, "Timepoint")
        wilcoxon_results = self._run_wilcoxon(prepost_df, "PRE", "POST")

        md_path = self.output_dir / filename
        lines = ["# State Transition Summary", ""]

        lines.extend(["## Block Transition Times (B1-B8)", ""])
        lines.extend(self._markdown_table(
            ["Participant_ID", *block_tests],
            [
                [self._format_md_value(row["Participant_ID"]), *[self._format_md_value(row[col]) for col in block_tests]]
                for _, row in block_df.iterrows()
            ],
        ))
        lines.extend(["", "### Descriptive Summary", ""])
        block_rows = [
            [
                self._format_md_value(column),
                self._format_md_value(row.get("N")),
                self._format_md_value(row.get("Median")),
                self._format_md_value(row.get("Q1")),
                self._format_md_value(row.get("Q3")),
                self._format_md_value(row.get("IQR")),
                self._format_md_value(row.get("Mean")),
                self._format_md_value(row.get("Std Dev")),
                self._format_md_value(row.get("Std Error")),
                self._format_md_value(row.get("CI Lower (95%)")),
                self._format_md_value(row.get("CI Upper (95%)")),
                self._format_md_value(row.get("Shapiro p")),
            ]
            for column, row in block_summary.iterrows()
        ]
        lines.extend(self._markdown_table(
            ["Block", "N", "Median", "Q1", "Q3", "IQR", "Mean", "Std Dev", "Std Error", "CI Lower (95%)", "CI Upper (95%)", "Shapiro p"],
            block_rows,
        ))
        lines.extend(["", "### Log Repeated-Measures ANOVA", ""])
        lines.extend(self._markdown_table(list(block_anova.columns), block_anova.astype(object).values.tolist()))

        lines.extend(["", "## Pretest/Posttest Transition Times", ""])
        lines.extend(self._markdown_table(
            ["Participant_ID", "PRE", "POST"],
            [
                [self._format_md_value(row["Participant_ID"]), self._format_md_value(row["PRE"]), self._format_md_value(row["POST"])]
                for _, row in prepost_df.iterrows()
            ],
        ))
        lines.extend(["", "### Descriptive Summary", ""])
        prepost_rows = [
            [
                self._format_md_value(column),
                self._format_md_value(row.get("N")),
                self._format_md_value(row.get("Median")),
                self._format_md_value(row.get("Q1")),
                self._format_md_value(row.get("Q3")),
                self._format_md_value(row.get("IQR")),
                self._format_md_value(row.get("Mean")),
                self._format_md_value(row.get("Std Dev")),
                self._format_md_value(row.get("Std Error")),
                self._format_md_value(row.get("CI Lower (95%)")),
                self._format_md_value(row.get("CI Upper (95%)")),
                self._format_md_value(row.get("Shapiro p")),
            ]
            for column, row in prepost_summary.iterrows()
        ]
        lines.extend(self._markdown_table(
            ["Timepoint", "N", "Median", "Q1", "Q3", "IQR", "Mean", "Std Dev", "Std Error", "CI Lower (95%)", "CI Upper (95%)", "Shapiro p"],
            prepost_rows,
        ))
        lines.extend(["", "### Log Repeated-Measures ANOVA", ""])
        lines.extend(self._markdown_table(list(prepost_anova.columns), prepost_anova.astype(object).values.tolist()))
        lines.extend(["", "### Wilcoxon Signed-Rank Test", ""])
        lines.extend(self._markdown_table(list(wilcoxon_results.columns), wilcoxon_results.astype(object).values.tolist()))

        self._write_markdown(md_path, lines)
        return md_path
