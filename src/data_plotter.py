import pandas as pd
import math
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
from pathlib import Path
from typing import Any, Callable


class FingerDataPlotter:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, filename: str) -> None:
        plt.gcf().savefig(self.output_dir / filename, dpi=200, bbox_inches="tight")

    def boxplot(self, df: pd.DataFrame) -> None:
        df = df.copy()

        cols = [c for c in df.columns if "correct_sequences" in c or "keystrokes" in c]
        num_plots = len(cols)

        # Set grid size
        cols_per_row = 2  # Adjust this to control layout
        rows = math.ceil(num_plots / cols_per_row)

        # Create subplots
        fig, axes = plt.subplots(
            nrows=rows, ncols=cols_per_row, figsize=(6.5 * cols_per_row, 5.5 * rows)
        )
        axes = axes.flatten()  # Flatten in case of multi-row layout-

        # Plot each boxplot
        for i, col in enumerate(cols):
            ax = axes[i]
            sns.boxplot(x="test", y=col, data=df, ax=ax)
            sns.stripplot(
                x="test", y=col, data=df, color="black", size=6, jitter=True, ax=ax
            )
            ax.set_title(f"{col}", fontsize=16)
            # Show y-label only on first column of each row to avoid label overlap.
            if i % cols_per_row == 0:
                ax.set_ylabel("Value", fontsize=14)
            else:
                ax.set_ylabel("")

            ax.tick_params(axis="y", labelsize=12)

            # Set y-axis limits
            if "correct_sequences" in col:
                ax.set_ylim(-2, 35)
            elif "keystrokes" in col:
                ax.set_ylim(-10, 180)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.5)
        self._save_figure("finger_boxplot")


class StateDataPlotter:
    FREQ_PALETTE = {
        "h": "#ff7f0e",  # frequent
        "s": "#2ca02c",  # rare
    }
    FREQ_ORDER = ["h", "s"]

    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _save_figure(self, filename: str) -> None:
        plt.gcf().savefig(self.output_dir / filename, dpi=200, bbox_inches="tight")

    def learning_curve(self, df: pd.DataFrame) -> None:
        participant_means = self._prepare_learning_curve_data(df)
        if participant_means.empty:
            print("No block tests (B1-B8) with transitions available for plotting.")
            return

        group_means, block_order = self._summarize_learning_curve(participant_means)
        self._plot_learning_curve(participant_means, group_means, block_order)

    def _prepare_learning_curve_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return per-participant block means for B1-B8 only."""
        df = df.copy()

        # Ensure transitions are list-like before exploding.
        df["transitions"] = df["transitions"].apply(
            lambda x: x if isinstance(x, list) else []
        )

        exploded = df.explode("transitions")
        exploded["onset_to_onset"] = exploded["transitions"].apply(
            lambda item: item.get("onset_to_onset") if isinstance(item, dict) else np.nan
        )
        exploded["frequency"] = exploded["transitions"].apply(
            lambda item: item.get("frequency") if isinstance(item, dict) else np.nan
        )

        # Keep only tests relevant for learning curves (B1-B8).
        exploded["block"] = exploded["test"].astype(str).str.lower().str.strip()
        exploded = exploded[exploded["block"].str.fullmatch(r"b[1-8]")]
        if exploded.empty:
            return exploded

        exploded["onset_h"] = exploded["onset_to_onset"].where(
            exploded["frequency"].eq("h")
        )
        exploded["onset_s"] = exploded["onset_to_onset"].where(
            exploded["frequency"].eq("s")
        )

        participant_means = exploded.groupby(
            ["participant_id", "block"], as_index=False
        ).agg(
            mean_onset_total=("onset_to_onset", "mean"),
            mean_onset_h=("onset_h", "mean"),
            mean_onset_s=("onset_s", "mean"),
        )

        block_order = [f"b{i}" for i in range(1, 9)]
        participant_means["block"] = pd.Categorical(
            participant_means["block"],
            categories=block_order,
            ordered=True,
        )
        return participant_means.sort_values(["participant_id", "block"])

    def _summarize_learning_curve(
        self, participant_means: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str]]:
        """Aggregate participant means into block means with 95% CIs."""
        block_order = [f"b{i}" for i in range(1, 9)]

        def summarize_with_ci(
            df_in: pd.DataFrame, value_col: str, out_prefix: str
        ) -> pd.DataFrame:
            summary = (
                df_in.groupby("block", as_index=False)[value_col]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            summary[f"{out_prefix}_mean"] = summary["mean"]
            summary[f"{out_prefix}_ci95"] = 1.96 * (
                summary["std"] / np.sqrt(summary["count"])
            )
            summary[f"{out_prefix}_ci95"] = summary[f"{out_prefix}_ci95"].fillna(0.0)
            return summary[["block", f"{out_prefix}_mean", f"{out_prefix}_ci95"]]

        group_total = summarize_with_ci(participant_means, "mean_onset_total", "total")
        group_h = summarize_with_ci(participant_means, "mean_onset_h", "h")
        group_s = summarize_with_ci(participant_means, "mean_onset_s", "s")

        group_means = (
            group_total.merge(group_h, on="block", how="outer")
            .merge(group_s, on="block", how="outer")
            .sort_values("block")
        )
        return group_means, block_order

    def _plot_learning_curve(
        self,
        participant_means: pd.DataFrame,
        group_means: pd.DataFrame,
        block_order: list[str],
    ) -> None:
        """Render and save the learning-curve plot."""
        fig, ax = plt.subplots(figsize=(12, 6))
        participant_colors = plt.cm.tab20(
            np.linspace(0, 1, max(1, participant_means["participant_id"].nunique()))
        )

        for idx, (_, one_participant) in enumerate(
            participant_means.groupby("participant_id")
        ):
            ax.plot(
                one_participant["block"].cat.codes,
                one_participant["mean_onset_total"],
                linewidth=0.8,
                alpha=0.35,
                color=participant_colors[idx % len(participant_colors)],
                zorder=2,
            )

        series = [
            ("total_mean", "total_ci95", "black", "Group mean (total)", 2.5, 1.5),
            ("h_mean", "h_ci95", "tab:blue", "Group mean (h)", 2.0, 1.3),
            ("s_mean", "s_ci95", "tab:orange", "Group mean (s)", 2.0, 1.3),
        ]
        
        for y_col, err_col, color, label, linewidth, elinewidth in series:
            ax.errorbar(
                group_means["block"].cat.codes,
                group_means[y_col],
                yerr=group_means[err_col],
                color=color,
                linewidth=linewidth,
                marker="o",
                label=label,
                capsize=5,
                elinewidth=elinewidth,
                zorder=3,
            )

        ax.set_xticks(range(len(block_order)))
        ax.set_xticklabels([b.upper() for b in block_order])
        ax.set_xlabel("Block")
        ax.set_ylabel("Mean onset-to-onset (s)")
        ax.set_title("Learning Curve (B1-B8)")
        ax.grid(axis="y", alpha=0.2)
        ax.legend()
        plt.tight_layout()
        self._save_figure("learning_curve.png")

    def pre_post_boxplot(self, df: pd.DataFrame) -> None:
        self._plot_pre_post_distribution(df, sns.boxplot)

    def pre_post_violin(self, df: pd.DataFrame) -> None:
        self._plot_pre_post_distribution(
            df,
            sns.violinplot,
            title_suffix=" (Violin Plot)",
        )

    def _prepare_pre_post_plot_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return cleaned pre/post transition rows ready for plotting."""
        df = df.copy()
        df_pre_post = df[
            df["test"].str.contains("pre|post", case=False, na=False)
        ].copy()
        exploded = df_pre_post.explode("transitions", ignore_index=True)
        exploded["freq"] = exploded["transitions"].apply(
            lambda item: item.get("frequency") if isinstance(item, dict) else np.nan
        )
        exploded["onset_to_onset"] = exploded["transitions"].apply(
            lambda item: (
                item.get("onset_to_onset") if isinstance(item, dict) else np.nan
            )
        )
        exploded["freq"] = exploded["freq"].astype(str).str.strip().str.lower()
        exploded = exploded[exploded["freq"].isin(self.FREQ_ORDER)].copy()
        return exploded.dropna(subset=["onset_to_onset"])

    def _plot_pre_post_distribution(
        self,
        df: pd.DataFrame,
        plot_func: Callable[..., Any],
        title_suffix: str = "",
    ) -> None:
        """Plot the pre/post transition distribution with a shared box/violin layout."""
        exploded = self._prepare_pre_post_plot_data(df)
        if exploded.empty:
            print("No pre/post transitions available for plotting.")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        metric = "onset_to_onset"
        label = "Transition time"

        plot_func(
            data=exploded,
            x="test",
            y=metric,
            hue="freq",
            ax=ax,
            palette=self.FREQ_PALETTE,
            hue_order=self.FREQ_ORDER,
        )
        sns.stripplot(
            data=exploded,
            x="test",
            y=metric,
            hue="freq",
            color="black",
            size=2,
            alpha=0.4,
            jitter=True,
            ax=ax,
            dodge=True,
            hue_order=self.FREQ_ORDER,
            legend=False,
        )

        ax.set_title(f"{label}{title_suffix}", fontsize=14, fontweight="bold")
        ax.set_xlabel("test", fontsize=12)
        ax.set_ylabel("Time (s)", fontsize=12)
        ax.tick_params(axis="y", labelsize=10)
        ax.legend(title="Frequency", loc="upper right")

        plt.tight_layout()
        filename = "pre_post_violin.png" if plot_func is sns.violinplot else "pre_post_boxplot.png"
        self._save_figure(filename)

    
