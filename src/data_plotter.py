import pandas as pd
import math
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import re

class FingerDataPlotter():
    def boxplot(self, df: pd.DataFrame) -> None:
        df = df.copy()
        
        cols = [c for c in df.columns if 'Correct_Sequences' in c or 'Keystrokes' in c]
        num_plots = len(cols)

        # Set grid size
        cols_per_row = 2  # Adjust this to control layout
        rows = math.ceil(num_plots / cols_per_row)

        # Create subplots
        fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(6.5 * cols_per_row, 5.5 * rows))
        axes = axes.flatten()  # Flatten in case of multi-row layout-

        # Plot each boxplot
        for i, col in enumerate(cols):
            ax = axes[i]
            sns.boxplot(x ='Test', y=col, data=df, ax=ax)
            sns.stripplot(x = 'Test', y=col, data=df, color='black', size=6, jitter=True, ax=ax)
            ax.set_title(f'{col}', fontsize=16)
            # Show y-label only on first column of each row to avoid label overlap.
            if i % cols_per_row == 0:
                ax.set_ylabel('Value', fontsize=14)
            else:
                ax.set_ylabel('')

            ax.tick_params(axis='y', labelsize=12)

            # Set y-axis limits
            if 'Correct_Sequences' in col:
                ax.set_ylim(-2, 35)
            elif 'Keystrokes' in col:
                ax.set_ylim(-10, 180)

        # Remove empty subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout(pad=2.0, w_pad=3.0, h_pad=2.5)
        plt.show()

class StateDataPlotter():
    def learning_curve(self, df: pd.DataFrame) -> None:
        df = df.copy()

        # Ensure transitions are list-like before exploding.
        df['transitions'] = df['transitions'].apply(
            lambda x: x if isinstance(x, list) else []
        )

        exploded = df.explode('transitions')
        exploded['onset_to_onset'] = exploded['transitions'].apply(
            lambda item: item.get('onset_to_onset') if isinstance(item, dict) else np.nan
        )
        exploded['frequency'] = exploded['transitions'].apply(
            lambda item: item.get('frequency') if isinstance(item, dict) else np.nan
        )
        exploded['onset_h'] = exploded['onset_to_onset'].where(exploded['frequency'].eq('h'))
        exploded['onset_s'] = exploded['onset_to_onset'].where(exploded['frequency'].eq('s'))

        # Keep only tests relevant for learning curves (B1-B8).
        exploded['block'] = exploded['Test'].astype(str).str.lower().str.strip()
        exploded = exploded[exploded['block'].str.fullmatch(r'b[1-8]')]

        if exploded.empty:
            print("No block tests (B1-B8) with transitions available for plotting.")
            return

        participant_means = (
            exploded.groupby(['Participant_ID', 'block'], as_index=False)
            .agg(
                mean_onset_total=('onset_to_onset', 'mean'),
                mean_onset_h=('onset_h', 'mean'),
                mean_onset_s=('onset_s', 'mean'),
            )
        )

        block_order = [f'b{i}' for i in range(1, 9)]
        participant_means['block'] = pd.Categorical(
            participant_means['block'],
            categories=block_order,
            ordered=True,
        )
        participant_means = participant_means.sort_values(['Participant_ID', 'block'])

        def summarize_with_ci(df_in: pd.DataFrame, value_col: str, out_prefix: str) -> pd.DataFrame:
            summary = (
                df_in.groupby('block', as_index=False)[value_col]
                .agg(['mean', 'std', 'count'])
                .reset_index()
            )
            summary[f'{out_prefix}_mean'] = summary['mean']
            summary[f'{out_prefix}_ci95'] = 1.96 * (summary['std'] / np.sqrt(summary['count']))
            summary[f'{out_prefix}_ci95'] = summary[f'{out_prefix}_ci95'].fillna(0.0)
            return summary[['block', f'{out_prefix}_mean', f'{out_prefix}_ci95']]

        group_total = summarize_with_ci(participant_means, 'mean_onset_total', 'total')
        group_h = summarize_with_ci(participant_means, 'mean_onset_h', 'h')
        group_s = summarize_with_ci(participant_means, 'mean_onset_s', 's')

        group_means = (
            group_total
            .merge(group_h, on='block', how='outer')
            .merge(group_s, on='block', how='outer')
            .sort_values('block')
        )

        plt.figure(figsize=(12, 6))
        participant_colors = plt.cm.tab20(np.linspace(0, 1, max(1, participant_means['Participant_ID'].nunique())))

        for idx, (participant, one_participant) in enumerate(participant_means.groupby('Participant_ID')):
            plt.plot(
                one_participant['block'].cat.codes,
                one_participant['mean_onset_total'],
                linewidth=0.8,
                alpha=0.35,
                color=participant_colors[idx % len(participant_colors)],
                zorder=2,
            )

        plt.errorbar(
            group_means['block'].cat.codes,
            group_means['total_mean'],
            yerr=group_means['total_ci95'],
            color='black',
            linewidth=2.5,
            marker='o',
            label='Group mean (total)',
            capsize=5,
            elinewidth=1.5,
            zorder=3,
        )

        plt.errorbar(
            group_means['block'].cat.codes,
            group_means['h_mean'],
            yerr=group_means['h_ci95'],
            color='tab:blue',
            linewidth=2.0,
            marker='o',
            label='Group mean (h)',
            capsize=5,
            elinewidth=1.3,
            zorder=3,
        )

        plt.errorbar(
            group_means['block'].cat.codes,
            group_means['s_mean'],
            yerr=group_means['s_ci95'],
            color='tab:orange',
            linewidth=2.0,
            marker='o',
            label='Group mean (s)',
            capsize=5,
            elinewidth=1.3,
            zorder=3,
        )

        plt.xticks(range(len(block_order)), [b.upper() for b in block_order])
        plt.xlabel('Block')
        plt.ylabel('Mean onset-to-onset (s)')
        plt.title('Learning Curve (B1-B8)')
        plt.grid(axis='y', alpha=0.2)
        plt.legend()
        plt.tight_layout()
        plt.show()




