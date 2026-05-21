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
from statsmodels.stats.multicomp import pairwise_tukeyhsd

class FingerDataAnalyser():
    def anova_test(self, df):
        # List of columns to test
        cols_to_test = [col for col in df.columns if ('correct' in col or 'keys' in col)]

        # Run ANOVA for each and collect significant columns
        significant_cols = []
        for col in cols_to_test:
            print(f"\n--- Prüfung: {col} ---")
            
            # Normalverteilung je Gruppe
            for group in df['Category'].unique():
                stat, p = shapiro(df[df['Category'] == group][col])
                print(f"Shapiro-Wilk für {group}: p = {p:.4f}")
            
            # ANOVA
            model = smf.ols(f'{col} ~ C(Category)', data=df).fit()
            anova = sm.stats.anova_lm(model, typ=2)
            p_value = anova['PR(>F)'][0]
            print(f"ANOVA: p = {p_value:.4f}")
            
            if p_value < 0.05:
                print(" → Signifikanter Unterschied! (ANOVA)")
                # Optional: Post-hoc schon erledigt durch Tukey
            else:
                # Wenn ANOVA nicht signifikant oder Normalverteilung fraglich
                # prüfe zusätzlich Mann-Whitney
                group1 = df[df['Category'] == 'Klassisch'][col]
                group2 = df[df['Category'] == 'AR'][col]
                stat, p_mwu = mannwhitneyu(group1, group2, alternative='two-sided')
                print(f"Mann-Whitney-U-Test: p = {p_mwu:.4f}")
        
    
        # # Plot all Tukey HSD results in one figure
        # if significant_cols:
        #     fig, axes = plt.subplots(nrows=len(significant_cols), figsize=(8, 4 * len(significant_cols)))
        #     if len(significant_cols) == 1:
        #         axes = [axes]  # make it iterable
        #     for ax, col in zip(axes, significant_cols):
        #         tukey = pairwise_tukeyhsd(endog=df[col], groups=df['Category'], alpha=0.05)
        #         tukey.plot_simultaneous(ax=ax)
        #         ax.set_title(f'Tukey HSD: {col}')
        #     plt.tight_layout()
        #     plt.show()

        # # --------- plot the results -----------
        # # Select columns to plot
        # cols = [c for c in df.columns if 'correct' in c or 'keys' in c]
        # num_plots = len(cols)

        # # Set grid size
        # cols_per_row = 4  # Adjust this to control layout
        # rows = math.ceil(num_plots / cols_per_row)

        # # Create subplots
        # fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(6 * cols_per_row, 5 * rows))
        # axes = axes.flatten()  # Flatten in case of multi-row layout

        # # Plot each boxplot
        # for i, col in enumerate(cols):
        #     ax = axes[i]
        #     sns.boxplot(x='Category', y=col, data=df, ax=ax)
        #     sns.stripplot(x='Category', y=col, data=df, color='black', size=6, jitter=True, ax=ax)
        #     ax.set_title(f'{col} by Category', fontsize=16)
        #     ax.set_xlabel('', fontsize=16)
        #     ax.set_ylabel('correct sequences', fontsize=14)

        #     ax.tick_params(axis='x', labelsize=16, rotation=45)
        #     ax.tick_params(axis='y', labelsize=16)

        #     # Set y-axis limits
        #     if 'correct' in col:
        #         ax.set_ylim(3, 35)
        #     elif 'keys' in col:
        #         ax.set_ylim(50, 180)

        # # Remove empty subplots
        # for j in range(i + 1, len(axes)):
        #     fig.delaxes(axes[j])

        # plt.tight_layout()
        # plt.show()

        return
