import pandas as pd
import math
import matplotlib.pyplot as plt 
import seaborn as sns

class FingerDataPlotter():
    def boxplot(self, df: pd.DataFrame) -> None:
        
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
    pass