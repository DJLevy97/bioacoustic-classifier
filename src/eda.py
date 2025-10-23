import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_grouped_histograms(
    df, 
    group_col, 
    value_col, 
    bins=30, 
    figsize=(12, 8), 
    alpha=0.6
):
    """
    Plot separate histograms of a quantitative variable grouped by a categorical variable.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the data.
    group_col : str
        Name of the categorical column to group by.
    value_col : str
        Name of the quantitative column to plot.
    bins : int, optional
        Number of histogram bins (default: 30).
    figsize : tuple, optional
        Figure size (default: (12, 8)).
    alpha : float, optional
        Histogram transparency (default: 0.6).
    """
    groups = df[group_col].unique()
    n_groups = len(groups)

    ncols = min(4, n_groups)
    nrows = int(np.ceil(n_groups / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    for i, group in enumerate(groups):
        subset = df[df[group_col] == group]
        axes[i].hist(subset[value_col], bins=bins, alpha=alpha, color='steelblue', edgecolor='black')
        axes[i].set_title(f"{group_col}: {group}")
        axes[i].set_xlabel(value_col)
        axes[i].set_ylabel("Count")

    # Hide unused subplots
    for j in range(i+1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()



def plot_feature_correlation_heatmap(df, feature_cols, figsize=(12, 10), corr_method='pearson', vmin=-1, vmax=1):
    """
    Plots a heatmap of correlations among feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features.
    feature_cols : list
        List of column names to include in the correlation matrix.
    figsize : tuple
        Figure size in inches (default = (12, 10)).
    corr_method : str
        Correlation method ('pearson', 'spearman', or 'kendall').
    vmin, vmax : float
        Limits for the color scale (default = -1 to 1).
    """
    corr = df[feature_cols].corr(method=corr_method)

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr, 
        cmap='coolwarm', 
        center=0, 
        annot=False, 
        fmt=".2f",
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"shrink": 0.8}
    )
    plt.title(f"Feature Correlation Heatmap ({corr_method.capitalize()} method)")
    plt.tight_layout()
    plt.show()