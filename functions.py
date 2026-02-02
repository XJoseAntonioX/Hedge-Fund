import pyarrow.parquet as pq
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def _clip01(x: float) -> float:
    return float(np.minimum(np.maximum(x, 0.0), 1.0))

def weighted_rmse_score(y_target, y_pred, w) -> float:
    denom = np.sum(w * y_target ** 2)
    ratio = np.sum(w * (y_target - y_pred) ** 2) / denom
    clipped = _clip01(ratio)
    val = 1.0 - clipped
    return float(np.sqrt(val))

def read_first_n_rows(file_path: str, n: int, columns=None) -> pd.DataFrame:
    """
    Efficiently read the first n rows from a Parquet file using row groups.

    Parameters:
    - file_path: Path to the Parquet file.
    - n: Number of rows to read.
    - columns: List of columns to read (optional, default reads all columns).

    Returns:
    - Pandas DataFrame with the first n rows.
    """
    pf = pq.ParquetFile(file_path)
    dfs = []
    rows_accum = 0

    for rg_index in range(pf.num_row_groups):
        rg = pf.read_row_group(rg_index, columns=columns).to_pandas()
        
        # Take only remaining rows if exceeding n
        if rows_accum + len(rg) > n:
            rg = rg.iloc[: n - rows_accum]
        
        dfs.append(rg)
        rows_accum += len(rg)
        
        if rows_accum >= n:
            break

    df = pd.concat(dfs, ignore_index=True)
    return df

def analyze_target_correlation(features, target, feature_prefix='Feature', min_corr=0.4, max_corr=1.0):
    """
    Analyze and visualize Pearson and Spearman correlations between features and target.
    
    Parameters:
    - features: numpy array or pandas DataFrame with feature columns
    - target: pandas Series or numpy array with target values
    - feature_prefix: str, prefix for feature names (default: 'Feature')
    - min_corr: float, minimum absolute correlation to display (default: 0.4)
    - max_corr: float, maximum absolute correlation to display (default: 1.0)
    
    Returns:
    - dict with 'pearson' and 'spearman' correlation series (filtered and sorted)
    """
    # Convert features to DataFrame if it's a numpy array
    if isinstance(features, np.ndarray):
        feature_names = [f"{feature_prefix}_{i+1}" for i in range(features.shape[1])]
        features_df = pd.DataFrame(features, columns=feature_names)
    else:
        features_df = features
    
    # Calculate Pearson correlation with target
    pearson_corr = features_df.corrwith(target, method='pearson')
    
    # Calculate Spearman correlation with target
    spearman_corr = features_df.corrwith(target, method='spearman')
    
    # Filter and sort correlations: keep only absolute values in specified range
    def filter_and_sort(corr_series):
        filtered = corr_series[np.abs(corr_series).between(min_corr, max_corr)]
        return filtered.reindex(filtered.abs().sort_values(ascending=False).index)
    
    pearson_filtered = filter_and_sort(pearson_corr)
    spearman_filtered = filter_and_sort(spearman_corr)
    
    # Create visualization if there are filtered correlations
    if len(pearson_filtered) > 0 or len(spearman_filtered) > 0:
        fig, axes = plt.subplots(2, 1, figsize=(12, max(6, len(pearson_filtered) * 0.4)))
        
        if len(pearson_filtered) > 0:
            # Pearson heatmap
            pearson_data = pearson_filtered.values.reshape(-1, 1)
            sns.heatmap(pearson_data, annot=True, fmt='.3f', cmap='coolwarm', 
                        center=0, vmin=-1, vmax=1, ax=axes[0], 
                        yticklabels=pearson_filtered.index, xticklabels=['y_target'],
                        cbar_kws={'label': 'Correlation'})
            axes[0].set_title(f'Pearson Correlation with y_target (|r| ∈ [{min_corr}, {max_corr}], sorted)', 
                              fontsize=14, fontweight='bold')
        else:
            axes[0].text(0.5, 0.5, f'No Pearson correlations found in range [{min_corr}, {max_corr}]', 
                        ha='center', va='center', fontsize=12)
            axes[0].set_xticks([])
            axes[0].set_yticks([])
        
        if len(spearman_filtered) > 0:
            # Spearman heatmap
            spearman_data = spearman_filtered.values.reshape(-1, 1)
            sns.heatmap(spearman_data, annot=True, fmt='.3f', cmap='coolwarm', 
                        center=0, vmin=-1, vmax=1, ax=axes[1], 
                        yticklabels=spearman_filtered.index, xticklabels=['y_target'],
                        cbar_kws={'label': 'Correlation'})
            axes[1].set_title(f'Spearman Correlation with y_target (|ρ| ∈ [{min_corr}, {max_corr}], sorted)', 
                              fontsize=14, fontweight='bold')
        else:
            axes[1].text(0.5, 0.5, f'No Spearman correlations found in range [{min_corr}, {max_corr}]', 
                        ha='center', va='center', fontsize=12)
            axes[1].set_xticks([])
            axes[1].set_yticks([])
        
        plt.tight_layout()
        plt.show()
    else:
        print(f'No meaningful correlations with y_target found in range [{min_corr}, {max_corr}]')
    
    return {'pearson': pearson_filtered, 'spearman': spearman_filtered}