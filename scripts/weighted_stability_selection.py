# weighted_stability_selection.py
"""
Weighted Stability Selection for NASAWIN Fragments
Handles multiple fragment occurrences per molecule as continuous variables
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def load_fragment_data(csv_file):
    """Load your 50 fragments + target variable"""
    df = pd.read_csv(csv_file, sep=',')
    
    # Find target column automatically
    target_col = None
    for col in df.columns:
        if 'logd' in col.lower() and ('ph' in col.lower() or '74' in col):
            target_col = col
            break
    
    if target_col is None:
        raise ValueError(f"Target column not found. Available: {list(df.columns)}")
    
    print(f"✓ Found target: '{target_col}'")
    y = pd.to_numeric(df[target_col], errors='coerce').values
    fragment_cols = [col for col in df.columns if col != target_col and col != 'Name']
    X = df[fragment_cols].values
    return X, y, fragment_cols, df

def weighted_stability_selection(X, y, fragment_names, n_subsamples=1000, subsample_size=0.8):
    """Stability selection treating fragment counts as continuous variables"""
    np.random.seed(42)
    n_samples, n_features = X.shape
    selection_counter = Counter()
    
    print(f"Running weighted stability selection ({n_subsamples} subsamples)...")
    
    for i in range(n_subsamples):
        if (i + 1) % 200 == 0:
            print(f"  Subsample {i+1}/{n_subsamples}")
        
        # Random subsample
        indices = np.random.choice(n_samples, size=int(n_samples * subsample_size), replace=False)
        X_sub, y_sub = X[indices], y[indices]
        
        # Standardize features
        scaler = StandardScaler()
        X_sub_scaled = scaler.fit_transform(X_sub)
        
        # LASSO with internal CV
        lasso = LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), max_iter=10000)
        lasso.fit(X_sub_scaled, y_sub)
        
        # Record non-zero coefficients (continuous treatment)
        non_zero_indices = np.where(np.abs(lasso.coef_) > 0.01)[0]  # Small threshold
        for idx in non_zero_indices:
            selection_counter[fragment_names[idx]] += 1
    
    # Calculate selection frequencies
    selection_freq = {name: count / n_subsamples for name, count in selection_counter.items()}
    results = pd.DataFrame({
        'Fragment': fragment_names,
        'Selection_Frequency': [selection_freq.get(name, 0) for name in fragment_names]
    }).sort_values('Selection_Frequency', ascending=False)
    
    return results

def select_top_fragments(stability_results, max_fragments=12, min_frequency=0.70):
    """Select top fragments with mechanistic relevance"""
    filtered = stability_results[
        (stability_results['Selection_Frequency'] >= min_frequency) & 
        (stability_results.index < max_fragments)
    ]
    return filtered

# Main execution
if __name__ == "__main__":
    # Load your data
    X, y, fragment_names, df = load_fragment_data('liposome_fragments.csv')
    
    # Run weighted stability selection
    stability_results = weighted_stability_selection(X, y, fragment_names)
    
    # Select top fragments
    top_fragments = select_top_fragments(stability_results, max_fragments=12, min_frequency=0.70)
    
    # Save results
    stability_results.to_csv('stability_results.csv', index=False)
    top_fragments.to_csv('top_fragments.csv', index=False)
    
    print("\n✅ Analysis complete!")
    print(f"Top {len(top_fragments)} stable fragments:")
    print(top_fragments.to_string(index=False))