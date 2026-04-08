# 10-fragment_model_statistics.py
"""
Complete statistical characterization for 10-fragment mechanistic model
Calculates coefficient standard errors, confidence intervals, and model performance metrics
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import scipy.stats as stats

def calculate_statistical_metrics(X, y, fragment_names, n_outer_folds=5, random_state=42):
    """
    Calculate comprehensive statistical metrics for mechanistic model
    
    Parameters:
    X: fragment matrix (306 x 10)
    y: target values (logD_lip_w_pH74)
    fragment_names: list of 10 fragment names
    """
    
    # Initialize storage for results across CV folds
    all_coefs = []
    all_intercepts = []
    train_scores = []
    test_scores = []
    test_rmse = []
    test_mae = []
    
    # Outer CV for unbiased performance estimation
    outer_cv = KFold(n_splits=n_outer_folds, shuffle=True, random_state=random_state)
    
    print("Running nested cross-validation for statistical characterization...")
    
    for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Inner CV for alpha tuning
        inner_cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
        
        # Pipeline with standardization
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # LASSO with internal CV
        lasso = LassoCV(cv=inner_cv, alphas=np.logspace(-4, 1, 50), max_iter=10000, random_state=random_state)
        lasso.fit(X_train_scaled, y_train)
        
        # Store coefficients and performance
        all_coefs.append(lasso.coef_)
        all_intercepts.append(lasso.intercept_)
        train_scores.append(r2_score(y_train, lasso.predict(X_train_scaled)))
        test_scores.append(r2_score(y_test, lasso.predict(X_test_scaled)))
        test_rmse.append(np.sqrt(mean_squared_error(y_test, lasso.predict(X_test_scaled))))
        test_mae.append(mean_absolute_error(y_test, lasso.predict(X_test_scaled)))
        
        print(f"  Fold {fold}: R²_train={train_scores[-1]:.3f}, Q²_EXT={test_scores[-1]:.3f}")
    
    # Calculate coefficient statistics
    coef_array = np.array(all_coefs)
    coef_means = np.mean(coef_array, axis=0)
    coef_stds = np.std(coef_array, axis=0)
    
    # Calculate 95% confidence intervals
    t_critical = stats.t.ppf(0.975, df=n_outer_folds-1)
    coef_ci_lower = coef_means - t_critical * coef_stds / np.sqrt(n_outer_folds)
    coef_ci_upper = coef_means + t_critical * coef_stds / np.sqrt(n_outer_folds)
    
    # Calculate p-values (two-tailed t-test against zero)
    t_stats = coef_means / (coef_stds / np.sqrt(n_outer_folds))
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n_outer_folds-1))
    
    # Model performance statistics
    model_stats = {
        'R2_train_mean': np.mean(train_scores),
        'R2_train_std': np.std(train_scores),
        'Q2_EXT_mean': np.mean(test_scores),
        'Q2_EXT_std': np.std(test_scores),
        'RMSE_mean': np.mean(test_rmse),
        'RMSE_std': np.std(test_rmse),
        'MAE_mean': np.mean(test_mae),
        'MAE_std': np.std(test_mae),
        'n_folds': n_outer_folds
    }
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Fragment': fragment_names,
        'Coefficient': coef_means,
        'Std_Error': coef_stds / np.sqrt(n_outer_folds),
        'CI_Lower_95': coef_ci_lower,
        'CI_Upper_95': coef_ci_upper,
        'p_value': p_values,
        'Significant_p05': p_values < 0.05,
        'Significant_p01': p_values < 0.01
    })
    
    return results_df, model_stats

def load_and_prepare_data(csv_file='liposome_fragments.csv'):
    """Load your specific 10-fragment data"""
    df = pd.read_csv(csv_file, sep=',')
    
    # Find target column
    target_col = None
    for col in df.columns:
        if 'logd' in col.lower() and ('ph' in col.lower() or '74' in col):
            target_col = col
            break
    
    y = pd.to_numeric(df[target_col], errors='coerce').values
    
    # Your 10 mechanistic fragments
    mechanistic_fragments = [
        'p3.Cl1C__Cl1.11',
        'p1.O__',
        'p3.CB2CB2OA1.41', 
        'p2.CB2CB_.4',
        'p1.___',
        'p3.CB1C__Cl_.41',
        'p2.N__N__.1',
        'p4.CB1CB2CB2NA2.441',
        'p2.CB_F__.1',
        'p5.Br1CB2CB2CB2CB2.1444'
    ]
    
    X = df[mechanistic_fragments].values
    return X, y, mechanistic_fragments

# Main execution
if __name__ == "__main__":
    # Load your data
    X, y, fragment_names = load_and_prepare_data('liposome_fragments.csv')
    
    # Calculate statistical metrics
    coefficient_results, model_performance = calculate_statistical_metrics(X, y, fragment_names)
    
    # Save results
    coefficient_results.to_csv('statistical_characterization.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("STATISTICAL CHARACTERIZATION COMPLETE")
    print("="*80)
    print("\nModel Performance:")
    print(f"  R²_train: {model_performance['R2_train_mean']:.3f} ± {model_performance['R2_train_std']:.3f}")
    print(f"  Q²_EXT:   {model_performance['Q2_EXT_mean']:.3f} ± {model_performance['Q2_EXT_std']:.3f}")
    print(f"  RMSE:     {model_performance['RMSE_mean']:.3f} ± {model_performance['RMSE_std']:.3f}")
    print(f"  MAE:      {model_performance['MAE_mean']:.3f} ± {model_performance['MAE_std']:.3f}")
    
    print("\nCoefficient Statistics:")
    print(coefficient_results[['Fragment', 'Coefficient', 'Std_Error', 'p_value']].to_string(index=False))
    
    print(f"\nSignificant fragments (p < 0.05): {coefficient_results['Significant_p05'].sum()}/10")
    print(f"Highly significant fragments (p < 0.01): {coefficient_results['Significant_p01'].sum()}/10")