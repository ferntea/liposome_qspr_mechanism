# 10-fragment_model.py
"""
Final 10-Fragment Mechanistic Model
Builds interpretable model using only mechanistically meaningful fragments
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

def build_mechanistic_model(csv_file='liposome_fragments.csv'):
    """Build final 10-fragment mechanistic model"""
    
    # Load data
    df = pd.read_csv(csv_file, sep=',')
    
    # Find target column
    target_col = None
    for col in df.columns:
        if 'logd' in col.lower() and ('ph' in col.lower() or '74' in col):
            target_col = col
            break
    
    y = pd.to_numeric(df[target_col], errors='coerce').values
    
    # Define mechanistic fragment set (10 key fragments)
    mechanistic_fragments = [
        'p1.O__',           # Ether/carbonyl oxygen → H-bond penalty
        'p1.___',           # Aliphatic carbon count → hydrophobic matching  
        'p2.CB2CB_.4',      # Aromatic C–C → π-stacking
        'p3.CB1C__Cl_.41',  # ortho-Chlorophenol → steric hindrance
        'p3.CB2CB2OA1.41',  # Phenolic OH → H-bond donation penalty
        'p2.CB_F__.1',      # Aromatic C–F → weak halogen bonding
        'p2.N__N__.1',      # Aliphatic N–N → ionization effects
        'p3.Cl1C__Cl1.11',  # para-Dichloro pattern → optimal halogen bonding
        'p4.CB1CB2CB2NA2.441', # Aromatic amine → combined effects
        'p5.Br1CB2CB2CB2CB2.1444' # Brominated aromatics → enhanced halogen bonding
    ]
    
    # Extract mechanistic fragment matrix
    X_mech = df[mechanistic_fragments].values
    fragment_names = mechanistic_fragments
    
    print(f"Building model with {len(fragment_names)} mechanistic fragments")
    print("Fragments:", fragment_names)
    
    # Train/test split (80/20, matching Lin et al.)
    X_train, X_test, y_train, y_test = train_test_split(
        X_mech, y, test_size=0.20, random_state=42
    )
    
    # Build pipeline with LASSO
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lasso', LassoCV(cv=5, alphas=np.logspace(-4, 1, 50), max_iter=10000))
    ])
    
    # Fit model
    pipeline.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)
    
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    print(f"\nModel Performance:")
    print(f"R²_train: {r2_train:.3f}")
    print(f"Q²_EXT:   {r2_test:.3f}")
    print(f"RMSE:     {rmse_test:.3f}")
    
    # Get coefficients
    coefficients = pipeline.named_steps['lasso'].coef_
    intercept = pipeline.named_steps['lasso'].intercept_
    
    # Create results DataFrame
    results_df = pd.DataFrame({
        'Fragment': fragment_names,
        'Coefficient': coefficients,
        'Abs_Coefficient': np.abs(coefficients)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\nFragment Coefficients:")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('mechanistic_model_coefficients.csv', index=False)
    
    # Plot observed vs predicted
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, alpha=0.7, s=30)
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
    plt.xlabel('Observed logD_lip/w (pH 7.4)')
    plt.ylabel('Predicted logD_lip/w (pH 7.4)')
    plt.title(f'Mechanistic Model Performance\nQ²_EXT = {r2_test:.3f}, RMSE = {rmse_test:.3f}')
    plt.grid(True, alpha=0.3)
    plt.savefig('mechanistic_model_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'pipeline': pipeline,
        'coefficients': coefficients,
        'intercept': intercept,
        'performance': {'R2_train': r2_train, 'Q2_EXT': r2_test, 'RMSE': rmse_test},
        'fragments': fragment_names,
        'data': {'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}
    }

# Run the model
if __name__ == "__main__":
    model_results = build_mechanistic_model('liposome_fragments.csv')