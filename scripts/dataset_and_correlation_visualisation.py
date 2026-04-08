# dataset_and_correlation_visualisation.py
"""
Complete standalone script to generate enhanced observed vs predicted plot
Works with your liposome_fragments.csv file
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_and_prepare_data(csv_file='liposome_fragments.csv'):
    """Load your CSV and prepare data for plotting"""
    df = pd.read_csv(csv_file, sep=',')
    
    # Find target column
    target_col = None
    for col in df.columns:
        if 'logd' in col.lower() and ('ph' in col.lower() or '74' in col):
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Target column not found in CSV")
    
    y = pd.to_numeric(df[target_col], errors='coerce').values
    
    # Your actual coefficients from stability selection results
    coefficients = {
        'p3.Cl1C__Cl1.11': 0.090432,
        'p1.O__': -0.489446,
        'p3.CB2CB2OA1.41': -0.027611,
        'p2.CB2CB_.4': 1.143418,
        'p1.___': 0.352844,
        'p3.CB1C__Cl_.41': 0.210006,
        'p2.N__N__.1': 0.070343,
        'p4.CB1CB2CB2NA2.441': -0.085260,
        'p2.CB_F__.1': -0.099838,
        'p5.Br1CB2CB2CB2CB2.1444': 0.056405
    }
    
    # Extract fragment columns that exist in your data
    available_fragments = [frag for frag in coefficients.keys() if frag in df.columns]
    X = df[available_fragments].values
    
    # Get coefficient vector in same order as X columns
    coef_vector = np.array([coefficients[frag] for frag in available_fragments])
    
    return X, y, coef_vector, available_fragments, df

def classify_compounds(df, available_fragments):
    """Classify compounds into chemical classes"""
    classes = []
    for idx, row in df.iterrows():
        # Extract key fragment values for classification
        p1_underscore = row['p1.___'] if 'p1.___' in available_fragments else 0
        p2_cb2cb = row['p2.CB2CB_.4'] if 'p2.CB2CB_.4' in available_fragments else 0
        p3_cb2cb2oa = row['p3.CB2CB2OA1.41'] if 'p3.CB2CB2OA1.41' in available_fragments else 0
        p2_n_nn = row['p2.N__N__.1'] if 'p2.N__N__.1' in available_fragments else 0
        
        # Classification logic
        if p3_cb2cb2oa > 0:  # Phenolic OH present
            classes.append('phenols')
        elif p2_n_nn > 0:  # Nitrogen-nitrogen bonds (pharmaceuticals)
            classes.append('pharmaceuticals')
        elif p2_cb2cb > 2:  # High aromatic content (PCBs, PBDEs)
            classes.append('halogenated_aromatics')
        elif p1_underscore > 10:  # Large aliphatic chains
            classes.append('neutral_hydrocarbons')
        elif 'F' in str(row.name) or 'fluoro' in str(row.name).lower():  # PFAS
            classes.append('PFAS')
        else:
            classes.append('other')
    
    return classes

def create_enhanced_plot():
    """Create the enhanced observed vs predicted plot"""
    try:
        # Load data
        X, y, coef_vector, available_fragments, df = load_and_prepare_data('liposome_fragments.csv')
        
        # Standardize features and calculate predictions
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        y_pred = X_scaled @ coef_vector + np.mean(y)  # Add intercept approximation
        
        # Classify compounds
        compound_classes = classify_compounds(df, available_fragments)
        
        # Color mapping
        class_colors = {
            'neutral_hydrocarbons': '#1f77b4',    # Blue
            'halogenated_aromatics': '#d62728',   # Red  
            'phenols': '#2ca02c',                 # Green
            'pharmaceuticals': '#9467bd',         # Purple
            'PFAS': '#ff7f0e',                    # Orange
            'other': '#7f7f7f'                    # Gray
        }
        
        colors = [class_colors.get(cls, 'gray') for cls in compound_classes]
        
        # Calculate statistics
        r2_val = np.corrcoef(y, y_pred)[0,1]**2
        rmse = np.sqrt(np.mean((y - y_pred)**2))
        
        # Create enhanced plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        scatter = ax.scatter(y, y_pred, c=colors, alpha=0.6, s=40, edgecolors='none')
        
        # Perfect prediction line
        ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, alpha=0.8)
        
        # Add class labels at strategic positions
        unique_classes = ['neutral_hydrocarbons', 'halogenated_aromatics', 
                         'phenols', 'pharmaceuticals', 'PFAS']
        
        # Position labels to avoid overlapping data
        label_positions = [
            (y.min() + 0.05*(y.max()-y.min()), y.max() - 0.05*(y.max()-y.min())),  # Top-left
            (y.max() - 0.05*(y.max()-y.min()), y.max() - 0.05*(y.max()-y.min())),  # Top-right
            (y.min() + 0.05*(y.max()-y.min()), y.min() + 0.05*(y.max()-y.min())),  # Bottom-left  
            (y.max() - 0.05*(y.max()-y.min()), y.min() + 0.05*(y.max()-y.min())),  # Bottom-right
            ((y.min() + y.max())/2, (y.min() + y.max())/2)                        # Center
        ]
        
 #       for i, cls in enumerate(unique_classes):
 #           if cls in compound_classes:
 #               color = class_colors[cls]
 #               ax.text(label_positions[i][0], label_positions[i][1], 
 #                      cls.replace('_', ' ').title(),
 #                      color=color, fontweight='bold', fontsize=11,
 #                      bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9,
 #                              edgecolor=color, linewidth=2))
        
        # Formatting
        ax.set_xlabel('Observed logD$_{lip/w}$ (pH 7.4)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Predicted logD$_{lip/w}$ (pH 7.4)', fontsize=14, fontweight='bold')
        ax.set_title(f'Liposome-Water Partitioning Model Performance\nR² = {r2_val:.3f}, RMSE = {rmse:.3f}', 
                    fontsize=16, fontweight='bold', pad=20)
        
        ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cls.replace('_', ' ').title()) 
                          for cls, color in class_colors.items() if cls in compound_classes]
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='lower right', 
                      title='Compound Classes', title_fontsize=12, fontsize=10,
                      framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        
        # Save high-quality versions
        plt.savefig('enhanced_observed_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.savefig('enhanced_observed_vs_predicted.pdf', bbox_inches='tight')
        
        print("✅ Enhanced plot generated successfully!")
        print("📁 Files saved:")
        print("   • enhanced_observed_vs_predicted.png (300 DPI)")
        print("   • enhanced_observed_vs_predicted.pdf (vector format)")
        print(f"📊 Model performance: R² = {r2_val:.3f}, RMSE = {rmse:.3f}")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure 'liposome_fragments.csv' is in the same directory")
        print("2. Verify the CSV uses commas as separators")
        print("3. Check that fragment names match exactly (case-sensitive)")
        print("4. Ensure the target column contains numeric values")

# Run the plot generation
if __name__ == "__main__":
    create_enhanced_plot()
