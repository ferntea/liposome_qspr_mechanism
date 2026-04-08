# umap_comparison_50_vs_10.py
"""
Fixed UMAP comparison code with proper legend placement
- Left plot (50 descriptors): legend at 'lower right'
- Right plot (10 descriptors): legend at 'upper right'
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from umap import UMAP
from sklearn.preprocessing import StandardScaler

# Set style for publication-quality plots
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})

def load_data_for_umap(csv_file='liposome_fragments.csv'):
    """Load both 50-fragment and 10-fragment datasets"""
    df = pd.read_csv(csv_file, sep=',')
    
    # Find target column
    target_col = None
    for col in df.columns:
        if 'logd' in col.lower() and ('ph' in col.lower() or '74' in col):
            target_col = col
            break
    
    if target_col is None:
        raise ValueError("Target column not found")
    
    y = pd.to_numeric(df[target_col], errors='coerce').values
    
    # Get all 50 fragments (excluding target and Name columns)
    all_columns = [col for col in df.columns if col not in ['Name', target_col]]
    X_50 = df[all_columns].values
    
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
    
    # Handle missing fragments gracefully
    available_10 = [frag for frag in mechanistic_fragments if frag in df.columns]
    X_10 = df[available_10].values
    
    return X_50, X_10, y, df

def classify_compounds_for_umap(df):
    """Classify compounds for coloring in UMAP plots"""
    classes = []
    for idx, row in df.iterrows():
        # Extract key fragment values
        p1_underscore = row['p1.___'] if 'p1.___' in row else 0
        p2_cb2cb = row['p2.CB2CB_.4'] if 'p2.CB2CB_.4' in row else 0
        p3_cb2cb2oa = row['p3.CB2CB2OA1.41'] if 'p3.CB2CB2OA1.41' in row else 0
        p2_n_nn = row['p2.N__N__.1'] if 'p2.N__N__.1' in row else 0
        
        if p3_cb2cb2oa > 0:
            classes.append('phenols')
        elif p2_n_nn > 0:
            classes.append('pharmaceuticals')
        elif p2_cb2cb > 2:
            classes.append('halogenated_aromatics')
        elif p1_underscore > 10:
            classes.append('neutral_hydrocarbons')
        elif 'F' in str(row.name) or 'fluoro' in str(row.name).lower():
            classes.append('PFAS')
        else:
            classes.append('other')
    
    return classes

def create_umap_comparison():
    """Create side-by-side UMAP comparison plots with proper legend placement"""
    try:
        # Load data
        X_50, X_10, y, df = load_data_for_umap('liposome_fragments.csv')
        
        # Classify compounds
        compound_classes = classify_compounds_for_umap(df)
        
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
        
        # Standardize features for UMAP
        scaler_50 = StandardScaler()
        X_50_scaled = scaler_50.fit_transform(X_50)
        
        scaler_10 = StandardScaler()
        X_10_scaled = scaler_10.fit_transform(X_10)
        
        # Create UMAP embeddings
        print("Creating UMAP embedding for 50 fragments...")
        umap_50 = UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
        embedding_50 = umap_50.fit_transform(X_50_scaled)
        
        print("Creating UMAP embedding for 10 fragments...")
        umap_10 = UMAP(n_neighbors=15, min_dist=0.1, random_state=42, n_components=2)
        embedding_10 = umap_10.fit_transform(X_10_scaled)
        
        # Create side-by-side plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Left panel: 50 fragments - legend at lower right
        scatter1 = ax1.scatter(embedding_50[:, 0], embedding_50[:, 1], 
                              c=colors, alpha=0.6, s=40, edgecolors='none')
        ax1.set_title('UMAP: 50 Original NASAWIN Fragments', fontweight='bold', fontsize=14)
        ax1.set_xlabel('UMAP1')
        ax1.set_ylabel('UMAP2')
        ax1.grid(True, alpha=0.3)
        
        # Right panel: 10 mechanistic fragments - legend at upper right  
        scatter2 = ax2.scatter(embedding_10[:, 0], embedding_10[:, 1], 
                              c=colors, alpha=0.6, s=40, edgecolors='none')
        ax2.set_title('UMAP: 10 Mechanistic Fragments', fontweight='bold', fontsize=14)
        ax2.set_xlabel('UMAP1')
        ax2.set_ylabel('UMAP2')
        ax2.grid(True, alpha=0.3)
        
        # Create legend elements
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=color, label=cls.replace('_', ' ').title()) 
                          for cls, color in class_colors.items() if cls in compound_classes]
        
        # Add legends with EXPLICIT positioning
        if legend_elements:
            # Left plot: lower right
            ax1.legend(handles=legend_elements, loc='lower right', 
                      title='Compound Classes', title_fontsize=12, fontsize=10, 
                      framealpha=0.9, edgecolor='black')
            
            # Right plot: upper right  
            ax2.legend(handles=legend_elements, loc='upper right',
                      title='Compound Classes', title_fontsize=12, fontsize=10,
                      framealpha=0.9, edgecolor='black')
        
        plt.tight_layout()
        
        # Save high-quality versions
        plt.savefig('umap_comparison_50_vs_10.png', dpi=300, bbox_inches='tight')
        plt.savefig('umap_comparison_50_vs_10.pdf', bbox_inches='tight')
        
        print("✅ UMAP comparison plots generated successfully!")
        print("📁 Files saved:")
        print("   • umap_comparison_50_vs_10.png (300 DPI)")
        print("   • umap_comparison_50_vs_10.pdf (vector format)")
        
        plt.show()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure 'liposome_fragments.csv' is in the same directory")
        print("2. Verify you have installed UMAP: pip install umap-learn")
        print("3. Check that fragment names match exactly (case-sensitive)")
        print("4. Ensure your CSV uses commas as separators")

# Run the UMAP comparison
if __name__ == "__main__":
    create_umap_comparison()
    