from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd
import numpy as np
import os

def sdf_to_dataframe(sdf_path, sanitize=True):
    """
    Convert SDF file to pandas DataFrame with SMILES and all properties.
    
    Parameters:
    -----------
    sdf_path : str
        Path to input SDF file
    sanitize : bool
        Whether to sanitize molecules (recommended=True)
        
    Returns:
    --------
    df : pd.DataFrame
        DataFrame with SMILES, validity flags, and all SDF properties
    invalid_mols : list
        List of (index, error_msg) for invalid molecules
    """
    supplier = Chem.ForwardSDMolSupplier(sdf_path, sanitize=sanitize, removeHs=False)
    
    data = []
    invalid_mols = []
    idx = 0
    
    for mol in supplier:
        idx += 1
        if mol is None:
            invalid_mols.append((idx, "RDKit could not parse molecule"))
            continue
            
        try:
            # Generate canonical SMILES
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            
            # Extract all SDF properties
            props = {}
            for prop_name in mol.GetPropNames():
                props[prop_name] = mol.GetProp(prop_name)
            
            # Basic molecular descriptors (optional but useful for QC)
            props.update({
                'SMILES': smiles,
                'MolWt': Descriptors.MolWt(mol),
                'HeavyAtomCount': rdMolDescriptors.CalcNumHeavyAtoms(mol),
                'RotatableBondCount': rdMolDescriptors.CalcNumRotatableBonds(mol),
                'HBD': rdMolDescriptors.CalcNumHBD(mol),
                'HBA': rdMolDescriptors.CalcNumHBA(mol),
                'RingCount': rdMolDescriptors.CalcNumRings(mol),
                'Molecule_Index_in_SDF': idx,
                'IsValid': True
            })
            
            data.append(props)
            
        except Exception as e:
            invalid_mols.append((idx, str(e)))
            continue
    
    df = pd.DataFrame(data)
    return df, invalid_mols


# === USAGE ===
sdf_file = "" \
"2019-S_Lin-pH7_4.sdf"  # ← REPLACE with your actual SDF path

if not os.path.exists(sdf_file):
    print(f"❌ File not found: {sdf_file}")
    print("Please verify the filename/path and try again.")
else:
    print(f"✅ Processing: {sdf_file}")
    df, invalid = sdf_to_dataframe(sdf_file)
    
    print(f"\n📊 Summary:")
    print(f"   Total molecules in SDF: {df['Molecule_Index_in_SDF'].max() + len(invalid)}")
    print(f"   Successfully parsed: {len(df)}")
    print(f"   Failed to parse: {len(invalid)}")
    
    if invalid:
        print("\n⚠️  Invalid molecules:")
        for idx, err in invalid[:10]:  # Show first 10 errors
            print(f"   Molecule #{idx}: {err}")
        if len(invalid) > 10:
            print(f"   ... and {len(invalid)-10} more")
    
    # Save results
    output_csv = sdf_file.replace('.sdf', '_decoded.csv')
    df.to_csv(output_csv, index=False)
    print(f"\n💾 Saved decoded data to: {output_csv}")
    
    # Show preview
    print("\n🔍 First 5 records (SMILES + key properties):")
    preview_cols = ['SMILES'] + [col for col in df.columns if col != 'SMILES'][:5]
    print(df[preview_cols].head())