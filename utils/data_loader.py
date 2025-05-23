import pandas as pd
import numpy as np
import torch

def load_polymer_data(csv_file_path, log_transform=True, use_qchem_features=True):
    """
    Load polymer data and quantum chemistry features
    
    Parameters:
    - csv_file_path: Path to CSV file with polymer data
    - log_transform: Whether to apply log transformation to conductivity
    - use_qchem_features: Whether to extract quantum chemistry features
    
    Returns:
    - polymer_smiles: List of polymer SMILES strings
    - conductivities: List of conductivity values
    - qchem_features: (optional) Numpy array of quantum chemistry features
    """
    # Read data from CSV
    data = pd.read_csv(csv_file_path)
    
    # Extract SMILES and conductivity
    polymer_smiles = data['SMILES'].tolist()
    
    # Use 'Conductivity (20%)' column as target
    conductivity_col = None
    for col in ['Conductivity (20%)', 'Conductivity(20%)', 'Conductivity']:
        if col in data.columns:
            conductivity_col = col
            break
    
    if conductivity_col:
        conductivities = data[conductivity_col].tolist()
    elif 'Log(20%)' in data.columns:
        # Fallback to Log(20%)
        print("Warning: Conductivity column not found, using 'Log(20%)' column")
        conductivities = data['Log(20%)'].tolist()
        log_transform = False  # Already in log form
    else:
        raise ValueError("No conductivity or log conductivity column found in CSV")
    
    if log_transform:
        # Apply log transformation to conductivity values
        # Add small constant to avoid log(0)
        conductivities = [np.log10(c + 1e-6) for c in conductivities]
        print("Applied log10 transformation to conductivity values")
        
        # Print statistics of transformed values
        transformed_values = np.array(conductivities)
        print(f"Log-transformed conductivity range: [{np.min(transformed_values):.4f}, {np.max(transformed_values):.4f}]")
        print(f"Log-transformed conductivity mean: {np.mean(transformed_values):.4f}")
    
    print(f"Loaded {len(polymer_smiles)} polymer samples with conductivity values")
    print(f"Sample SMILES: {polymer_smiles[0]}")
    print(f"Sample conductivity: {conductivities[0]}")
    
    # Extract quantum chemistry features if requested
    qchem_features = None
    if use_qchem_features:
        # Check which quantum chemistry features are available
        qchem_cols = []
        potential_cols = ['HOMO', 'LUMO', 'HOCO', 'LUCO', 'Indirect Gap', 'EA/eV', 'HOMO-LUMO Gap']
        
        for col in potential_cols:
            if col in data.columns:
                qchem_cols.append(col)
        
        if qchem_cols:
            qchem_features = data[qchem_cols].values
            
            # Check and handle NaN values
            if np.isnan(qchem_features).any():
                print(f"Warning: NaN values found in quantum chemistry features. Replacing with column means.")
                col_means = np.nanmean(qchem_features, axis=0)
                for i in range(qchem_features.shape[1]):
                    mask = np.isnan(qchem_features[:, i])
                    qchem_features[mask, i] = col_means[i]
            
            print(f"Extracted {len(qchem_cols)} quantum chemistry features: {qchem_cols}")
            # Print feature statistics
            print(f"Feature ranges:")
            for i, col in enumerate(qchem_cols):
                feat_min = np.min(qchem_features[:, i])
                feat_max = np.max(qchem_features[:, i])
                print(f"  {col}: [{feat_min:.4f}, {feat_max:.4f}]")
        else:
            print("No quantum chemistry features found in dataset")
    
    if qchem_features is not None:
        return polymer_smiles, conductivities, qchem_features
    else:
        return polymer_smiles, conductivities

        