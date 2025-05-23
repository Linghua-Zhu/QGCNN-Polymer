import numpy as np
import networkx as nx
import torch
import math
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

def smiles_to_mol(smiles):
    """
    Convert SMILES string to RDKit molecule object
    
    Parameters:
    - smiles: SMILES string
    
    Returns:
    - mol: RDKit molecule object
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Failed to parse SMILES: {smiles}")
        return None
    
    # Add hydrogen atoms
    mol = Chem.AddHs(mol)
    
    # Generate 3D conformation
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)  # Force field optimization
    except:
        print(f"Failed to generate 3D conformation: {smiles}")
        return None
    
    return mol

def mol_to_graph(mol):
    """
    Convert RDKit molecule object to NetworkX graph
    
    Parameters:
    - mol: RDKit molecule object
    
    Returns:
    - G: NetworkX graph
    """
    if mol is None:
        return None
    
    G = nx.Graph()
    
    # Add nodes
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = atom.GetIsAromatic()
        
        G.add_node(atom_idx, 
                   symbol=atom_symbol,
                   atomic_num=atomic_num,
                   formal_charge=formal_charge,
                   hybridization=int(hybridization),
                   is_aromatic=int(is_aromatic))
    
    # Add edges
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = bond.GetBondTypeAsDouble()
        is_conjugated = int(bond.GetIsConjugated())
        is_in_ring = int(bond.IsInRing())
        
        G.add_edge(i, j, 
                   bond_type=bond_type,
                   is_conjugated=is_conjugated,
                   is_in_ring=is_in_ring)
    
    return G

def mol_to_fully_connected_graph(mol):
    """
    Convert RDKit molecule object to fully connected NetworkX graph
    
    Parameters:
    - mol: RDKit molecule object
    
    Returns:
    - G: NetworkX graph (fully connected)
    """
    if mol is None:
        return None
    
    G = nx.Graph()
    
    # Add nodes (same as original)
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_symbol = atom.GetSymbol()
        atomic_num = atom.GetAtomicNum()
        formal_charge = atom.GetFormalCharge()
        hybridization = atom.GetHybridization()
        is_aromatic = atom.GetIsAromatic()
        
        G.add_node(atom_idx, 
                   symbol=atom_symbol,
                   atomic_num=atomic_num,
                   formal_charge=formal_charge,
                   hybridization=int(hybridization),
                   is_aromatic=int(is_aromatic))
    
    # Add edges (fully connected with distance-based features)
    # Get 3D coordinates for all atoms
    conformer = mol.GetConformer()
    
    # Connect all atoms with weighted edges based on distance
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        pos_i = conformer.GetAtomPosition(i)
        for j in range(i+1, num_atoms):
            pos_j = conformer.GetAtomPosition(j)
            
            # Calculate Euclidean distance
            distance = math.sqrt((pos_i.x - pos_j.x)**2 + 
                                (pos_i.y - pos_j.y)**2 + 
                                (pos_i.z - pos_j.z)**2)
            
            # Check if there's a bond (for edge features)
            bond = mol.GetBondBetweenAtoms(i, j)
            is_bond = bond is not None
            
            if is_bond:
                bond_type = bond.GetBondTypeAsDouble()
                is_conjugated = int(bond.GetIsConjugated())
                is_in_ring = int(bond.IsInRing())
            else:
                bond_type = 0.0  # No bond
                is_conjugated = 0
                is_in_ring = 0
            
            # Add edge with distance and other features
            G.add_edge(i, j, 
                       distance=distance,
                       bond_type=bond_type,
                       is_bond=int(is_bond),
                       is_conjugated=is_conjugated,
                       is_in_ring=is_in_ring)
    
    return G

def extract_node_features(G):
    """
    Extract node features from NetworkX graph
    
    Parameters:
    - G: NetworkX graph
    
    Returns:
    - Node feature matrix
    """
    if G is None:
        return None
    
    # One-hot encoding for common atom types
    atom_types = {'C': 0, 'H': 1, 'N': 2, 'O': 3, 'S': 4, 'F': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'P': 9}
    
    features = []
    for node, data in G.nodes(data=True):
        # Atom type one-hot encoding
        atom_type = [0] * len(atom_types)
        if data['symbol'] in atom_types:
            atom_type[atom_types[data['symbol']]] = 1
        
        # Add other features
        other_features = [
            data['atomic_num'],
            data['formal_charge'],
            data['hybridization'],
            data['is_aromatic']
        ]
        
        # Combine features
        node_features = atom_type + other_features
        features.append(node_features)
    
    return np.array(features, dtype=np.float32)

def extract_edge_features(G):
    """
    Extract edge features from NetworkX graph
    
    Parameters:
    - G: NetworkX graph with edge attributes
    
    Returns:
    - Edge feature matrix
    """
    if G is None:
        return None
    
    edge_features = []
    for u, v, data in G.edges(data=True):
        # For fully connected graph
        if 'distance' in data:
            features = [
                data['distance'],
                data['bond_type'],
                data['is_bond'],
                data['is_conjugated'],
                data['is_in_ring']
            ]
        # For regular graph
        else:
            features = [
                1.0,  # Default distance
                data['bond_type'],
                1,    # Is a bond
                data['is_conjugated'],
                data['is_in_ring']
            ]
        
        edge_features.append(features)
    
    return np.array(edge_features, dtype=np.float32)

def graph_to_pyg_data(G, node_features):
    """
    Convert NetworkX graph and features to PyTorch Geometric data object
    
    Parameters:
    - G: NetworkX graph
    - node_features: Node feature matrix
    
    Returns:
    - data: PyTorch Geometric data object
    """
    if G is None or node_features is None:
        return None
    
    # Convert to PyTorch tensors
    x = torch.tensor(node_features, dtype=torch.float)
    
    # Create edge indices
    edge_index = []
    
    # Extract edge features if available
    has_edge_features = False
    if G.number_of_edges() > 0:
        edge_features = extract_edge_features(G)
        has_edge_features = True
    
    for u, v in G.edges():
        edge_index.append([u, v])
        edge_index.append([v, u])  # Add both directions for undirected graph
    
    if len(edge_index) > 0:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t()
        
        # Double the edge features for undirected graph
        if has_edge_features:
            edge_attr = []
            for u, v, data in G.edges(data=True):
                # For fully connected graph
                if 'distance' in data:
                    feat = [
                        data['distance'],
                        data['bond_type'],
                        data['is_bond'],
                        data['is_conjugated'],
                        data['is_in_ring']
                    ]
                # For regular graph
                else:
                    feat = [
                        1.0,  # Default distance
                        data['bond_type'],
                        1,    # Is a bond
                        data['is_conjugated'],
                        data['is_in_ring']
                    ]
                
                edge_attr.append(feat)
                edge_attr.append(feat)  # Add twice for undirected graph
                
            edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        else:
            edge_attr = None
    else:
        # Handle isolated nodes (no edges)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = None
    
    # Create PyTorch Geometric data object
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr
    )
    
    return data

def smiles_to_pyg_data(smiles, use_fully_connected=False):
    """
    Directly convert SMILES string to PyTorch Geometric data
    
    Parameters:
    - smiles: SMILES string
    - use_fully_connected: Whether to create a fully connected graph
    
    Returns:
    - data: PyTorch Geometric data object
    - G: NetworkX graph
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None, None
    
    if use_fully_connected:
        G = mol_to_fully_connected_graph(mol)
    else:
        G = mol_to_graph(mol)
    
    if G is None:
        return None, None
    
    features = extract_node_features(G)
    data = graph_to_pyg_data(G, features)
    
    return data, G
