import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.molecular_graph import smiles_to_pyg_data

class PolymerDataset(Dataset):
    """
    Dataset class for polymer conductivity prediction
    """
    def __init__(self, polymer_smiles, conductivities, qchem_features=None, use_fully_connected=False):
        """
        Initialize dataset
        
        Parameters:
        - polymer_smiles: List of polymer SMILES strings
        - conductivities: List of conductivity values
        - qchem_features: (optional) Numpy array of quantum chemistry features
        - use_fully_connected: Whether to use fully connected graphs
        """
        self.polymer_smiles = polymer_smiles
        self.conductivities = conductivities
        self.qchem_features = qchem_features
        self.use_fully_connected = use_fully_connected
        
        # Pre-process all molecules to save time during training
        self.processed_data = []
        
        for i in range(len(polymer_smiles)):
            try:
                # Process polymer with specified graph type
                polymer_data, _ = smiles_to_pyg_data(polymer_smiles[i], 
                                                    use_fully_connected=self.use_fully_connected)
                
                if polymer_data is None:
                    print(f"Warning: Failed to process polymer {i}")
                    continue
                
                # Create data dictionary
                data_dict = {
                    'polymer': polymer_data,
                    'conductivity': conductivities[i]
                }
                
                # Add quantum chemistry features if available
                if qchem_features is not None:
                    data_dict['qchem_features'] = qchem_features[i]
                
                # Store processed data
                self.processed_data.append(data_dict)
            except Exception as e:
                print(f"Error processing polymer {i}: {e}")
    
    def __len__(self):
        """Return dataset size"""
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        """Get a single data sample"""
        return self.processed_data[idx]

def create_adjacency(edge_index, num_nodes, edge_attr=None, use_weighted=True):
    """
    Create adjacency matrix from edge indices with optional edge weights
    
    Parameters:
    - edge_index: Tensor of shape [2, num_edges] containing source and target nodes
    - num_nodes: Number of nodes in the graph
    - edge_attr: Optional tensor of edge attributes
    - use_weighted: Whether to use edge weights (distances) for adjacency
    
    Returns:
    - adj: Weighted or binary adjacency matrix
    """
    adj = torch.zeros(num_nodes, num_nodes)
    
    # If edge attributes are provided and we want weighted adjacency
    if edge_attr is not None and use_weighted and edge_attr.shape[1] >= 1:
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            
            # Use inverse distance as weight (closer = stronger connection)
            # The first feature is assumed to be distance
            distance = edge_attr[i, 0]
            if distance > 0:  # Avoid division by zero
                weight = 1.0 / distance
            else:
                weight = 1.0  # Default weight for zero distance
                
            adj[src, dst] = weight
    else:
        # Binary adjacency matrix (unweighted)
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            adj[src, dst] = 1.0
    
    return adj

def collate_fn(batch):
    """
    Custom collate function for batching molecular graphs
    
    Parameters:
    - batch: List of dictionaries containing polymer and conductivity
    
    Returns:
    - Batched data ready for model input
    """
    # Extract components
    polymers = [item['polymer'] for item in batch]
    conductivities = torch.tensor([item['conductivity'] for item in batch], dtype=torch.float).unsqueeze(1)
    
    # Check if quantum chemistry features exist
    has_qchem = 'qchem_features' in batch[0]
    if has_qchem:
        qchem_features = torch.tensor([item['qchem_features'] for item in batch], dtype=torch.float)
    
    # Process polymers
    polymer_features = [p.x for p in polymers]
    polymer_edge_indices = [p.edge_index for p in polymers]
    
    # Extract edge attributes if available
    has_edge_attr = hasattr(polymers[0], 'edge_attr') and polymers[0].edge_attr is not None
    if has_edge_attr:
        polymer_edge_attrs = [p.edge_attr for p in polymers]
    
    # Create batch
    batch_size = len(batch)
    max_polymer_nodes = max(p.shape[0] for p in polymer_features)
    
    feature_dim = polymer_features[0].shape[1]
    
    # Initialize batch tensors
    batch_features = torch.zeros(batch_size, max_polymer_nodes, feature_dim)
    batch_adj = torch.zeros(batch_size, max_polymer_nodes, max_polymer_nodes)
    
    # Fill batch tensors
    for i in range(batch_size):
        # Get number of nodes
        n_polymer = polymer_features[i].shape[0]
        
        # Add node features
        batch_features[i, :n_polymer] = polymer_features[i]
        
        # Create adjacency matrix with edge attributes if available
        if has_edge_attr:
            polymer_adj = create_adjacency(polymer_edge_indices[i], n_polymer, 
                                           edge_attr=polymer_edge_attrs[i], use_weighted=True)
        else:
            polymer_adj = create_adjacency(polymer_edge_indices[i], n_polymer)
        
        # Add adjacency matrix to batch
        batch_adj[i, :n_polymer, :n_polymer] = polymer_adj
    
    if has_qchem:
        return batch_features, batch_adj, conductivities, qchem_features
    else:
        return batch_features, batch_adj, conductivities

def prepare_data_loaders(polymer_smiles, conductivities, qchem_features=None,
                         batch_size=16, train_ratio=0.7, val_ratio=0.15, 
                         test_ratio=0.15, seed=42, use_fully_connected=False):
    """
    Prepare data loaders for training, validation, and testing
    
    Parameters:
    - polymer_smiles: List of polymer SMILES strings
    - conductivities: List of conductivity values
    - qchem_features: (optional) Numpy array of quantum chemistry features
    - batch_size: Batch size for data loaders
    - train_ratio: Ratio of training data
    - val_ratio: Ratio of validation data
    - test_ratio: Ratio of test data
    - seed: Random seed for reproducibility
    - use_fully_connected: Whether to use fully connected graphs
    
    Returns:
    - train_loader, val_loader, test_loader
    """
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"
    
    # Set seed for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create indices
    indices = np.arange(len(polymer_smiles))
    np.random.shuffle(indices)
    
    # Split indices
    train_size = int(len(indices) * train_ratio)
    val_size = int(len(indices) * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Extract quantum chemistry features for each dataset
    train_qchem = None
    val_qchem = None
    test_qchem = None
    
    if qchem_features is not None:
        train_qchem = qchem_features[train_indices]
        val_qchem = qchem_features[val_indices]
        test_qchem = qchem_features[test_indices]
    
    # Create datasets
    train_dataset = PolymerDataset(
        [polymer_smiles[i] for i in train_indices],
        [conductivities[i] for i in train_indices],
        qchem_features=train_qchem,
        use_fully_connected=use_fully_connected
    )
    
    val_dataset = PolymerDataset(
        [polymer_smiles[i] for i in val_indices],
        [conductivities[i] for i in val_indices],
        qchem_features=val_qchem,
        use_fully_connected=use_fully_connected
    )
    
    test_dataset = PolymerDataset(
        [polymer_smiles[i] for i in test_indices],
        [conductivities[i] for i in test_indices],
        qchem_features=test_qchem,
        use_fully_connected=use_fully_connected
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        collate_fn=collate_fn, drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        collate_fn=collate_fn, drop_last=False
    )
    
    return train_loader, val_loader, test_loader
