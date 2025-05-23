import torch
import torch.nn as nn
import torch.nn.functional as F
from .multi_layer_gcn import MultiLayerGCN
from .graph_pooling import GraphPooling
from .quantum_encoding import QuantumFeatureReduction
from .quantum_circuit import QuantumCircuit

class EnhancedQGCNN(nn.Module):
    """
    Enhanced Quantum Graph Convolutional Neural Network with multi-layer GCN
    """
    def __init__(self, node_features, hidden_dims=[32, 64, 128], n_qubits=8, qc_layers=3, 
                 qchem_dim=0, dropout=0.2, pool_type='attention'):
        """
        Initialize the enhanced QGCNN model
        
        Parameters:
        - node_features: Number of input features per node
        - hidden_dims: List of hidden dimensions for GCN layers
        - n_qubits: Number of qubits for quantum circuit
        - qc_layers: Number of layers in quantum circuit
        - qchem_dim: Dimension of quantum chemistry features (0 if not used)
        - dropout: Dropout probability
        - pool_type: Graph pooling type ('mean', 'max', 'sum', or 'attention')
        """
        super(EnhancedQGCNN, self).__init__()
        
        # Multi-layer Graph Convolutional Network
        self.gcn = MultiLayerGCN(node_features, hidden_dims, dropout)
        
        # Graph pooling layer
        self.graph_pooling = GraphPooling(hidden_dims[-1], pool_type=pool_type)
        
        # Quantum Feature Reduction
        self.feature_reducer = QuantumFeatureReduction(hidden_dims[-1], n_qubits)
        
        # Quantum Circuit
        self.quantum_circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=qc_layers)
        
        # Quantum Chemistry Feature Processing
        self.use_qchem = qchem_dim > 0
        if self.use_qchem:
            self.homo_idx = None
            self.lumo_idx = None
            
            # Process quantum chemistry features
            self.qchem_processor = nn.Sequential(
                nn.Linear(qchem_dim, 32),
                nn.ReLU(),
                nn.BatchNorm1d(32),
                nn.Dropout(dropout),
                nn.Linear(32, 16)
            )
            
            # Output layer dimension depends on whether we use qchem features
            output_dim = n_qubits + 16
        else:
            output_dim = n_qubits
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def set_qchem_feature_names(self, feature_names):
        """
        Set quantum chemistry feature names for physical feature computation
        
        Parameters:
        - feature_names: List of feature names
        """
        if not self.use_qchem:
            return
            
        for i, name in enumerate(feature_names):
            if name.upper() in ['HOMO', 'HOCO']:
                self.homo_idx = i
                print(f"HOMO/HOCO index set to {i}")
            elif name.upper() in ['LUMO', 'LUCO']:
                self.lumo_idx = i
                print(f"LUMO/LUCO index set to {i}")
    
    def compute_physical_features(self, qchem):
        """
        Compute physical features from quantum chemistry features
        
        Parameters:
        - qchem: Quantum chemistry features [batch_size, qchem_dim]
        
        Returns:
        - Enhanced features [batch_size, qchem_dim + n_physical_features]
        """
        batch_size = qchem.shape[0]
        features = [qchem]  # Start with original features
        
        if self.homo_idx is not None and self.lumo_idx is not None:
            homo = qchem[:, self.homo_idx]
            lumo = qchem[:, self.lumo_idx]
            
            # HOMO-LUMO gap (measure of chemical reactivity)
            gap = lumo - homo
            features.append(gap.view(batch_size, 1))
            
            # Chemical hardness (resistance to charge transfer)
            hardness = 0.5 * gap
            features.append(hardness.view(batch_size, 1))
            
            # Electronegativity (ability to attract electrons)
            electronegativity = -0.5 * (homo + lumo)
            features.append(electronegativity.view(batch_size, 1))
            
            # Electrophilicity index
            gap_safe = torch.clamp(gap, min=1e-6)
            electrophilicity = torch.pow(homo + lumo, 2) / (8 * gap_safe)
            features.append(electrophilicity.view(batch_size, 1))
        
        return torch.cat(features, dim=1)
    
    def forward(self, x, adj, qchem=None):
        """
        Forward pass of EnhancedQGCNN
        
        Parameters:
        - x: Node features [batch_size, num_nodes, node_features]
        - adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        - qchem: (optional) Quantum chemistry features [batch_size, qchem_dim]
        
        Returns:
        - prediction: Final prediction [batch_size, 1]
        """
        # Multi-layer Graph Convolution
        x = self.gcn(x, adj)
        
        # Create node mask (identify padding)
        node_mask = (torch.sum(x, dim=-1) != 0).float()
        
        # Graph pooling
        x_pooled = self.graph_pooling(x, node_mask)
        
        # Quantum Feature Reduction
        quantum_features = self.feature_reducer(x_pooled)
        
        # Quantum Circuit Processing
        quantum_output = self.quantum_circuit(quantum_features)
        
        # Process quantum chemistry features if available
        if self.use_qchem and qchem is not None:
            # Compute physical features
            qchem_ext = self.compute_physical_features(qchem)
            
            # Process through neural network
            qchem_output = self.qchem_processor(qchem_ext)
            
            # Combine quantum output with quantum chemistry features
            combined = torch.cat([quantum_output, qchem_output], dim=1)
            prediction = self.output_layer(combined)
        else:
            # Use only quantum output
            prediction = self.output_layer(quantum_output)
        
        return prediction
