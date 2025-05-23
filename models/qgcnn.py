import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph_conv import GraphConvLayer
from .quantum_encoding import QuantumFeatureReduction
from .quantum_circuit import QuantumCircuit
from .qchem_processor import QChemProcessor

class QGCNN(nn.Module):
    """
    Quantum Graph Convolutional Neural Network with quantum chemistry feature integration
    """
    def __init__(self, node_features, hidden_dim=16, n_qubits=8, qc_layers=3, qchem_dim=0):
        super(QGCNN, self).__init__()
        
        # Graph Convolution Layer
        self.graph_conv = GraphConvLayer(node_features, hidden_dim)
        
        # Quantum Feature Reduction
        self.feature_reducer = QuantumFeatureReduction(hidden_dim, n_qubits)
        
        # Quantum Circuit
        self.quantum_circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=qc_layers)
        
        # Quantum Chemistry Feature Processing
        self.use_qchem = qchem_dim > 0
        if self.use_qchem:
            self.qchem_processor = QChemProcessor(qchem_dim, output_dim=8)
            # Adjust output layer to include qchem features
            final_dim = n_qubits + 8
        else:
            final_dim = n_qubits
        
        # Output Layer
        self.output_layer = nn.Sequential(
            nn.Linear(final_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add some regularization
            nn.Linear(16, 1)
        )
    
    def set_qchem_feature_names(self, feature_names):
        """
        Set quantum chemistry feature names for the processor
        """
        if self.use_qchem:
            self.qchem_processor.set_feature_indices(feature_names)
    
    def forward(self, x, adj, qchem=None):
        """
        Forward pass of QGCNN
        
        Parameters:
        - x: Node features [batch_size, num_nodes, node_features]
        - adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        - qchem: (optional) Quantum chemistry features [batch_size, qchem_dim]
        
        Returns:
        - prediction: Final prediction [batch_size, 1]
        """
        # Graph Convolution
        conv_features = self.graph_conv(x, adj)
        
        # Quantum Feature Reduction
        quantum_features = self.feature_reducer(conv_features)
        
        # Quantum Circuit Processing
        quantum_output = self.quantum_circuit(quantum_features)
        
        # Process and incorporate quantum chemistry features if available
        if self.use_qchem and qchem is not None:
            qchem_output = self.qchem_processor(qchem)
            # Combine quantum output with quantum chemistry features
            combined = torch.cat([quantum_output, qchem_output], dim=1)
            prediction = self.output_layer(combined)
        else:
            # Use only quantum output
            prediction = self.output_layer(quantum_output)
        
        return prediction
        