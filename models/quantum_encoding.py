import torch
import torch.nn as nn
import numpy as np
import pennylane as qml

class QuantumFeatureReduction(nn.Module):
    """
    Quantum Feature Reduction Module - Reduces high-dimensional features from graph convolution
    to dimensions suitable for quantum circuits
    """
    def __init__(self, input_dim, n_qubits):
        super(QuantumFeatureReduction, self).__init__()
        self.input_dim = input_dim
        self.n_qubits = n_qubits
        self.output_dim = 2**n_qubits  # Dimension for quantum amplitude encoding
        
        # Multi-layer network for dimensionality reduction
        # More layers for higher dimensional output
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.LayerNorm(input_dim * 2),
            nn.Linear(input_dim * 2, input_dim * 4),
            nn.ReLU(),
            nn.LayerNorm(input_dim * 4),
            nn.Linear(input_dim * 4, self.output_dim),
            nn.Tanh()  # Using Tanh to ensure outputs between -1 and 1
        )
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input features [batch_size, num_nodes, input_dim]
        
        Returns:
        - Reduced features [batch_size, output_dim]
        """
        batch_size = x.shape[0]
        num_nodes = x.shape[1]
        
        # Graph pooling - average all node features
        x_pool = torch.mean(x, dim=1)  # [batch_size, input_dim]
        
        # Feature reduction
        x_reduced = self.layers(x_pool)  # [batch_size, output_dim]
        
        # Normalize for quantum amplitude encoding
        norm = torch.norm(x_reduced, dim=1, keepdim=True)
        x_normalized = x_reduced / (norm + 1e-8)
        
        return x_normalized

class AmplitudeEncoding:
    """
    Helper class for Quantum Amplitude Encoding
    """
    @staticmethod
    def normalize(features):
        """
        Normalize feature vector for amplitude encoding
        
        Parameters:
        - features: Input feature vector
        
        Returns:
        - Normalized feature vector
        """
        norm = np.linalg.norm(features)
        if norm < 1e-8:
            return features
        return features / norm
    
    @staticmethod
    def pad_features(features, target_dim):
        """
        Pad or truncate feature vector to 2^n dimensions
        
        Parameters:
        - features: Input feature vector
        - target_dim: Target dimension (2^n)
        
        Returns:
        - Adjusted feature vector
        """
        if len(features) < target_dim:
            # Pad with zeros if features are shorter than target
            padded = np.zeros(target_dim)
            padded[:len(features)] = features
            return padded
        elif len(features) > target_dim:
            # Truncate if features are longer than target
            return features[:target_dim]
        else:
            # Dimension is already correct
            return features
            