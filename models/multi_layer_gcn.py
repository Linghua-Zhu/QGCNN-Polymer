import torch
import torch.nn as nn
import torch.nn.functional as F
from .enhanced_graph_conv import EnhancedGraphConvLayer

class MultiLayerGCN(nn.Module):
    """
    Multi-layer Graph Convolutional Network with residual connections and normalization
    """
    def __init__(self, in_features, hidden_dims, dropout=0.2):
        """
        Initialize the multi-layer GCN
        
        Parameters:
        - in_features: Number of input features per node
        - hidden_dims: List of hidden dimensions for each layer
        - dropout: Dropout probability
        """
        super(MultiLayerGCN, self).__init__()
        
        # Create list of graph convolution layers
        self.gc_layers = nn.ModuleList()
        
        # Input layer
        self.gc_layers.append(
            EnhancedGraphConvLayer(in_features, hidden_dims[0], 
                                   use_residual=False, use_layer_norm=True)
        )
        
        # Hidden layers with residual connections
        for i in range(1, len(hidden_dims)):
            self.gc_layers.append(
                EnhancedGraphConvLayer(hidden_dims[i-1], hidden_dims[i], 
                                       use_residual=True, use_layer_norm=True)
            )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Final dimension
        self.final_dim = hidden_dims[-1]
    
    def forward(self, x, adj):
        """
        Forward pass through multiple GCN layers
        
        Parameters:
        - x: Node features [batch_size, num_nodes, in_features]
        - adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
        - Node embeddings after multiple GCN layers [batch_size, num_nodes, final_dim]
        """
        # First layer
        x = self.gc_layers[0](x, adj)
        x = F.relu(x)
        x = self.dropout(x)
        
        # Remaining layers with residual connections
        for i in range(1, len(self.gc_layers)):
            x = self.gc_layers[i](x, adj)
            x = F.relu(x)
            x = self.dropout(x)
        
        return x
