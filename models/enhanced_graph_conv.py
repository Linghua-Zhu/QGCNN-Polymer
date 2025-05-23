import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedGraphConvLayer(nn.Module):
    """
    Enhanced Graph Convolution Layer with residual connections and layer normalization
    """
    def __init__(self, in_features, out_features, use_residual=True, use_layer_norm=True):
        super(EnhancedGraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        
        # Bias term
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
        # Layer normalization
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(out_features)
        
        # Projection for residual connection if input and output dimensions differ
        if use_residual and in_features != out_features:
            self.residual_proj = nn.Linear(in_features, out_features, bias=False)
        
        # Parameter initialization
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x, adj):
        """
        Forward pass for graph convolution
        
        Parameters:
        - x: Node features [batch_size, num_nodes, in_features]
        - adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
        - Updated node features [batch_size, num_nodes, out_features]
        """
        batch_size, num_nodes, _ = x.shape
        
        # Store original input for residual connection
        identity = x
        
        # Reshape for matrix multiplication
        x_reshaped = x.reshape(batch_size * num_nodes, self.in_features)
        
        # Linear transformation
        support = torch.matmul(x_reshaped, self.weight)
        support = support.view(batch_size, num_nodes, self.out_features)
        
        # Neighborhood aggregation
        output = torch.bmm(adj, support)
        
        # Add bias
        output = output + self.bias
        
        # Add residual connection if requested
        if self.use_residual:
            if self.in_features != self.out_features:
                identity = self.residual_proj(identity)
            output = output + identity
        
        # Apply layer normalization if requested
        if self.use_layer_norm:
            output = self.layer_norm(output)
        
        return output
