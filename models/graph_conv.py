import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConvLayer(nn.Module):
    """
    Graph Convolution Layer implementation
    """
    def __init__(self, in_features, out_features):
        super(GraphConvLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Learnable weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        # Bias term
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        
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
        batch_size, num_nodes, _ = x.size()
        
        # Normalize adjacency matrix for each batch
        adj_norm = self.normalize_adjacency(adj)
        
        # Reshape for batch matrix multiplication
        x_reshaped = x.view(batch_size * num_nodes, self.in_features)
        
        # Linear transformation
        support = torch.matmul(x_reshaped, self.weight)
        support = support.view(batch_size, num_nodes, self.out_features)
        
        # Neighborhood aggregation with batch processing
        output = torch.bmm(adj_norm, support)
        
        # Add bias
        output = output + self.bias
        
        return output
    

    def normalize_adjacency(self, adj):
        """
        Normalize adjacency matrix with batch support
        
        Parameters:
        - adj: Adjacency matrix [batch_size, num_nodes, num_nodes]
        
        Returns:
        - Normalized adjacency matrix
        """
        batch_size, num_nodes, _ = adj.size()
        
        # Add self-loops to adjacency matrix
        identity = torch.eye(num_nodes, device=adj.device).unsqueeze(0).expand(batch_size, -1, -1)
        adj_with_self = adj + identity
        
        # Compute D^(-1/2) for each graph in the batch
        rowsum = adj_with_self.sum(dim=2)  # [batch_size, num_nodes]
        d_inv_sqrt = torch.pow(rowsum, -0.5)  # [batch_size, num_nodes]
        d_inv_sqrt = torch.where(torch.isinf(d_inv_sqrt), torch.zeros_like(d_inv_sqrt), d_inv_sqrt)
        
        # Simpler approach to create batch diagonal matrices
        normalized_adj = torch.zeros_like(adj_with_self)
        
        for b in range(batch_size):
            # Create diagonal matrix for this batch
            diag_matrix = torch.diag(d_inv_sqrt[b])
            
            # Normalized adjacency: D^(-1/2) A D^(-1/2) for this batch
            normalized_adj[b] = diag_matrix @ adj_with_self[b] @ diag_matrix
        
        return normalized_adj
        