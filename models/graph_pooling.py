import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphPooling(nn.Module):
    """
    Advanced graph pooling methods to aggregate node features
    """
    def __init__(self, input_dim, pool_type='mean'):
        """
        Initialize the graph pooling layer
        
        Parameters:
        - input_dim: Input feature dimension
        - pool_type: Pooling type ('mean', 'max', 'sum', or 'attention')
        """
        super(GraphPooling, self).__init__()
        self.input_dim = input_dim
        self.pool_type = pool_type
        
        if pool_type == 'attention':
            # Attention-based pooling
            self.attention = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )
    
    def forward(self, x, mask=None):
        """
        Forward pass for graph pooling
        
        Parameters:
        - x: Node features [batch_size, num_nodes, input_dim]
        - mask: Optional mask for padding [batch_size, num_nodes]
        
        Returns:
        - Pooled graph features [batch_size, input_dim]
        """
        if mask is None:
            # Create a default mask (all nodes are valid)
            mask = torch.ones(x.size(0), x.size(1), device=x.device)
        
        # Expand mask for broadcasting
        expanded_mask = mask.unsqueeze(-1).expand_as(x)
        
        if self.pool_type == 'mean':
            # Mean pooling
            sum_x = torch.sum(x * expanded_mask, dim=1)
            # Count valid nodes for proper averaging
            node_count = torch.sum(mask, dim=1, keepdim=True).clamp(min=1)
            return sum_x / node_count
            
        elif self.pool_type == 'sum':
            # Sum pooling
            return torch.sum(x * expanded_mask, dim=1)
            
        elif self.pool_type == 'max':
            # Max pooling (with masking)
            x_masked = x * expanded_mask - 1e9 * (1 - expanded_mask)
            return torch.max(x_masked, dim=1)[0]
            
        elif self.pool_type == 'attention':
            # Attention pooling
            scores = self.attention(x)  # [batch_size, num_nodes, 1]
            
            # Apply mask to attention scores
            scores = scores.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            
            # Attention weights
            attn_weights = F.softmax(scores, dim=1)  # [batch_size, num_nodes, 1]
            
            # Apply attention weights
            weighted_x = x * attn_weights
            
            # Sum the weighted features
            return torch.sum(weighted_x, dim=1)
        
        else:
            raise ValueError(f"Unknown pooling type: {self.pool_type}")
