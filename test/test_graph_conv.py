import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.molecular_graph import smiles_to_pyg_data
from models.graph_conv import GraphConvLayer

# Test SMILES
smiles = 'CN1C(=O)c2ccc3c4c(c(-c5ccc(-c6cccs6)s5)cc(c24)C1=O)C(=O)N(C)C3=O'

# Convert to PyTorch Geometric data
data, G = smiles_to_pyg_data(smiles)

# Extract node features and edge indices
x = data.x
edge_index = data.edge_index

# Create adjacency matrix
n = x.size(0)
adj = torch.zeros((n, n), dtype=torch.float)
for i in range(edge_index.size(1)):
    src, dst = edge_index[0, i], edge_index[1, i]
    adj[src, dst] = 1.0
    adj[dst, src] = 1.0  # Undirected graph

# Simulate batch size of 1
x = x.unsqueeze(0)  # [1, num_nodes, features]
adj = adj.unsqueeze(0)  # [1, num_nodes, num_nodes]

# Initialize graph convolution layer
in_features = x.size(2)
out_features = 16
graph_conv = GraphConvLayer(in_features, out_features)

# Apply graph convolution
output = graph_conv(x, adj)

# Print shape information
print(f"Input shape: {x.shape}")
print(f"Adjacency matrix shape: {adj.shape}")
print(f"Output shape after graph convolution: {output.shape}")

# Visualize feature changes - take the first batch
x_vis = x[0].detach().numpy()
output_vis = output[0].detach().numpy()

plt.figure(figsize=(12, 5))

# Original features visualization
plt.subplot(1, 2, 1)
plt.imshow(x_vis, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Original Node Features")
plt.xlabel("Feature Dimension")
plt.ylabel("Node ID")

# Convolved features visualization
plt.subplot(1, 2, 2)
plt.imshow(output_vis, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Node Features After Graph Convolution")
plt.xlabel("Feature Dimension")
plt.ylabel("Node ID")

plt.tight_layout()
plt.savefig("graph_conv_visualization.png")
plt.show()

print("Graph convolution test completed!")