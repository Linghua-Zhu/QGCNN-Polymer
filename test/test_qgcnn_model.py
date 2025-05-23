import torch
import torch.nn as nn  
import torch.optim as optim 
import numpy as np
import matplotlib.pyplot as plt
from utils.molecular_graph import smiles_to_pyg_data
from models.qgcnn import QGCNN

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Test SMILES
smiles = 'CN1C(=O)c2ccc3c4c(c(-c5ccc(-c6cccs6)s5)cc(c24)C1=O)C(=O)N(C)C3=O'

# Create a "synthetic" conductivity value for testing
synthetic_conductivity = 0.75  # Just for testing

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
target = torch.tensor([[synthetic_conductivity]], dtype=torch.float)

# Create QGCNN model
model = QGCNN(
    node_features=x.size(2),
    hidden_dim=16,
    n_qubits=8,
    qc_layers=3
)

# Forward pass
prediction = model(x, adj)

# Print information
print(f"Input features shape: {x.shape}")
print(f"Target conductivity: {target.item()}")
print(f"Predicted conductivity: {prediction.item()}")
print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")

# Test a simple optimization step
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Simple training loop for testing
for epoch in range(5):
    # Forward pass
    prediction = model(x, adj)
    
    # Compute loss
    loss = criterion(prediction, target)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"Epoch {epoch+1}/5, Loss: {loss.item():.6f}, Prediction: {prediction.item():.6f}")

print("QGCNN model test completed!")