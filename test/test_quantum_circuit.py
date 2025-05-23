import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.molecular_graph import smiles_to_pyg_data
from models.graph_conv import GraphConvLayer
from models.quantum_encoding import QuantumFeatureReduction
from models.quantum_circuit import QuantumCircuit

import random


random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

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

# Graph convolution layer
in_features = x.size(2)
hidden_dim = 16
graph_conv = GraphConvLayer(in_features, hidden_dim)

# Apply graph convolution
conv_output = graph_conv(x, adj)  # [1, num_nodes, hidden_dim]

# Quantum feature reduction
n_qubits = 8  # 8 qubits -> 2^8 = 256 dimensions
quantum_reducer = QuantumFeatureReduction(hidden_dim, n_qubits)

# Apply reduction
reduced_features = quantum_reducer(conv_output)  # [1, 2^n_qubits]

# Quantum circuit
quantum_circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=3)

# Apply quantum circuit
circuit_output = quantum_circuit(reduced_features)

# Print information
print(f"Original feature dimensions: {x.shape}")
print(f"Feature dimensions after graph convolution: {conv_output.shape}")
print(f"Feature dimensions after quantum reduction: {reduced_features.shape}")
print(f"Number of qubits used: {n_qubits}")
print(f"Quantum state amplitude dimensions: {2**n_qubits}")
print(f"Quantum circuit output dimensions: {circuit_output.shape}")
print(f"Number of circuit parameters: {quantum_circuit.n_params}")

# Visualize quantum circuit output
plt.figure(figsize=(10, 4))
plt.bar(range(circuit_output.shape[1]), circuit_output[0].detach().numpy())
plt.title("Quantum Circuit Measurement Results")
plt.xlabel("Qubit Index")
plt.ylabel("Expectation Value <Z>")
plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
plt.grid(axis='y', alpha=0.3)
plt.savefig("quantum_circuit_output.png")
plt.show()

print("Quantum circuit test completed!")
