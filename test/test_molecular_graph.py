import matplotlib.pyplot as plt
import networkx as nx
from utils.molecular_graph import smiles_to_pyg_data

# Test SMILES
smiles = 'CN1C(=O)c2ccc3c4c(c(-c5ccc(-c6cccs6)s5)cc(c24)C1=O)C(=O)N(C)C3=O'

# Convert to PyTorch Geometric data and NetworkX graph
data, G = smiles_to_pyg_data(smiles)

# Print PyTorch Geometric data information
print("PyTorch Geometric Data:")
print(f"Number of nodes: {data.x.shape[0]}")
print(f"Node feature dimension: {data.x.shape[1]}")
print(f"Number of edges: {data.edge_index.shape[1]}")

# Visualize molecular graph
plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)  # Set node positions
labels = nx.get_node_attributes(G, 'symbol')  # Get atom labels

nx.draw(G, pos, with_labels=True, labels=labels, node_color='skyblue', 
        node_size=700, edge_color='gray', linewidths=1, font_size=15)

plt.title(f"Molecular Graph: {smiles}")
plt.savefig("molecule_graph.png")
plt.show()

print("Done!")