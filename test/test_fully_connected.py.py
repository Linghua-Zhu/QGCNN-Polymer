import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from utils.molecular_graph import smiles_to_mol, mol_to_graph, mol_to_fully_connected_graph, smiles_to_pyg_data
from utils.data_loader import load_polymer_data
from utils.data_utils import create_adjacency

def visualize_graphs(smiles):
    """
    Visualize and compare regular graph vs fully connected graph
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        print(f"Failed to create molecule from SMILES: {smiles}")
        return
    
    # Create regular graph
    regular_graph = mol_to_graph(mol)
    
    # Create fully connected graph
    fully_connected_graph = mol_to_fully_connected_graph(mol)
    
    # Visualize both graphs
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Get positions for visualization (use 2D coordinates from RDKit)
    pos = {}
    conformer = mol.GetConformer()
    for i in range(mol.GetNumAtoms()):
        pos[i] = (conformer.GetAtomPosition(i).x, conformer.GetAtomPosition(i).y)
    
    # Plot regular graph
    ax1.set_title("Regular Molecular Graph")
    nx.draw(regular_graph, pos=pos, with_labels=True, node_color='skyblue', 
            node_size=500, edge_color='black', width=2, ax=ax1)
    
    # Plot fully connected graph
    ax2.set_title("Fully Connected Graph")
    
    # Color edges based on bond type vs non-bond
    edge_colors = []
    edge_widths = []
    
    for u, v, data in fully_connected_graph.edges(data=True):
        if data.get('is_bond', 0) == 1:
            edge_colors.append('black')
            edge_widths.append(2)
        else:
            # Lighter color for non-bond edges
            edge_colors.append('lightgray')
            edge_widths.append(0.5)
    
    nx.draw(fully_connected_graph, pos=pos, with_labels=True, node_color='lightgreen', 
            node_size=500, edge_color=edge_colors, width=edge_widths, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f"graph_comparison_{len(regular_graph.nodes())}_nodes.png")
    plt.close()
    
    # Visualize adjacency matrices
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    
    # Get regular adjacency matrix
    regular_adj = nx.to_numpy_array(regular_graph)
    
    # Get fully connected adjacency matrix
    # Convert to PyG data and back to get proper weighted adjacency
    pyg_data, _ = smiles_to_pyg_data(smiles, use_fully_connected=True)
    if pyg_data is not None and hasattr(pyg_data, 'edge_attr'):
        fully_connected_adj = create_adjacency(
            pyg_data.edge_index, 
            pyg_data.x.shape[0], 
            edge_attr=pyg_data.edge_attr, 
            use_weighted=True
        ).numpy()
    else:
        fully_connected_adj = nx.to_numpy_array(fully_connected_graph)
    
    ax1.set_title("Regular Graph Adjacency Matrix")
    im1 = ax1.imshow(regular_adj, cmap='Blues')
    plt.colorbar(im1, ax=ax1)
    
    ax2.set_title("Fully Connected Graph Adjacency Matrix (weighted)")
    im2 = ax2.imshow(fully_connected_adj, cmap='Blues')
    plt.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(f"adjacency_comparison_{len(regular_graph.nodes())}_nodes.png")
    plt.close()
    
    # Print statistics
    print(f"Regular graph: {regular_graph.number_of_nodes()} nodes, {regular_graph.number_of_edges()} edges")
    print(f"Fully connected graph: {fully_connected_graph.number_of_nodes()} nodes, {fully_connected_graph.number_of_edges()} edges")
    
    # Compute edge density
    n = regular_graph.number_of_nodes()
    max_edges = n * (n - 1) // 2
    regular_density = regular_graph.number_of_edges() / max_edges
    fully_connected_density = fully_connected_graph.number_of_edges() / max_edges
    
    print(f"Regular graph density: {regular_density:.4f}")
    print(f"Fully connected graph density: {fully_connected_density:.4f}")
    
    return regular_adj, fully_connected_adj

def test_pyg_conversion():
    """Test conversion to PyTorch Geometric data"""
    test_smiles = "C1=CC=CC=C1"  # Benzene
    
    # Test regular graph
    regular_data, regular_G = smiles_to_pyg_data(test_smiles, use_fully_connected=False)
    print("\nRegular Graph PyG Data:")
    print(f"Node features shape: {regular_data.x.shape}")
    print(f"Edge index shape: {regular_data.edge_index.shape}")
    print(f"Has edge attributes: {hasattr(regular_data, 'edge_attr')}")
    if hasattr(regular_data, 'edge_attr'):
        print(f"Edge attributes shape: {regular_data.edge_attr.shape}")
    
    # Test fully connected graph
    fc_data, fc_G = smiles_to_pyg_data(test_smiles, use_fully_connected=True)
    print("\nFully Connected Graph PyG Data:")
    print(f"Node features shape: {fc_data.x.shape}")
    print(f"Edge index shape: {fc_data.edge_index.shape}")
    print(f"Has edge attributes: {hasattr(fc_data, 'edge_attr')}")
    if hasattr(fc_data, 'edge_attr'):
        print(f"Edge attributes shape: {fc_data.edge_attr.shape}")
    
    # Verify numbers match NetworkX graphs
    print(f"\nRegular graph edges (NetworkX): {regular_G.number_of_edges()}")
    print(f"Regular graph edges (PyG): {regular_data.edge_index.shape[1] // 2}")  # Divide by 2 for undirected
    
    print(f"Fully connected graph edges (NetworkX): {fc_G.number_of_edges()}")
    print(f"Fully connected graph edges (PyG): {fc_data.edge_index.shape[1] // 2}")  # Divide by 2 for undirected

def main():
    # Test on a simple molecule first
    print("Testing on simple molecules:")
    simple_molecules = ["C1=CC=CC=C1", "CC(=O)O", "C1CCCCC1"]  # Benzene, Acetic acid, Cyclohexane
    
    for i, smiles in enumerate(simple_molecules):
        print(f"\nMolecule {i+1}: {smiles}")
        visualize_graphs(smiles)
    
    # Test PyG conversion
    test_pyg_conversion()
    
    # Load real examples from the dataset (if available)
    try:
        data_path = "data/Dataset_With_SMILES.csv"
        if os.path.exists(data_path):
            print("\nTesting on real dataset examples:")
            polymer_smiles, _ = load_polymer_data(data_path, log_transform=False, use_qchem_features=False)
            
            # Take first 3 examples for visualization
            for i, smiles in enumerate(polymer_smiles[:3]):
                print(f"\nPolymer {i+1}: {smiles}")
                visualize_graphs(smiles)
        else:
            print(f"\nDataset file not found: {data_path}")
    except Exception as e:
        print(f"Error loading dataset: {e}")

if __name__ == "__main__":
    main()