import torch
import numpy as np
import time
import os
from utils.data_loader import load_polymer_data
from utils.data_utils import prepare_data_loaders, PolymerDataset, collate_fn
from utils.molecular_graph import smiles_to_pyg_data
from models.graph_conv import GraphConvLayer
from models.quantum_encoding import QuantumFeatureReduction
from models.quantum_circuit import QuantumCircuit
from models.qgcnn import QGCNN

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory
os.makedirs("test_results", exist_ok=True)

# Test 1: SMILES processing
def test_smiles_processing():
    print("\n=== Test 1: SMILES Processing ===")
    smiles = "C=Cc1ccccc1"
    start_time = time.time()
    print(f"Processing SMILES: {smiles}")
    data, G = smiles_to_pyg_data(smiles)
    
    if data is not None:
        print(f"Success! Processed in {time.time() - start_time:.2f} seconds")
        print(f"Number of nodes: {data.x.shape[0]}")
        print(f"Node feature dimension: {data.x.shape[1]}")
        print(f"Number of edges: {data.edge_index.shape[1]}")
        return True
    else:
        print("Failed to process SMILES")
        return False

# Test 2: Data loading
def test_data_loading(limit=5):
    print("\n=== Test 2: Data Loading ===")
    csv_path = '/Users/zhulinghua/Dropbox/UW_research/QML_Polymer/polymer-qgcnn/data/Dataset_With_SMILES.csv'
    
    start_time = time.time()
    polymer_smiles, conductivities = load_polymer_data(csv_path)
    print(f"Data loaded in {time.time() - start_time:.2f} seconds")
    
    # Use only a subset for testing
    test_smiles = polymer_smiles[:limit]
    test_conductivities = conductivities[:limit]
    
    print(f"Testing with {len(test_smiles)} samples")
    for i, (smiles, cond) in enumerate(zip(test_smiles, test_conductivities)):
        print(f"Sample {i+1}: {smiles[:30]}... -> Conductivity: {cond}")
    
    return test_smiles, test_conductivities

# Test 3: Dataset creation
def test_dataset_creation(test_smiles, test_conductivities):
    print("\n=== Test 3: Dataset Creation ===")
    start_time = time.time()
    dataset = PolymerDataset(test_smiles, test_conductivities)
    print(f"Dataset created in {time.time() - start_time:.2f} seconds")
    print(f"Dataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample polymer nodes: {sample['polymer'].x.shape[0]}")
        print(f"Sample conductivity: {sample['conductivity']}")
        return dataset
    else:
        print("Failed to create dataset")
        return None

# Test 4: Data batching
def test_data_batching(dataset, batch_size=2):
    print("\n=== Test 4: Data Batching ===")
    if dataset is None:
        print("Dataset is None, cannot test batching")
        return None, None, None
    
    batch = [dataset[i] for i in range(min(batch_size, len(dataset)))]
    start_time = time.time()
    features, adj, targets = collate_fn(batch)
    print(f"Batch created in {time.time() - start_time:.2f} seconds")
    
    print(f"Batch features shape: {features.shape}")
    print(f"Batch adjacency matrix shape: {adj.shape}")
    print(f"Batch targets shape: {targets.shape}")
    
    return features, adj, targets

# Test 5: Graph convolution
def test_graph_convolution(features, adj):
    print("\n=== Test 5: Graph Convolution ===")
    if features is None or adj is None:
        print("Features or adjacency matrix is None, cannot test graph convolution")
        return None
    
    in_features = features.size(2)
    out_features = 16
    
    graph_conv = GraphConvLayer(in_features, out_features)
    print(f"Graph convolution layer created with in_features={in_features}, out_features={out_features}")
    
    start_time = time.time()
    output = graph_conv(features, adj)
    print(f"Graph convolution completed in {time.time() - start_time:.2f} seconds")
    print(f"Output shape: {output.shape}")
    
    return output

# Test 6: Quantum feature reduction
def test_quantum_feature_reduction(conv_output, n_qubits=8):
    print("\n=== Test 6: Quantum Feature Reduction ===")
    if conv_output is None:
        print("Convolution output is None, cannot test quantum feature reduction")
        return None
    
    input_dim = conv_output.size(2)
    quantum_reducer = QuantumFeatureReduction(input_dim, n_qubits)
    print(f"Quantum feature reducer created with input_dim={input_dim}, n_qubits={n_qubits}")
    
    start_time = time.time()
    reduced_features = quantum_reducer(conv_output)
    print(f"Feature reduction completed in {time.time() - start_time:.2f} seconds")
    print(f"Reduced features shape: {reduced_features.shape}")
    print(f"Expected dimension (2^n_qubits): {2**n_qubits}")
    
    # Check normalization
    norms = torch.norm(reduced_features, dim=1)
    print(f"Norms of reduced features: {norms}")
    
    return reduced_features

# Test 7: Quantum circuit
def test_quantum_circuit(reduced_features, n_qubits=8, qc_layers=3):
    print("\n=== Test 7: Quantum Circuit ===")
    if reduced_features is None:
        print("Reduced features is None, cannot test quantum circuit")
        return None
    
    # Set seed for reproducible circuit initialization
    torch.manual_seed(42)
    
    quantum_circuit = QuantumCircuit(n_qubits=n_qubits, n_layers=qc_layers)
    print(f"Quantum circuit created with n_qubits={n_qubits}, n_layers={qc_layers}")
    print(f"Number of quantum circuit parameters: {quantum_circuit.n_params}")
    
    start_time = time.time()
    print("Starting quantum circuit execution (this may take a while)...")
    circuit_output = quantum_circuit(reduced_features)
    elapsed_time = time.time() - start_time
    print(f"Quantum circuit execution completed in {elapsed_time:.2f} seconds")
    print(f"Circuit output shape: {circuit_output.shape}")
    print(f"Circuit output values: {circuit_output}")
    
    return circuit_output, elapsed_time

# Test 8: Complete QGCNN model
def test_complete_model(features, adj, targets, n_qubits=8, qc_layers=3):
    print("\n=== Test 8: Complete QGCNN Model ===")
    if features is None or adj is None:
        print("Features or adjacency matrix is None, cannot test complete model")
        return
    
    feature_dim = features.size(2)
    hidden_dim = 16
    
    # Set seed for reproducible model initialization
    torch.manual_seed(42)
    
    model = QGCNN(
        node_features=feature_dim,
        hidden_dim=hidden_dim,
        n_qubits=n_qubits,
        qc_layers=qc_layers
    )
    
    print(f"QGCNN model created with feature_dim={feature_dim}, hidden_dim={hidden_dim}")
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    
    start_time = time.time()
    print("Starting complete model forward pass (this may take a while)...")
    outputs = model(features, adj)
    elapsed_time = time.time() - start_time
    
    print(f"Model forward pass completed in {elapsed_time:.2f} seconds")
    print(f"Model output shape: {outputs.shape}")
    print(f"Model outputs: {outputs}")
    print(f"Target values: {targets}")
    
    # Calculate loss
    criterion = torch.nn.MSELoss()
    loss = criterion(outputs, targets)
    print(f"Initial loss: {loss.item()}")
    
    # Test single optimization step
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Re-evaluate after optimization step
    with torch.no_grad():
        new_outputs = model(features, adj)
        new_loss = criterion(new_outputs, targets)
    
    print(f"Loss after one optimization step: {new_loss.item()}")
    print(f"Loss change: {loss.item() - new_loss.item()}")

# Run all tests sequentially
if __name__ == "__main__":
    print("Starting QGCNN component tests...\n")
    
    # Test 1: SMILES processing
    if not test_smiles_processing():
        print("SMILES processing test failed, aborting further tests.")
        exit(1)
    
    # Test 2: Data loading
    test_smiles, test_conductivities = test_data_loading(limit=3)
    
    # Test 3: Dataset creation
    dataset = test_dataset_creation(test_smiles, test_conductivities)
    
    # Test 4: Data batching
    features, adj, targets = test_data_batching(dataset, batch_size=2)
    
    # Test 5: Graph convolution
    conv_output = test_graph_convolution(features, adj)
    
    # Test 6: Quantum feature reduction
    reduced_features = test_quantum_feature_reduction(conv_output, n_qubits=8)
    
    # Test 7: Quantum circuit (this may be the slow part)
    circuit_output, circuit_time = test_quantum_circuit(reduced_features, n_qubits=8, qc_layers=3)
    
    # If quantum circuit is very slow, warn the user
    if circuit_time > 60:
        print("\nWARNING: Quantum circuit execution is very slow.")
        print(f"It took {circuit_time:.2f} seconds for just {len(features)} samples with {features.size(1)} nodes.")
        print("Training the full model may take a very long time.")
        print("Consider reducing the number of qubits or circuit layers for faster execution.")
    
    # Test 8: Complete model
    test_complete_model(features, adj, targets, n_qubits=8, qc_layers=3)
    
    print("\nAll component tests completed!")