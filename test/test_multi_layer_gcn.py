import torch
import numpy as np
import matplotlib.pyplot as plt
from utils.molecular_graph import smiles_to_pyg_data
from models.multi_layer_gcn import MultiLayerGCN
from models.enhanced_graph_conv import EnhancedGraphConvLayer
from models.graph_pooling import GraphPooling

def test_gcn_layer():
    """Test enhanced graph convolution layer"""
    # Create random input
    batch_size = 2
    num_nodes = 5
    in_features = 3
    out_features = 6
    
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.zeros(batch_size, num_nodes, num_nodes)
    
    # Create simple adjacency matrix
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j or abs(i-j) == 1:
                    adj[b, i, j] = 1.0
    
    # Test without residual and normalization
    layer1 = EnhancedGraphConvLayer(in_features, out_features, 
                                   use_residual=False, use_layer_norm=False)
    out1 = layer1(x, adj)
    
    # Test with residual but no normalization
    layer2 = EnhancedGraphConvLayer(in_features, out_features, 
                                   use_residual=True, use_layer_norm=False)
    out2 = layer2(x, adj)
    
    # Test with both residual and normalization
    layer3 = EnhancedGraphConvLayer(in_features, out_features, 
                                   use_residual=True, use_layer_norm=True)
    out3 = layer3(x, adj)
    
    print("\nEnhanced Graph Convolution Layer:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape (no residual, no norm): {out1.shape}")
    print(f"Output shape (with residual, no norm): {out2.shape}")
    print(f"Output shape (with residual and norm): {out3.shape}")
    
    # Check residual effect
    diff = torch.norm(out2 - out1)
    print(f"Difference with vs without residual: {diff.item():.4f}")
    
    return True

def test_multi_layer_gcn():
    """Test multi-layer GCN module"""
    # Create random input
    batch_size = 2
    num_nodes = 5
    in_features = 3
    hidden_dims = [8, 16, 32]
    
    x = torch.randn(batch_size, num_nodes, in_features)
    adj = torch.zeros(batch_size, num_nodes, num_nodes)
    
    # Create simple adjacency matrix
    for b in range(batch_size):
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j or abs(i-j) == 1:
                    adj[b, i, j] = 1.0
    
    # Create multi-layer GCN
    multi_gcn = MultiLayerGCN(in_features, hidden_dims, dropout=0.2)
    
    # Forward pass
    out = multi_gcn(x, adj)
    
    print("\nMulti-Layer GCN:")
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Expected output dimension: {hidden_dims[-1]}")
    
    return True

def test_graph_pooling():
    """Test graph pooling methods"""
    # Create random input
    batch_size = 2
    num_nodes = 5
    features = 8
    
    x = torch.randn(batch_size, num_nodes, features)
    
    # Create mask (some nodes are padding)
    mask = torch.ones(batch_size, num_nodes)
    mask[0, 3:] = 0  # Last two nodes in first graph are padding
    
    # Test different pooling methods
    mean_pool = GraphPooling(features, pool_type='mean')
    max_pool = GraphPooling(features, pool_type='max')
    sum_pool = GraphPooling(features, pool_type='sum')
    attn_pool = GraphPooling(features, pool_type='attention')
    
    mean_out = mean_pool(x, mask)
    max_out = max_pool(x, mask)
    sum_out = sum_pool(x, mask)
    attn_out = attn_pool(x, mask)
    
    print("\nGraph Pooling:")
    print(f"Input shape: {x.shape}")
    print(f"Mean pooling output shape: {mean_out.shape}")
    print(f"Max pooling output shape: {max_out.shape}")
    print(f"Sum pooling output shape: {sum_out.shape}")
    print(f"Attention pooling output shape: {attn_out.shape}")
    
    return True

def test_on_real_molecule():
    """Test the multi-layer GCN on a real molecule"""
    # Define a simple molecule
    smiles = "CCO"  # Ethanol
    
    # Convert to PyG data
    data, _ = smiles_to_pyg_data(smiles)
    
    if data is None:
        print("Failed to process molecule")
        return False
    
    # Extract features and create batch
    x = data.x.unsqueeze(0)  # Add batch dimension
    edge_index = data.edge_index
    
    # Create adjacency matrix
    num_nodes = x.size(1)
    adj = torch.zeros(1, num_nodes, num_nodes)
    
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i], edge_index[1, i]
        adj[0, src, dst] = 1.0
    
    # Create multi-layer GCN
    in_features = x.size(2)
    hidden_dims = [16, 32, 64]
    multi_gcn = MultiLayerGCN(in_features, hidden_dims)
    
    # Forward pass
    out = multi_gcn(x, adj)
    
    print("\nTesting on real molecule (Ethanol):")
    print(f"Number of atoms: {num_nodes}")
    print(f"Input feature dimension: {in_features}")
    print(f"Output shape: {out.shape}")
    
    # Create a visualization of the node embeddings
    embeddings = out[0].detach().numpy()  # Remove batch dimension
    
    # Use PCA to reduce to 2D for visualization
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=100)
    
    # Add atom indices
    for i in range(num_nodes):
        plt.annotate(str(i), (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    fontsize=12)
    
    plt.title("Multi-Layer GCN Node Embeddings for Ethanol")
    plt.savefig("ethanol_embeddings.png")
    plt.close()
    
    return True

def main():
    print("Testing Enhanced Graph Convolution components...")
    
    # Test enhanced graph convolution layer
    test_gcn_layer()
    
    # Test multi-layer GCN
    test_multi_layer_gcn()
    
    # Test graph pooling
    test_graph_pooling()
    
    # Test on real molecule
    test_on_real_molecule()
    
    print("\nAll tests completed successfully!")

if __name__ == "__main__":
    main()