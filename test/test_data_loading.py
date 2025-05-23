from utils.data_loader import load_polymer_data
from utils.data_utils import prepare_data_loaders

# Load data
csv_path = 'data/Dataset_With_SMILES.csv'
polymer_smiles, conductivities = load_polymer_data(csv_path)

print("Data loaded successfully!")
print(f"Total of {len(polymer_smiles)} polymer samples")

# Prepare data loaders
train_loader, val_loader, test_loader = prepare_data_loaders(
    polymer_smiles, conductivities,
    batch_size=4,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
)

print("\nData loaders created successfully!")
print(f"Training set size: {len(train_loader.dataset)} samples")
print(f"Validation set size: {len(val_loader.dataset)} samples")
print(f"Test set size: {len(test_loader.dataset)} samples")

# Check first batch
for x, adj, y in train_loader:
    print("\nFirst batch:")
    print(f"Feature tensor shape: {x.shape}")
    print(f"Adjacency matrix shape: {adj.shape}")
    print(f"Target values shape: {y.shape}")
    break

print("\nTest completed!")