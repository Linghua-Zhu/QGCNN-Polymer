import torch
import numpy as np
import os
import argparse
import random
import pandas as pd
from models.qgcnn import QGCNN
from utils.data_utils import prepare_data_loaders
from utils.data_loader import load_polymer_data
from train import train_model, evaluate_model

def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def main(args):
    # Set random seed
    set_seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load polymer data
    data_result = load_polymer_data(
        args.data_path, 
        log_transform=True,
        use_qchem_features=args.use_qchem_features
    )
    
    # Check if quantum chemistry features were loaded
    if len(data_result) == 3:
        polymer_smiles, conductivities, qchem_features = data_result
        print(f"Using quantum chemistry features with shape: {qchem_features.shape}")
        qchem_dim = qchem_features.shape[1]
        
        # Get feature names (if available)
        try:
            data = pd.read_csv(args.data_path)
            qchem_cols = []
            potential_cols = ['HOMO', 'LUMO', 'HOCO', 'LUCO', 'Indirect Gap', 'EA/eV', 'HOMO-LUMO Gap']
            for col in potential_cols:
                if col in data.columns:
                    qchem_cols.append(col)
            print(f"Quantum chemistry features: {qchem_cols}")
        except:
            qchem_cols = None
            print("Could not extract quantum chemistry feature names")
    else:
        polymer_smiles, conductivities = data_result
        qchem_features = None
        qchem_dim = 0
        qchem_cols = None
        print("No quantum chemistry features used")
    
    # Prepare data loaders
    train_loader, val_loader, test_loader = prepare_data_loaders(
        polymer_smiles, conductivities,
        qchem_features=qchem_features,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Get feature dimension from first batch
    for batch_data in train_loader:
        if len(batch_data) == 4:
            x, adj, _, _ = batch_data
        else:
            x, adj, _ = batch_data
        feature_dim = x.size(2)
        break
    
    # Create model
    model = QGCNN(
        node_features=feature_dim,
        hidden_dim=args.hidden_dim,
        n_qubits=args.n_qubits,
        qc_layers=args.qc_layers,
        qchem_dim=qchem_dim
    )
    
    # Set feature names in model if available
    if qchem_cols and hasattr(model, 'set_qchem_feature_names'):
        model.set_qchem_feature_names(qchem_cols)
    
    # Print model info
    print(f"Total model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Using {args.n_qubits} qubits")
    print(f"Using {args.qc_layers} quantum circuit layers")
    if qchem_dim > 0:
        print(f"Integrated {qchem_dim} quantum chemistry features")
    
    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    if not args.eval_only:
        model, history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=args.epochs,
            patience=args.patience,
            model_save_path=args.output_dir
        )
        
        # Save training history
        torch.save(history, os.path.join(args.output_dir, 'training_history.pth'))
    else:
        # Load pre-trained model
        checkpoint = torch.load(os.path.join(args.output_dir, 'best_model.pth'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pre-trained model from epoch {checkpoint['epoch']+1}")
    
    # Evaluate model on test set
    model = model.to(device)
    test_metrics = evaluate_model(model, test_loader, device)
    
    # Save evaluation results
    torch.save(test_metrics, os.path.join(args.output_dir, 'test_metrics.pth'))
    
    print("Training and evaluation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QGCNN for Polymer Conductivity Prediction")
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True, help='Path to the data CSV file')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training')
    parser.add_argument('--train_ratio', type=float, default=0.7, help='Ratio of training data')
    parser.add_argument('--val_ratio', type=float, default=0.15, help='Ratio of validation data')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test data')
    parser.add_argument('--use_qchem_features', action='store_true', help='Use quantum chemistry features (HOMO, LUMO, etc.)')
    
    # Model parameters
    parser.add_argument('--hidden_dim', type=int, default=16, help='Hidden dimension size')
    parser.add_argument('--n_qubits', type=int, default=8, help='Number of qubits for quantum circuit')
    parser.add_argument('--qc_layers', type=int, default=3, help='Number of layers in quantum circuit')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--eval_only', action='store_true', help='Only run evaluation (no training)')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    main(args)
    