import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from models.qgcnn import QGCNN
from utils.data_utils import prepare_data_loaders

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                device, n_epochs=100, patience=10, model_save_path='results'):
    """
    Train the QGCNN model
    
    Parameters:
    - model: QGCNN model
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - criterion: Loss function
    - optimizer: Optimizer
    - device: Device for training (cpu or cuda)
    - n_epochs: Number of training epochs
    - patience: Early stopping patience
    - model_save_path: Path to save model checkpoints
    
    Returns:
    - trained model and training history
    """
    # Create directory for saving results
    os.makedirs(model_save_path, exist_ok=True)
    
    # Move model to device
    model = model.to(device)
    
    # Initialize tracking variables
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0
    
    # Training start time
    start_time = time.time()
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        epoch_loss = 0.0
        epoch_mae = 0.0
        n_batches = 0
        
        for batch_data in train_loader:
            # Check if batch includes quantum chemistry features
            if len(batch_data) == 4:
                x, adj, y, qchem = batch_data
                qchem = qchem.to(device)
            else:
                x, adj, y = batch_data
                qchem = None
            
            # Move data to device
            x = x.to(device)
            adj = adj.to(device)
            y = y.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(x, adj, qchem)
            loss = criterion(outputs, y)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            epoch_loss += loss.item()
            epoch_mae += mean_absolute_error(y.cpu().detach().numpy(), 
                                           outputs.cpu().detach().numpy())
            n_batches += 1
        
        # Calculate average metrics
        avg_train_loss = epoch_loss / n_batches
        avg_train_mae = epoch_mae / n_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        n_val_batches = 0
        
        with torch.no_grad():
            for batch_data in val_loader:
                # Check if batch includes quantum chemistry features
                if len(batch_data) == 4:
                    x, adj, y, qchem = batch_data
                    qchem = qchem.to(device)
                else:
                    x, adj, y = batch_data
                    qchem = None
                
                # Move data to device
                x = x.to(device)
                adj = adj.to(device)
                y = y.to(device)
                
                # Forward pass
                outputs = model(x, adj, qchem)
                loss = criterion(outputs, y)
                
                # Calculate metrics
                val_loss += loss.item()
                val_mae += mean_absolute_error(y.cpu().numpy(), outputs.cpu().numpy())
                n_val_batches += 1
        
        # Calculate average metrics
        avg_val_loss = val_loss / n_val_batches
        avg_val_mae = val_mae / n_val_batches
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)
        
        # Print progress
        print(f'Epoch {epoch+1}/{n_epochs} - '
              f'Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f}, '
              f'Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}')
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'mae': avg_val_mae
            }, os.path.join(model_save_path, 'best_model.pth'))
            
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f'Early stopping at epoch {epoch+1} as validation loss has not improved for {patience} epochs')
                break
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss
            }, os.path.join(model_save_path, f'model_epoch_{epoch+1}.pth'))
    
    # Training end time
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Best model was at epoch {best_epoch+1} with validation loss {best_val_loss:.6f}")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    # Loss curve
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # MAE curve
    plt.subplot(1, 2, 2)
    plt.plot(train_maes, label='Training MAE')
    plt.plot(val_maes, label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.title('Training and Validation MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_save_path, 'training_curves.png'))
    
    # Load best model
    checkpoint = torch.load(os.path.join(model_save_path, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_mae': train_maes,
        'val_mae': val_maes,
        'best_epoch': best_epoch,
        'training_time': training_time
    }
    
    return model, history

def evaluate_model(model, data_loader, device, log_transform=True):
    """
    Evaluate the model on a dataset
    
    Parameters:
    - model: Trained QGCNN model
    - data_loader: Data loader for evaluation
    - device: Device to use
    - log_transform: Whether predictions should be transformed back from log scale
    
    Returns:
    - Dictionary of evaluation metrics
    """
    model.eval()
    
    # Initialize lists to store predictions and targets
    predictions = []
    targets = []
    
    # Evaluation loop
    with torch.no_grad():
        for batch_data in data_loader:
            # Check if batch includes quantum chemistry features
            if len(batch_data) == 4:
                x, adj, y, qchem = batch_data
                qchem = qchem.to(device)
            else:
                x, adj, y = batch_data
                qchem = None
                
            # Move data to device
            x = x.to(device)
            adj = adj.to(device)
            
            # Forward pass
            outputs = model(x, adj, qchem)
            
            # Store predictions and targets
            predictions.extend(outputs.cpu().numpy().flatten())
            targets.extend(y.cpu().numpy().flatten())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Transform back to original scale if needed
    if log_transform:
        predictions_original = 10 ** predictions - 1e-6
        targets_original = 10 ** targets - 1e-6
        
        # Also calculate metrics in original scale
        mae_original = mean_absolute_error(targets_original, predictions_original)
        mse_original = mean_squared_error(targets_original, predictions_original)
        rmse_original = np.sqrt(mse_original)
        r2_original = r2_score(targets_original, predictions_original)
        
        print("\nMetrics in original scale:")
        print(f"MAE: {mae_original:.6f}")
        print(f"MSE: {mse_original:.6f}")
        print(f"RMSE: {rmse_original:.6f}")
        print(f"R2 Score: {r2_original:.6f}")
    
    # Calculate metrics in log scale
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(targets, predictions)
    
    # Print metrics
    print(f"Evaluation Metrics (log scale):")
    print(f"MAE: {mae:.6f}")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2 Score: {r2:.6f}")
    
    # Create scatter plot
    plt.figure(figsize=(8, 8))
    
    if log_transform:
        plt.scatter(targets_original, predictions_original, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(targets_original), np.min(predictions_original))
        max_val = max(np.max(targets_original), np.max(predictions_original))
        
        # Use log scale for better visualization
        plt.xscale('log')
        plt.yscale('log')
    else:
        plt.scatter(targets, predictions, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(np.min(targets), np.min(predictions))
        max_val = max(np.max(targets), np.max(predictions))
    
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Conductivity')
    plt.ylabel('Predicted Conductivity')
    plt.title('True vs Predicted Conductivity')
    plt.grid(True, alpha=0.3)
    
    # Add metrics as text
    if log_transform:
        plt.text(min_val * 10, max_val / 10,
                f'MAE: {mae_original:.4f}\nRMSE: {rmse_original:.4f}\nR²: {r2_original:.4f}',
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
        
        # Return metrics in both scales
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae_original': mae_original,
            'mse_original': mse_original,
            'rmse_original': rmse_original,
            'r2_original': r2_original,
            'predictions': predictions,
            'targets': targets,
            'predictions_original': predictions_original,
            'targets_original': targets_original
        }
    else:
        plt.text(min_val + 0.1 * (max_val - min_val), 
                max_val - 0.1 * (max_val - min_val),
                f'MAE: {mae:.4f}\nRMSE: {rmse:.4f}\nR²: {r2:.4f}',
                fontsize=12, bbox=dict(facecolor='white', alpha=0.7))
    
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'predictions': predictions,
            'targets': targets
        }
        