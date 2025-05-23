import pennylane as qml
import torch
import torch.nn as nn
import numpy as np
import random

class QuantumCircuit(nn.Module):
    """
    Variational Quantum Circuit module for QGCNN
    """
    def __init__(self, n_qubits=8, n_layers=3):
        super(QuantumCircuit, self).__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        
        # Calculate number of parameters
        # 3 rotation parameters per qubit (RX, RY, RZ) per layer
        self.n_params = n_qubits * 3 * n_layers
        
        # Set all random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Use fixed parameters instead of random initialization
        # Initialize all parameters to 0.1
        self.params = nn.Parameter(torch.ones(self.n_params) * 0.1)
        
        # Create quantum device and circuit
        self.dev = qml.device("default.qubit", wires=n_qubits, shots=None)
        self.qnode = qml.QNode(self.circuit, self.dev, interface="torch", diff_method="parameter-shift")
    
    def circuit(self, x, params):
        """
        Quantum circuit definition
        
        Parameters:
        - x: Input features (to be encoded)
        - params: Circuit parameters
        
        Returns:
        - Measurement expectation values
        """
        # Amplitude encoding of the input
        qml.AmplitudeEmbedding(x, wires=range(self.n_qubits), normalize=True)
        
        # Variational circuit layers
        param_idx = 0
        for layer in range(self.n_layers):
            # Single-qubit rotations
            for q in range(self.n_qubits):
                qml.RX(params[param_idx], wires=q)
                param_idx += 1
                qml.RY(params[param_idx], wires=q)
                param_idx += 1
                qml.RZ(params[param_idx], wires=q)
                param_idx += 1
            
            # Entangling layer - ring topology
            for q in range(self.n_qubits):
                qml.CNOT(wires=[q, (q + 1) % self.n_qubits])
        
        # Return expectation values for Pauli-Z on each qubit
        return [qml.expval(qml.PauliZ(q)) for q in range(self.n_qubits)]
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input features [batch_size, 2^n_qubits]
        
        Returns:
        - Quantum measurement results [batch_size, n_qubits]
        """
        batch_size = x.shape[0]
        results = torch.zeros(batch_size, self.n_qubits, device=x.device)
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Convert to numpy array for PennyLane
            x_i = x[i].detach().numpy()
            
            # Apply quantum circuit
            results[i] = torch.tensor(self.qnode(x_i, self.params))
        
        return results
        