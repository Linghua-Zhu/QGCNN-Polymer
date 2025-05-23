import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class QChemProcessor(nn.Module):
    """
    Process quantum chemistry features and create physically meaningful composite features
    """
    def __init__(self, input_dim, output_dim=8):
        super(QChemProcessor, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Identify feature names (if possible)
        self.homo_idx = None
        self.lumo_idx = None
        
        # Feature transformation network - will be defined in forward
        # since input dimension depends on physical features
        self.feature_net = None
        self.initialized = False
    
    def set_feature_indices(self, feature_names):
        """
        Set feature indices for computing physical features
        
        Parameters:
        - feature_names: List of quantum chemistry feature names
        """
        for i, name in enumerate(feature_names):
            if name.upper() in ['HOMO', 'HOCO']:
                self.homo_idx = i
                print(f"HOMO/HOCO index set to {i}")
            elif name.upper() in ['LUMO', 'LUCO']:
                self.lumo_idx = i
                print(f"LUMO/LUCO index set to {i}")
        
        if self.homo_idx is not None and self.lumo_idx is not None:
            print("Physical composite features can be calculated (e.g., HOMO-LUMO gap)")
    
    def compute_physical_features(self, x):
        """
        Compute physically meaningful composite features
        
        Parameters:
        - x: Input features [batch_size, input_dim]
        
        Returns:
        - Extended feature set [batch_size, input_dim + n_physical_features]
        """
        batch_size = x.shape[0]
        
        # Initialize physical features list
        physical_features = []
        
        # If we have HOMO and LUMO indices, compute related features
        if self.homo_idx is not None and self.lumo_idx is not None:
            homo = x[:, self.homo_idx]
            lumo = x[:, self.lumo_idx]
            
            # 1. HOMO-LUMO gap (measure of chemical reactivity)
            gap = lumo - homo
            physical_features.append(gap.view(batch_size, 1))
            
            # 2. Electronegativity (ability to attract electrons)
            electronegativity = -0.5 * (homo + lumo)
            physical_features.append(electronegativity.view(batch_size, 1))
            
            # 3. Chemical hardness (resistance to charge transfer)
            hardness = 0.5 * (lumo - homo)
            physical_features.append(hardness.view(batch_size, 1))
            
            # 4. Electrophilicity index (electrophilic reactivity)
            # Avoid division by zero
            gap_safe = torch.clamp(gap, min=1e-6)
            electrophilicity = torch.pow(homo + lumo, 2) / (8 * gap_safe)
            physical_features.append(electrophilicity.view(batch_size, 1))
        
        # If physical features were computed, concatenate them with input
        if physical_features:
            physical_tensor = torch.cat(physical_features, dim=1)
            return torch.cat([x, physical_tensor], dim=1)
        else:
            return x
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        - x: Input quantum chemistry features [batch_size, input_dim]
        
        Returns:
        - Processed features [batch_size, output_dim]
        """
        # Compute physically meaningful composite features
        if self.homo_idx is not None and self.lumo_idx is not None:
            x_extended = self.compute_physical_features(x)
        else:
            x_extended = x
        
        # Initialize network on first forward pass after knowing actual input size
        if not self.initialized:
            input_size = x_extended.size(1)
            self.feature_net = nn.Sequential(
                nn.Linear(input_size, 16),
                nn.LeakyReLU(0.2),
                nn.BatchNorm1d(16),
                nn.Dropout(0.3),
                nn.Linear(16, self.output_dim)
            ).to(x.device)
            self.initialized = True
        
        # Apply feature transformation network
        return self.feature_net(x_extended)