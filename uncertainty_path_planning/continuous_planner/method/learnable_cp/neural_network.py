#!/usr/bin/env python3
"""
Neural Network Module for Learnable Conformal Prediction
Predicts adaptive tau values for each waypoint based on local features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple


class AdaptiveTauNetwork(nn.Module):
    """
    Neural network that predicts tau (safety margin) for each waypoint.
    
    Key Design Choices:
    1. No clamping on output - allows negative nonconformity scores
    2. BatchNorm for stability during training
    3. Dropout for regularization
    4. Residual connections for better gradient flow
    """
    
    def __init__(self, 
                 input_dim: int = 20,
                 hidden_dims: list = [128, 64, 32],
                 dropout_rate: float = 0.2,
                 use_batch_norm: bool = True,
                 use_residual: bool = True):
        """
        Initialize the adaptive tau network.
        
        Args:
            input_dim: Number of input features
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout probability
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super(AdaptiveTauNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Build the network layers
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList() if use_batch_norm else None
        self.dropouts = nn.ModuleList()
        
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_dims):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer - single tau value (no activation to allow full range)
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Residual projection layers if dimensions don't match
        if use_residual:
            self.residual_projections = nn.ModuleList()
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                if prev_dim != hidden_dim:
                    self.residual_projections.append(nn.Linear(prev_dim, hidden_dim))
                else:
                    self.residual_projections.append(nn.Identity())
                prev_dim = hidden_dim
        
        # Initialize weights
        self._initialize_weights()
        
        # Track training statistics
        self.register_buffer('running_mean', torch.zeros(1))
        self.register_buffer('running_std', torch.ones(1))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Predicted tau values [batch_size, 1]
        """
        # Input validation
        assert x.shape[-1] == self.input_dim, f"Expected input dim {self.input_dim}, got {x.shape[-1]}"
        
        # Store input for potential skip connections
        identity = x
        
        # Forward through hidden layers
        for i, layer in enumerate(self.layers):
            # Linear transformation
            x = layer(x)
            
            # Batch normalization
            if self.use_batch_norm and x.shape[0] > 1:  # BatchNorm needs batch_size > 1
                x = self.batch_norms[i](x)
            
            # Activation
            x = F.relu(x)
            
            # Residual connection
            if self.use_residual:
                residual = self.residual_projections[i](identity)
                x = x + residual
                identity = x
            
            # Dropout
            x = self.dropouts[i](x)
        
        # Output layer - no activation to allow negative values
        tau = self.output_layer(x)
        
        # Update running statistics during training
        if self.training:
            with torch.no_grad():
                self.num_batches_tracked += 1
                batch_mean = tau.mean()
                batch_std = tau.std()
                
                momentum = 0.1
                self.running_mean = (1 - momentum) * self.running_mean + momentum * batch_mean
                self.running_std = (1 - momentum) * self.running_std + momentum * batch_std
        
        return tau
    
    def predict_tau(self, features: np.ndarray) -> float:
        """
        Predict tau for a single waypoint (inference mode).
        
        Args:
            features: Feature vector [input_dim]
            
        Returns:
            Predicted tau value
        """
        self.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            tau = self.forward(features_tensor)
            return tau.item()
    
    def get_model_stats(self) -> Dict:
        """Get model statistics for monitoring"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'running_mean': self.running_mean.item(),
            'running_std': self.running_std.item(),
            'num_batches': self.num_batches_tracked.item()
        }


class EnsembleTauNetwork(nn.Module):
    """
    Ensemble of networks for uncertainty estimation.
    Uses multiple networks and aggregates predictions.
    """
    
    def __init__(self, 
                 num_models: int = 5,
                 input_dim: int = 20,
                 hidden_dims: list = [128, 64, 32]):
        """
        Initialize ensemble of networks.
        
        Args:
            num_models: Number of models in ensemble
            input_dim: Number of input features
            hidden_dims: Hidden layer dimensions
        """
        super(EnsembleTauNetwork, self).__init__()
        
        self.num_models = num_models
        self.models = nn.ModuleList([
            AdaptiveTauNetwork(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                dropout_rate=0.2,  # Different dropout for diversity
                use_batch_norm=True,
                use_residual=True
            ) for _ in range(num_models)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input features [batch_size, input_dim]
            
        Returns:
            Mean predictions and uncertainty estimates
        """
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        predictions = torch.stack(predictions, dim=0)  # [num_models, batch_size, 1]
        
        # Mean and std across ensemble
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def predict_with_uncertainty(self, features: np.ndarray) -> Tuple[float, float]:
        """
        Predict tau with uncertainty estimate.
        
        Args:
            features: Feature vector
            
        Returns:
            (mean_tau, std_tau)
        """
        self.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(features).unsqueeze(0)
            mean_tau, std_tau = self.forward(features_tensor)
            return mean_tau.item(), std_tau.item()