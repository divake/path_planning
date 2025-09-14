#!/usr/bin/env python3
"""
Custom loss function for adaptive tau learning.
Balances safety, efficiency, smoothness, and coverage requirements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class AdaptiveTauLoss(nn.Module):
    """
    Multi-component loss function for learning adaptive tau.
    
    Components:
    1. Safety Loss: Penalizes safety violations (tau < clearance)
    2. Efficiency Loss: Encourages smaller tau values for efficiency
    3. Smoothness Loss: Encourages smooth tau changes along path
    4. Coverage Loss: Ensures statistical validity (90% coverage)
    """
    
    def __init__(self, 
                 safety_weight: float = 1.0,  # Balanced weights
                 efficiency_weight: float = 0.3,  # Reduced
                 smoothness_weight: float = 0.2,
                 coverage_weight: float = 1.0,
                 violation_penalty: float = 1.5):  # Reduced
        """
        Initialize loss function.
        
        Args:
            safety_weight: Weight for safety violations
            efficiency_weight: Weight for efficiency (smaller tau)
            smoothness_weight: Weight for smoothness along path
            coverage_weight: Weight for coverage guarantee
            violation_penalty: Extra penalty for safety violations
        """
        super(AdaptiveTauLoss, self).__init__()
        
        self.safety_weight = safety_weight
        self.efficiency_weight = efficiency_weight
        self.smoothness_weight = smoothness_weight
        self.coverage_weight = coverage_weight
        self.violation_penalty = violation_penalty
        
        # Track statistics for monitoring
        self.total_samples = 0
        self.violations = 0
        self.avg_tau = 0.0
        
    def forward(self,
                predicted_tau: torch.Tensor,
                actual_clearance: torch.Tensor,
                features: Optional[torch.Tensor] = None,
                path_indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute multi-component loss.
        
        Args:
            predicted_tau: Predicted safety margins (batch_size,)
            actual_clearance: Actual clearances to obstacles (batch_size,)
            features: Input features for context-aware penalties (batch_size, num_features)
            path_indices: Path index for each waypoint for smoothness (batch_size,)
        
        Returns:
            Total loss value
        """
        batch_size = predicted_tau.shape[0]
        
        # 1. SAFETY LOSS - Critical for conformal guarantee
        # Use Huber loss (smooth L1) to prevent gradient explosion
        safety_diff = predicted_tau - actual_clearance
        
        # Asymmetric Huber loss: more penalty for unsafe predictions
        safety_loss = torch.mean(
            torch.where(
                safety_diff >= 0,  # Safe predictions (tau >= clearance)
                F.smooth_l1_loss(predicted_tau, actual_clearance, reduction='none') * 0.5,
                F.smooth_l1_loss(predicted_tau, actual_clearance, reduction='none') * 2.0  # More penalty for unsafe
            )
        )
        
        # 2. EFFICIENCY LOSS - Encourage reasonable tau values
        # Target range: 0.2 to 0.4 meters (reasonable safety margins)
        target_tau = 0.3  # Target tau value
        efficiency_loss = F.smooth_l1_loss(predicted_tau, 
                                          torch.full_like(predicted_tau, target_tau))
        
        # 3. SMOOTHNESS LOSS - Encourage smooth changes along path
        smoothness_loss = torch.tensor(0.0, device=predicted_tau.device)
        if path_indices is not None:
            unique_paths = torch.unique(path_indices)
            for path_id in unique_paths:
                mask = (path_indices == path_id)
                path_tau = predicted_tau[mask]
                
                if len(path_tau) > 1:
                    # Compute differences between consecutive waypoints
                    tau_diff = torch.diff(path_tau)
                    # Use L2 norm for smoothness
                    smoothness_loss += torch.mean(tau_diff ** 2)
            
            smoothness_loss /= max(len(unique_paths), 1)
        
        # 4. COVERAGE LOSS - Ensure 90% statistical validity
        # Count how many predictions are valid (tau >= actual_clearance)
        valid_predictions = (predicted_tau >= actual_clearance).float()
        empirical_coverage = torch.mean(valid_predictions)
        target_coverage = 0.9  # 90% coverage guarantee
        
        # Soft penalty around target coverage
        coverage_loss = F.smooth_l1_loss(empirical_coverage, 
                                        torch.tensor(target_coverage, device=predicted_tau.device))
        
        # Additional penalty if coverage is too low
        if empirical_coverage < target_coverage:
            coverage_loss += (target_coverage - empirical_coverage) * self.violation_penalty
        
        # 5. REGULARIZATION - Prevent extreme values
        # Penalize very large or very negative tau values
        reg_loss = torch.mean(
            F.relu(predicted_tau - 1.0) * 0.1 +  # Penalty for tau > 1.0m
            F.relu(-0.05 - predicted_tau) * 0.5  # Penalty for tau < -0.05m
        )
        
        # Combine all losses
        total_loss = (
            self.safety_weight * safety_loss +
            self.efficiency_weight * efficiency_loss +
            self.smoothness_weight * smoothness_loss +
            self.coverage_weight * coverage_loss +
            0.1 * reg_loss  # Small regularization weight
        )
        
        # Update statistics
        with torch.no_grad():
            self.total_samples += batch_size
            self.violations += torch.sum(predicted_tau < actual_clearance).item()
            self.avg_tau = 0.9 * self.avg_tau + 0.1 * torch.mean(predicted_tau).item()
        
        return total_loss
    
    def get_stats(self) -> dict:
        """Get loss statistics for monitoring."""
        violation_rate = self.violations / max(self.total_samples, 1)
        return {
            'violation_rate': violation_rate,
            'coverage': 1.0 - violation_rate,
            'avg_tau': self.avg_tau,
            'total_samples': self.total_samples
        }
    
    def reset_stats(self):
        """Reset statistics."""
        self.total_samples = 0
        self.violations = 0
        self.avg_tau = 0.0