#!/usr/bin/env python3
"""
Loss Functions for Learnable Conformal Prediction
Defines the losses used to train the neural network for adaptive tau prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional


class AdaptiveTauLoss(nn.Module):
    """
    Combined loss function for training adaptive tau predictor.
    
    Components:
    1. Safety Loss: Penalizes violations (when actual clearance < predicted tau)
    2. Efficiency Loss: Encourages smaller tau values for efficiency
    3. Smoothness Loss: Encourages smooth tau changes along path
    4. Coverage Loss: Ensures statistical validity (90% coverage)
    """
    
    def __init__(self, 
                 safety_weight: float = 10.0,
                 efficiency_weight: float = 1.0,
                 smoothness_weight: float = 0.5,
                 coverage_weight: float = 2.0,
                 violation_penalty: float = 5.0):
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
        self.reset_stats()
    
    def forward(self, 
                predicted_tau: torch.Tensor,
                actual_clearance: torch.Tensor,
                features: torch.Tensor,
                path_indices: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        """
        Compute combined loss.
        
        Args:
            predicted_tau: Predicted safety margins [batch_size]
            actual_clearance: Actual clearances observed [batch_size]
            features: Input features [batch_size, num_features]
            path_indices: Indices indicating which path each waypoint belongs to
            
        Returns:
            Total loss and dictionary of individual loss components
        """
        batch_size = predicted_tau.shape[0]
        
        # 1. SAFETY LOSS - Heavily penalize violations
        violations = F.relu(predicted_tau - actual_clearance)  # Positive when tau > clearance
        safety_loss = torch.mean(violations * self.violation_penalty)
        
        # Additional exponential penalty for large violations
        large_violations = F.relu(violations - 0.1)  # Violations > 10cm
        safety_loss += torch.mean(torch.exp(large_violations) - 1.0)
        
        # 2. EFFICIENCY LOSS - Encourage smaller tau values
        # Use asymmetric loss: small penalty for reasonable tau, large for excessive
        reasonable_tau = 0.3  # 30cm is reasonable safety margin
        efficiency_loss = torch.mean(
            torch.where(
                predicted_tau < reasonable_tau,
                0.1 * predicted_tau,  # Small penalty for tau < 30cm
                (predicted_tau - reasonable_tau) ** 2  # Quadratic penalty for larger tau
            )
        )
        
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
                    smoothness_loss += torch.mean(tau_diff ** 2)
            
            smoothness_loss /= max(len(unique_paths), 1)
        
        # 4. COVERAGE LOSS - Ensure 90% statistical validity
        # Count how many predictions are valid (tau >= actual_clearance)
        valid_predictions = (predicted_tau >= actual_clearance).float()
        empirical_coverage = torch.mean(valid_predictions)
        target_coverage = 0.9  # 90% coverage guarantee
        
        # Penalize deviation from target coverage
        coverage_loss = (empirical_coverage - target_coverage) ** 2
        
        # Additional penalty if coverage is too low
        if empirical_coverage < target_coverage:
            coverage_loss += 2.0 * (target_coverage - empirical_coverage)
        
        # 5. FEATURE-AWARE ADJUSTMENTS
        # Extract specific features that should influence tau
        min_clearance_feature = features[:, 0]  # First feature is min_clearance
        corridor_feature = features[:, 15] if features.shape[1] > 15 else torch.zeros_like(predicted_tau)
        
        # Penalize small tau in tight spaces
        tight_space_mask = (min_clearance_feature < 0.2)  # Clearance < 20cm (normalized)
        tight_space_penalty = torch.mean(
            F.relu(0.4 - predicted_tau[tight_space_mask])
        ) if tight_space_mask.any() else torch.tensor(0.0, device=predicted_tau.device)
        
        # Encourage larger tau in corridors
        corridor_penalty = torch.mean(
            F.relu(0.35 - predicted_tau[corridor_feature > 0.5])
        ) if (corridor_feature > 0.5).any() else torch.tensor(0.0, device=predicted_tau.device)
        
        # Combine all losses
        total_loss = (
            self.safety_weight * safety_loss +
            self.efficiency_weight * efficiency_loss +
            self.smoothness_weight * smoothness_loss +
            self.coverage_weight * coverage_loss +
            2.0 * tight_space_penalty +
            1.5 * corridor_penalty
        )
        
        # Store statistics
        with torch.no_grad():
            self.num_violations += torch.sum(violations > 0).item()
            self.total_samples += batch_size
            self.avg_tau += torch.mean(predicted_tau).item()
            self.avg_clearance += torch.mean(actual_clearance).item()
            self.num_batches += 1
        
        # Return total loss and components for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'safety_loss': safety_loss.item(),
            'efficiency_loss': efficiency_loss.item(),
            'smoothness_loss': smoothness_loss.item(),
            'coverage_loss': coverage_loss.item(),
            'tight_space_penalty': tight_space_penalty.item(),
            'corridor_penalty': corridor_penalty.item(),
            'empirical_coverage': empirical_coverage.item(),
            'avg_predicted_tau': torch.mean(predicted_tau).item(),
            'avg_actual_clearance': torch.mean(actual_clearance).item(),
            'violation_rate': torch.mean(valid_predictions).item()
        }
        
        return total_loss, loss_dict
    
    def reset_stats(self):
        """Reset running statistics"""
        self.num_violations = 0
        self.total_samples = 0
        self.avg_tau = 0.0
        self.avg_clearance = 0.0
        self.num_batches = 0
    
    def get_stats(self) -> Dict:
        """Get accumulated statistics"""
        if self.num_batches == 0:
            return {}
        
        return {
            'violation_rate': self.num_violations / max(self.total_samples, 1),
            'avg_tau': self.avg_tau / self.num_batches,
            'avg_clearance': self.avg_clearance / self.num_batches,
            'total_samples': self.total_samples
        }


class QuantileLoss(nn.Module):
    """
    Quantile loss for directly optimizing the 90th percentile.
    Used as an alternative or additional loss for conformal prediction.
    """
    
    def __init__(self, quantile: float = 0.9):
        """
        Initialize quantile loss.
        
        Args:
            quantile: Target quantile (0.9 for 90th percentile)
        """
        super(QuantileLoss, self).__init__()
        self.quantile = quantile
    
    def forward(self, predicted: torch.Tensor, actual: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball loss for quantile regression.
        
        Args:
            predicted: Predicted quantile values
            actual: Actual values
            
        Returns:
            Quantile loss
        """
        errors = actual - predicted
        
        # Asymmetric loss based on quantile
        loss = torch.where(
            errors >= 0,
            self.quantile * errors,
            (self.quantile - 1) * errors
        )
        
        return torch.mean(loss)


class WeightedMSELoss(nn.Module):
    """
    Weighted MSE loss that gives more importance to safety-critical waypoints.
    """
    
    def __init__(self):
        super(WeightedMSELoss, self).__init__()
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, 
                predicted_tau: torch.Tensor,
                target_tau: torch.Tensor,
                features: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted MSE loss.
        
        Args:
            predicted_tau: Predicted tau values
            target_tau: Target tau values (from calibration)
            features: Input features for weight computation
            
        Returns:
            Weighted MSE loss
        """
        # Base MSE
        mse_loss = self.mse(predicted_tau, target_tau)
        
        # Compute weights based on features
        # Higher weight for: tight spaces, corridors, near obstacles
        min_clearance = features[:, 0]
        corridor_indicator = features[:, 15] if features.shape[1] > 15 else torch.zeros_like(predicted_tau)
        
        # Weight calculation
        weights = torch.ones_like(predicted_tau)
        weights += (1.0 - min_clearance) * 2.0  # Higher weight for smaller clearances
        weights += corridor_indicator * 1.5  # Higher weight in corridors
        
        # Normalize weights
        weights = weights / torch.mean(weights)
        
        # Apply weights
        weighted_loss = mse_loss * weights
        
        return torch.mean(weighted_loss)


class ContrastiveSafetyLoss(nn.Module):
    """
    Contrastive loss that learns to distinguish between safe and unsafe tau values.
    """
    
    def __init__(self, margin: float = 0.1):
        """
        Initialize contrastive loss.
        
        Args:
            margin: Margin for contrastive loss
        """
        super(ContrastiveSafetyLoss, self).__init__()
        self.margin = margin
    
    def forward(self,
                tau_safe: torch.Tensor,
                tau_unsafe: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss between safe and unsafe tau predictions.
        
        Args:
            tau_safe: Tau values for safe waypoints
            tau_unsafe: Tau values for unsafe waypoints
            
        Returns:
            Contrastive loss
        """
        # Ensure safe tau > unsafe tau + margin
        loss = F.relu(tau_unsafe - tau_safe + self.margin)
        return torch.mean(loss)


def compute_calibration_loss(predicted_tau: torch.Tensor,
                            nonconformity_scores: torch.Tensor,
                            quantile: float = 0.9) -> torch.Tensor:
    """
    Loss for calibrating tau to match the desired quantile of nonconformity scores.
    
    Args:
        predicted_tau: Predicted tau values [batch_size]
        nonconformity_scores: Actual nonconformity scores [batch_size]
        quantile: Target quantile
        
    Returns:
        Calibration loss
    """
    # Compute empirical quantile of nonconformity scores
    empirical_quantile = torch.quantile(nonconformity_scores, quantile)
    
    # Loss encourages predicted tau to match empirical quantile
    calibration_loss = F.mse_loss(torch.mean(predicted_tau), empirical_quantile)
    
    # Additional penalty if too many violations
    violation_rate = torch.mean((nonconformity_scores > predicted_tau).float())
    target_violation_rate = 1.0 - quantile
    
    violation_penalty = F.relu(violation_rate - target_violation_rate) * 10.0
    
    return calibration_loss + violation_penalty