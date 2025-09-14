#!/usr/bin/env python3
"""
Tau Calculator for Learnable Conformal Prediction
Manages the computation and calibration of adaptive tau values
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging


class AdaptiveTauCalculator:
    """
    Calculates and manages adaptive tau (safety margin) values.
    
    Key responsibilities:
    1. Compute tau using neural network predictions
    2. Maintain calibration statistics
    3. Apply safety constraints and bounds
    4. Track performance metrics
    """
    
    def __init__(self, 
                 neural_network: torch.nn.Module,
                 feature_extractor,
                 config: Dict):
        """
        Initialize tau calculator.
        
        Args:
            neural_network: Trained neural network for tau prediction
            feature_extractor: Feature extraction module
            config: Configuration dictionary
        """
        self.neural_network = neural_network
        self.feature_extractor = feature_extractor
        self.config = config
        
        # Tau bounds
        self.tau_min = config.get('tau_min', 0.05)  # 5cm minimum
        self.tau_max = config.get('tau_max', 0.50)  # 50cm maximum
        self.tau_default = config.get('tau_default', 0.30)  # 30cm default
        
        # Calibration parameters
        self.target_coverage = 0.9  # 90% coverage guarantee
        self.calibration_window = 100  # Number of recent predictions for calibration
        self.calibration_alpha = 0.1  # Learning rate for calibration updates
        
        # Calibration statistics
        self.nonconformity_scores = deque(maxlen=self.calibration_window)
        self.coverage_history = deque(maxlen=50)
        self.calibration_factor = 1.0  # Multiplicative calibration factor
        
        # Performance tracking
        self.total_predictions = 0
        self.total_violations = 0
        self.tau_history = []
        
        logging.info(f"AdaptiveTauCalculator initialized with bounds [{self.tau_min}, {self.tau_max}]")
    
    def compute_tau_for_waypoint(self,
                                waypoint: np.ndarray,
                                path: List[np.ndarray],
                                waypoint_idx: int,
                                occupancy_grid: np.ndarray,
                                origin: np.ndarray,
                                resolution: float,
                                noise_type: str = None) -> float:
        """
        Compute adaptive tau for a single waypoint.
        
        Args:
            waypoint: Current waypoint [x, y]
            path: Full path as list of waypoints
            waypoint_idx: Index of current waypoint
            occupancy_grid: Occupancy grid
            origin: Map origin
            resolution: Map resolution
            noise_type: Type of noise
            
        Returns:
            Adaptive tau value in meters
        """
        # Extract features
        features = self.feature_extractor.extract_features(
            waypoint, path, waypoint_idx, 
            occupancy_grid, origin, resolution, noise_type
        )
        
        # Convert to tensor
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Get neural network prediction
        self.neural_network.eval()
        with torch.no_grad():
            tau_raw = self.neural_network(features_tensor).item()
        
        # Apply calibration factor
        tau_calibrated = tau_raw * self.calibration_factor
        
        # Apply safety bounds
        tau_bounded = np.clip(tau_calibrated, self.tau_min, self.tau_max)
        
        # Apply feature-based adjustments
        tau_adjusted = self._apply_feature_adjustments(tau_bounded, features)
        
        # Track for statistics
        self.tau_history.append(tau_adjusted)
        self.total_predictions += 1
        
        return tau_adjusted
    
    def compute_tau_for_path(self,
                            path: List[np.ndarray],
                            occupancy_grid: np.ndarray,
                            origin: np.ndarray,
                            resolution: float,
                            noise_type: str = None) -> List[float]:
        """
        Compute adaptive tau for entire path.
        
        Args:
            path: Full path as list of waypoints
            occupancy_grid: Occupancy grid
            origin: Map origin
            resolution: Map resolution
            noise_type: Type of noise
            
        Returns:
            List of tau values for each waypoint
        """
        tau_values = []
        
        for idx, waypoint in enumerate(path):
            tau = self.compute_tau_for_waypoint(
                waypoint, path, idx,
                occupancy_grid, origin, resolution, noise_type
            )
            tau_values.append(tau)
        
        # Apply smoothing if enabled
        if self.config.get('smooth_tau', True):
            tau_values = self._smooth_tau_values(tau_values)
        
        return tau_values
    
    def update_calibration(self, 
                          predicted_tau: float,
                          actual_clearance: float):
        """
        Update calibration based on observed clearance.
        
        Args:
            predicted_tau: Predicted tau value
            actual_clearance: Actual observed clearance
        """
        # Compute nonconformity score
        nonconformity = predicted_tau - actual_clearance
        self.nonconformity_scores.append(nonconformity)
        
        # Track violations
        if actual_clearance < predicted_tau:
            self.total_violations += 1
        
        # Update calibration factor if enough data
        if len(self.nonconformity_scores) >= 20:
            self._update_calibration_factor()
    
    def _update_calibration_factor(self):
        """Update calibration factor based on recent performance"""
        if not self.nonconformity_scores:
            return
        
        # Compute empirical coverage
        violations = sum(1 for score in self.nonconformity_scores if score > 0)
        empirical_coverage = 1.0 - (violations / len(self.nonconformity_scores))
        
        self.coverage_history.append(empirical_coverage)
        
        # Adjust calibration factor
        if empirical_coverage < self.target_coverage - 0.05:
            # Coverage too low, increase tau
            self.calibration_factor *= (1 + self.calibration_alpha)
        elif empirical_coverage > self.target_coverage + 0.05:
            # Coverage too high, can decrease tau for efficiency
            self.calibration_factor *= (1 - self.calibration_alpha * 0.5)
        
        # Bound calibration factor
        self.calibration_factor = np.clip(self.calibration_factor, 0.5, 2.0)
        
        logging.debug(f"Calibration updated: coverage={empirical_coverage:.2f}, factor={self.calibration_factor:.2f}")
    
    def _apply_feature_adjustments(self, tau: float, features: np.ndarray) -> float:
        """
        Apply feature-based adjustments to tau.
        
        Args:
            tau: Base tau value
            features: Feature vector
            
        Returns:
            Adjusted tau value
        """
        # Extract key features (indices based on feature extractor)
        min_clearance = features[0] * 5.0  # Denormalize (was normalized by max_clearance=5.0)
        is_corridor = features[15]
        is_near_corner = features[17]
        
        # Adjustments based on environment
        if is_corridor > 0.5:
            # In corridor, need more conservative tau
            tau = max(tau, 0.35)
        
        if is_near_corner > 0.5:
            # Near corner, increase tau
            tau *= 1.2
        
        if min_clearance < 0.3:
            # Very tight space, ensure minimum safety
            tau = max(tau, 0.4)
        elif min_clearance > 2.0:
            # Open space, can be more efficient
            tau = min(tau, 0.25)
        
        # Final bounds check
        return np.clip(tau, self.tau_min, self.tau_max)
    
    def _smooth_tau_values(self, tau_values: List[float]) -> List[float]:
        """
        Apply smoothing to tau values along path.
        
        Args:
            tau_values: Raw tau values
            
        Returns:
            Smoothed tau values
        """
        if len(tau_values) <= 2:
            return tau_values
        
        smoothed = tau_values.copy()
        window_size = 3
        
        for i in range(1, len(tau_values) - 1):
            # Simple moving average
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(tau_values), i + window_size // 2 + 1)
            smoothed[i] = np.mean(tau_values[start_idx:end_idx])
        
        # Ensure smoothing doesn't violate safety
        for i in range(len(smoothed)):
            smoothed[i] = max(smoothed[i], tau_values[i] * 0.9)  # Don't reduce by more than 10%
        
        return smoothed
    
    def compute_quantile_tau(self, 
                            nonconformity_scores: List[float],
                            quantile: float = 0.9) -> float:
        """
        Compute tau as quantile of nonconformity scores (fallback method).
        
        Args:
            nonconformity_scores: List of nonconformity scores
            quantile: Target quantile
            
        Returns:
            Tau value
        """
        if not nonconformity_scores:
            return self.tau_default
        
        sorted_scores = sorted(nonconformity_scores)
        idx = int(np.ceil(quantile * len(sorted_scores))) - 1
        idx = min(idx, len(sorted_scores) - 1)
        
        tau = sorted_scores[idx]
        return np.clip(tau, self.tau_min, self.tau_max)
    
    def get_statistics(self) -> Dict:
        """Get current statistics"""
        if self.total_predictions == 0:
            return {
                'total_predictions': 0,
                'violation_rate': 0.0,
                'avg_tau': self.tau_default,
                'calibration_factor': self.calibration_factor
            }
        
        return {
            'total_predictions': self.total_predictions,
            'violation_rate': self.total_violations / self.total_predictions,
            'avg_tau': np.mean(self.tau_history) if self.tau_history else self.tau_default,
            'std_tau': np.std(self.tau_history) if self.tau_history else 0.0,
            'min_tau': min(self.tau_history) if self.tau_history else self.tau_min,
            'max_tau': max(self.tau_history) if self.tau_history else self.tau_max,
            'calibration_factor': self.calibration_factor,
            'recent_coverage': np.mean(self.coverage_history) if self.coverage_history else 0.0
        }
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.total_predictions = 0
        self.total_violations = 0
        self.tau_history = []
        self.nonconformity_scores.clear()
        self.coverage_history.clear()
        self.calibration_factor = 1.0


class HybridTauCalculator:
    """
    Hybrid tau calculator that combines neural network predictions with rule-based fallbacks.
    """
    
    def __init__(self,
                 neural_calculator: AdaptiveTauCalculator,
                 config: Dict):
        """
        Initialize hybrid calculator.
        
        Args:
            neural_calculator: Neural network-based calculator
            config: Configuration
        """
        self.neural_calculator = neural_calculator
        self.config = config
        
        # Fallback tau values for different scenarios
        self.fallback_tau = {
            'transparency': 0.32,
            'occlusion': 0.33,
            'localization': 0.32,
            'combined': 0.35,
            'default': 0.30
        }
        
        # Confidence threshold for using neural predictions
        self.confidence_threshold = 0.7
    
    def compute_tau(self,
                   waypoint: np.ndarray,
                   path: List[np.ndarray],
                   waypoint_idx: int,
                   occupancy_grid: np.ndarray,
                   origin: np.ndarray,
                   resolution: float,
                   noise_type: str = None,
                   use_neural: bool = True) -> Tuple[float, str]:
        """
        Compute tau using hybrid approach.
        
        Args:
            waypoint: Current waypoint
            path: Full path
            waypoint_idx: Waypoint index
            occupancy_grid: Occupancy grid
            origin: Map origin
            resolution: Map resolution
            noise_type: Noise type
            use_neural: Whether to use neural network
            
        Returns:
            (tau_value, method_used)
        """
        if use_neural:
            try:
                # Try neural network prediction
                tau = self.neural_calculator.compute_tau_for_waypoint(
                    waypoint, path, waypoint_idx,
                    occupancy_grid, origin, resolution, noise_type
                )
                
                # Validate prediction
                if self._is_valid_prediction(tau, waypoint, occupancy_grid, origin, resolution):
                    return tau, "neural"
                else:
                    logging.debug("Neural prediction invalid, using fallback")
            except Exception as e:
                logging.warning(f"Neural prediction failed: {e}")
        
        # Use fallback
        tau = self.fallback_tau.get(noise_type, self.fallback_tau['default'])
        return tau, "rule_based"
    
    def _is_valid_prediction(self,
                           tau: float,
                           waypoint: np.ndarray,
                           occupancy_grid: np.ndarray,
                           origin: np.ndarray,
                           resolution: float) -> bool:
        """
        Validate neural network prediction.
        
        Args:
            tau: Predicted tau
            waypoint: Current waypoint
            occupancy_grid: Occupancy grid
            origin: Map origin
            resolution: Map resolution
            
        Returns:
            True if prediction is valid
        """
        # Check bounds
        if tau < 0.01 or tau > 1.0:
            return False
        
        # Check if tau is reasonable for environment
        grid_x = int((waypoint[0] - origin[0]) / resolution)
        grid_y = int((waypoint[1] - origin[1]) / resolution)
        
        # Simple clearance check
        h, w = occupancy_grid.shape
        if 0 <= grid_x < w and 0 <= grid_y < h:
            # If in occupied space, tau should be large
            if occupancy_grid[grid_y, grid_x] > 50 and tau < 0.4:
                return False
        
        return True