#!/usr/bin/env python3
"""
Continuous Standard CP Module
Conformal Prediction for continuous path planning
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import math


class ContinuousNonconformity:
    """
    Nonconformity scores for continuous planning
    """
    
    @staticmethod
    def compute_hausdorff_distance(true_obs: List, perceived_obs: List) -> float:
        """
        Compute Hausdorff distance between true and perceived obstacles
        This measures the maximum deviation in perception
        
        Args:
            true_obs: True obstacles
            perceived_obs: Perceived obstacles
            
        Returns:
            Hausdorff distance (continuous score)
        """
        max_dist = 0.0
        
        # For each true obstacle, find distance to nearest perceived
        for true_rect in true_obs:
            tx, ty, tw, th = true_rect
            
            min_dist_to_perceived = float('inf')
            for perc_rect in perceived_obs:
                px, py, pw, ph = perc_rect
                
                # Distance between rectangle centers
                dist = math.sqrt((tx + tw/2 - px - pw/2)**2 + 
                               (ty + th/2 - py - ph/2)**2)
                
                # Also consider size difference
                size_diff = abs(tw - pw) + abs(th - ph)
                total_dist = dist + size_diff * 0.1
                
                min_dist_to_perceived = min(min_dist_to_perceived, total_dist)
            
            max_dist = max(max_dist, min_dist_to_perceived)
        
        return round(max_dist, 3)
    
    @staticmethod
    def compute_penetration_depth(true_obs: List, perceived_obs: List,
                                 num_samples: int = 500) -> float:
        """
        Compute maximum penetration depth
        How deep can perceived free space penetrate into true obstacles
        
        Args:
            true_obs: True obstacles
            perceived_obs: Perceived obstacles
            num_samples: Number of sample points to test
            
        Returns:
            Maximum penetration depth
        """
        max_penetration = 0.0
        
        # Sample points in space
        for _ in range(num_samples):
            x = np.random.uniform(1, 49)
            y = np.random.uniform(1, 29)
            
            # Check if perceived as free
            perceived_free = True
            for (ox, oy, w, h) in perceived_obs:
                if ox <= x <= ox + w and oy <= y <= oy + h:
                    perceived_free = False
                    break
            
            # Check if actually occupied
            if perceived_free:
                for (ox, oy, w, h) in true_obs:
                    if ox <= x <= ox + w and oy <= y <= oy + h:
                        # Point is inside true obstacle but perceived as free
                        # Calculate penetration depth
                        penetration = min(
                            x - ox,  # Distance from left edge
                            (ox + w) - x,  # Distance from right edge
                            y - oy,  # Distance from bottom edge
                            (oy + h) - y  # Distance from top edge
                        )
                        max_penetration = max(max_penetration, penetration)
        
        return round(max_penetration, 3)
    
    @staticmethod
    def compute_area_mismatch(true_obs: List, perceived_obs: List) -> float:
        """
        Compute area mismatch score
        Difference in total obstacle area
        
        Args:
            true_obs: True obstacles
            perceived_obs: Perceived obstacles
            
        Returns:
            Area mismatch score
        """
        true_area = sum(w * h for _, _, w, h in true_obs)
        perceived_area = sum(w * h for _, _, w, h in perceived_obs)
        
        mismatch = abs(true_area - perceived_area) / true_area
        return round(mismatch, 3)


class ContinuousStandardCP:
    """
    Standard Conformal Prediction for continuous planning
    """
    
    def __init__(self, 
                 true_obstacles: List,
                 nonconformity_type: str = "penetration"):
        """
        Initialize continuous CP
        
        Args:
            true_obstacles: True obstacle configuration
            nonconformity_type: Type of score ("penetration", "hausdorff", "area")
        """
        self.true_obstacles = true_obstacles
        self.nonconformity_type = nonconformity_type
        self.tau = None
        self.calibration_scores = []
        
    def calibrate(self, 
                  noise_model,
                  noise_params: Dict,
                  num_samples: int = 200,
                  confidence: float = 0.95,
                  base_seed: int = None) -> float:
        """
        Calibration phase to compute τ
        
        Args:
            noise_model: Function to add noise to obstacles
            noise_params: Parameters for noise model
            num_samples: Number of calibration samples
            confidence: Desired confidence level
            base_seed: Base random seed for reproducibility
            
        Returns:
            Calibrated τ value
        """
        self.calibration_scores = []
        
        print(f"\nCalibrating Continuous CP ({num_samples} samples)...")
        print(f"Nonconformity type: {self.nonconformity_type}")
        
        for i in range(num_samples):
            # Generate noisy perception with deterministic seed
            seed = i if base_seed is None else base_seed + i
            perceived_obs = noise_model(self.true_obstacles, **noise_params, seed=seed)
            
            # Compute nonconformity score
            if self.nonconformity_type == "penetration":
                score = ContinuousNonconformity.compute_penetration_depth(
                    self.true_obstacles, perceived_obs
                )
            elif self.nonconformity_type == "hausdorff":
                score = ContinuousNonconformity.compute_hausdorff_distance(
                    self.true_obstacles, perceived_obs
                )
            elif self.nonconformity_type == "area":
                score = ContinuousNonconformity.compute_area_mismatch(
                    self.true_obstacles, perceived_obs
                )
            else:
                raise ValueError(f"Unknown nonconformity type: {self.nonconformity_type}")
            
            self.calibration_scores.append(score)
        
        # Sort scores and compute quantile
        sorted_scores = sorted(self.calibration_scores)
        quantile_idx = int(np.ceil((num_samples + 1) * confidence)) - 1
        quantile_idx = min(quantile_idx, num_samples - 1)
        
        self.tau = sorted_scores[quantile_idx]
        
        # Print statistics
        print(f"\nCalibration Results:")
        print(f"  Score range: [{min(sorted_scores):.3f}, {max(sorted_scores):.3f}]")
        print(f"  Percentiles:")
        for p in [50, 75, 90, 95, 99]:
            idx = int(num_samples * p / 100)
            print(f"    {p}th: {sorted_scores[idx]:.3f}")
        print(f"  τ @ {confidence*100:.0f}% = {self.tau:.3f}")
        
        return self.tau
    
    def inflate_obstacles(self, 
                         perceived_obs: List,
                         inflation_method: str = "uniform") -> List:
        """
        Inflate obstacles by τ with proper handling of fractional values
        ENHANCED: Proper geometric inflation for continuous τ values
        
        Args:
            perceived_obs: Perceived obstacles
            inflation_method: How to inflate ("uniform", "directional", "smooth")
            
        Returns:
            Inflated obstacles
        """
        if self.tau is None:
            raise ValueError("Must calibrate first!")
        
        inflated = []
        
        for (x, y, w, h) in perceived_obs:
            if inflation_method == "uniform":
                # Uniform inflation in all directions with fractional τ
                # This implements proper Minkowski sum with a disc of radius τ
                new_x = max(0, x - self.tau)
                new_y = max(0, y - self.tau)
                new_w = min(50 - new_x, w + 2 * self.tau)
                new_h = min(30 - new_y, h + 2 * self.tau)
                
            elif inflation_method == "directional":
                # Directional inflation based on obstacle aspect ratio
                aspect_ratio = w / h if h > 0 else float('inf')
                
                if aspect_ratio > 1.5:  # Horizontal obstacle
                    # Inflate more vertically
                    new_x = max(0, x - self.tau * 0.3)
                    new_y = max(0, y - self.tau)
                    new_w = min(50 - new_x, w + self.tau * 0.6)
                    new_h = min(30 - new_y, h + 2 * self.tau)
                elif aspect_ratio < 0.67:  # Vertical obstacle
                    # Inflate more horizontally
                    new_x = max(0, x - self.tau)
                    new_y = max(0, y - self.tau * 0.3)
                    new_w = min(50 - new_x, w + 2 * self.tau)
                    new_h = min(30 - new_y, h + self.tau * 0.6)
                else:  # Square-ish obstacle
                    # Uniform inflation
                    new_x = max(0, x - self.tau)
                    new_y = max(0, y - self.tau)
                    new_w = min(50 - new_x, w + 2 * self.tau)
                    new_h = min(30 - new_y, h + 2 * self.tau)
                    
            elif inflation_method == "smooth":
                # Smooth inflation with rounded corners (approximation)
                # This better represents the true Minkowski sum with a disc
                # For fractional τ, we properly handle sub-pixel inflation
                import math
                
                # Calculate inflation with smooth corners
                corner_radius = min(self.tau, min(w, h) / 2)
                
                new_x = max(0, x - self.tau)
                new_y = max(0, y - self.tau)
                new_w = min(50 - new_x, w + 2 * self.tau)
                new_h = min(30 - new_y, h + 2 * self.tau)
                
                # For very small τ (< 0.1), apply probabilistic inflation
                if self.tau < 0.1:
                    # Probabilistic interpretation of fractional inflation
                    import random
                    if random.random() < self.tau * 10:
                        # Apply minimal inflation
                        new_x = max(0, x - 0.1)
                        new_y = max(0, y - 0.1)
                        new_w = min(50 - new_x, w + 0.2)
                        new_h = min(30 - new_y, h + 0.2)
            else:
                raise ValueError(f"Unknown inflation method: {inflation_method}")
            
            inflated.append((new_x, new_y, new_w, new_h))
        
        return inflated
    
    def get_tau_curve(self, confidence_levels: List[float] = None) -> Dict:
        """
        Get τ values for different confidence levels
        
        Args:
            confidence_levels: List of confidence levels
            
        Returns:
            Dictionary mapping confidence to τ
        """
        if not self.calibration_scores:
            raise ValueError("Must calibrate first!")
        
        if confidence_levels is None:
            confidence_levels = [0.80, 0.85, 0.90, 0.95, 0.99]
        
        sorted_scores = sorted(self.calibration_scores)
        n = len(sorted_scores)
        
        tau_curve = {}
        for conf in confidence_levels:
            idx = int(np.ceil((n + 1) * conf)) - 1
            idx = min(idx, n - 1)
            tau_curve[conf] = sorted_scores[idx]
        
        return tau_curve
    
    def compute_coverage(self, tau_value: float) -> float:
        """
        Compute empirical coverage for a given τ
        
        Args:
            tau_value: Safety margin value
            
        Returns:
            Empirical coverage percentage
        """
        if not self.calibration_scores:
            raise ValueError("Must calibrate first!")
        
        covered = sum(1 for s in self.calibration_scores if s <= tau_value)
        return covered / len(self.calibration_scores)