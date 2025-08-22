#!/usr/bin/env python3
"""
Phase 2: Uncertainty wrapper around existing Hybrid A* planner
This wraps the existing, tested planner with learnable uncertainty
"""

import numpy as np
import sys
import os
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

# Add path to existing planner
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from HybridAstarPlanner import hybrid_astar
from HybridAstarPlanner.hybrid_astar import C, hybrid_astar_planning

class UncertaintyNetwork(nn.Module):
    """Neural network for predicting uncertainty based on local environment"""
    
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Output between 0 and 1
        )
        
    def forward(self, x):
        return self.network(x)

class UncertaintyAwareHybridAstar:
    """Wrapper around existing Hybrid A* with uncertainty quantification"""
    
    def __init__(self, method='learnable_cp'):
        """
        Initialize uncertainty-aware planner
        
        Args:
            method: 'naive', 'ensemble', or 'learnable_cp'
        """
        self.method = method
        self.config = C()  # Use existing configuration
        
        if method == 'learnable_cp':
            self.uncertainty_model = UncertaintyNetwork()
            # Load pre-trained weights if available
            model_path = 'icra_implementation/phase2/uncertainty_model.pth'
            if os.path.exists(model_path):
                self.uncertainty_model.load_state_dict(torch.load(model_path))
                self.uncertainty_model.eval()
        
    def extract_features(self, x, y, yaw, ox, oy, gx, gy):
        """Extract features for uncertainty prediction"""
        features = []
        
        # 1. Distance to nearest obstacle
        if ox and oy:
            min_dist = min(np.sqrt((x - ox_i)**2 + (y - oy_i)**2) 
                          for ox_i, oy_i in zip(ox, oy))
            features.append(min(min_dist / 10.0, 1.0))
        else:
            features.append(1.0)
        
        # 2. Obstacle density in 5m radius
        density_5m = sum(1 for ox_i, oy_i in zip(ox, oy) 
                        if np.sqrt((x - ox_i)**2 + (y - oy_i)**2) < 5.0)
        features.append(min(density_5m / 50.0, 1.0))
        
        # 3. Obstacle density in 10m radius
        density_10m = sum(1 for ox_i, oy_i in zip(ox, oy) 
                         if np.sqrt((x - ox_i)**2 + (y - oy_i)**2) < 10.0)
        features.append(min(density_10m / 100.0, 1.0))
        
        # 4. Distance to goal
        goal_dist = np.sqrt((x - gx)**2 + (y - gy)**2)
        features.append(min(goal_dist / 50.0, 1.0))
        
        # 5. Heading alignment with goal
        goal_angle = np.arctan2(gy - y, gx - x)
        angle_diff = abs(np.arctan2(np.sin(goal_angle - yaw), 
                                   np.cos(goal_angle - yaw)))
        features.append(angle_diff / np.pi)
        
        # 6. Passage width estimation
        passage_width = self.estimate_passage_width(x, y, yaw, ox, oy)
        features.append(min(passage_width / 10.0, 1.0))
        
        # 7. Number of clear directions
        clear_dirs = self.count_clear_directions(x, y, ox, oy)
        features.append(clear_dirs / 8.0)
        
        # 8. Local path curvature requirement
        curvature = self.estimate_required_curvature(x, y, yaw, gx, gy)
        features.append(min(curvature / 2.0, 1.0))
        
        # 9. Obstacle asymmetry
        asymmetry = self.calculate_obstacle_asymmetry(x, y, ox, oy)
        features.append(asymmetry)
        
        # 10. Collision risk
        risk = self.estimate_collision_risk(x, y, yaw, ox, oy)
        features.append(risk)
        
        return features
    
    def estimate_passage_width(self, x, y, yaw, ox, oy):
        """Estimate width of passage ahead"""
        if not ox or not oy:
            return 10.0
            
        # Check perpendicular to heading
        left_dist = 10.0
        right_dist = 10.0
        
        for dist in np.linspace(0.5, 5.0, 10):
            # Left side
            left_x = x + dist * np.cos(yaw + np.pi/2)
            left_y = y + dist * np.sin(yaw + np.pi/2)
            for ox_i, oy_i in zip(ox, oy):
                if np.sqrt((left_x - ox_i)**2 + (left_y - oy_i)**2) < 0.5:
                    left_dist = min(left_dist, dist)
                    break
            
            # Right side
            right_x = x + dist * np.cos(yaw - np.pi/2)
            right_y = y + dist * np.sin(yaw - np.pi/2)
            for ox_i, oy_i in zip(ox, oy):
                if np.sqrt((right_x - ox_i)**2 + (right_y - oy_i)**2) < 0.5:
                    right_dist = min(right_dist, dist)
                    break
        
        return left_dist + right_dist
    
    def count_clear_directions(self, x, y, ox, oy):
        """Count number of clear directions from current position"""
        if not ox or not oy:
            return 8
            
        clear_count = 0
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            is_clear = True
            for dist in np.linspace(1, 5, 5):
                test_x = x + dist * np.cos(angle)
                test_y = y + dist * np.sin(angle)
                for ox_i, oy_i in zip(ox, oy):
                    if np.sqrt((test_x - ox_i)**2 + (test_y - oy_i)**2) < 1.0:
                        is_clear = False
                        break
                if not is_clear:
                    break
            if is_clear:
                clear_count += 1
        
        return clear_count
    
    def estimate_required_curvature(self, x, y, yaw, gx, gy):
        """Estimate curvature required to reach goal"""
        goal_angle = np.arctan2(gy - y, gx - x)
        angle_diff = np.arctan2(np.sin(goal_angle - yaw), 
                               np.cos(goal_angle - yaw))
        dist = np.sqrt((x - gx)**2 + (y - gy)**2)
        if dist > 0:
            return abs(angle_diff) / dist
        return 0
    
    def calculate_obstacle_asymmetry(self, x, y, ox, oy):
        """Calculate asymmetry of obstacle distribution"""
        if not ox or not oy:
            return 0
            
        quadrants = [0, 0, 0, 0]
        for ox_i, oy_i in zip(ox, oy):
            if np.sqrt((x - ox_i)**2 + (y - oy_i)**2) < 10.0:
                dx = ox_i - x
                dy = oy_i - y
                if dx >= 0 and dy >= 0:
                    quadrants[0] += 1
                elif dx < 0 and dy >= 0:
                    quadrants[1] += 1
                elif dx < 0 and dy < 0:
                    quadrants[2] += 1
                else:
                    quadrants[3] += 1
        
        if sum(quadrants) > 0:
            variance = np.var(quadrants)
            mean = np.mean(quadrants)
            return min(variance / (mean + 1), 1.0)
        return 0
    
    def estimate_collision_risk(self, x, y, yaw, ox, oy):
        """Estimate collision risk along predicted trajectory"""
        if not ox or not oy:
            return 0
            
        risk = 0
        for dist in np.linspace(0.5, 5.0, 10):
            pred_x = x + dist * np.cos(yaw)
            pred_y = y + dist * np.sin(yaw)
            
            for ox_i, oy_i in zip(ox, oy):
                clearance = np.sqrt((pred_x - ox_i)**2 + (pred_y - oy_i)**2)
                if clearance < self.config.W/2:  # Vehicle width
                    risk = 1.0
                    break
                elif clearance < self.config.W:
                    risk = max(risk, 1.0 - clearance/self.config.W)
        
        return risk
    
    def predict_uncertainty(self, x, y, yaw, ox, oy, gx, gy):
        """Predict uncertainty at given state"""
        if self.method == 'learnable_cp':
            features = self.extract_features(x, y, yaw, ox, oy, gx, gy)
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            
            with torch.no_grad():
                uncertainty = self.uncertainty_model(features_tensor).item()
            
            return uncertainty
        
        elif self.method == 'ensemble':
            # Simple uncertainty based on obstacle density
            features = self.extract_features(x, y, yaw, ox, oy, gx, gy)
            return 0.3 + 0.3 * features[1]  # Based on obstacle density
        
        else:  # naive
            return 0.0  # No uncertainty
    
    def adapt_obstacles(self, ox, oy, path, gx, gy):
        """Adapt obstacles based on predicted uncertainty along path"""
        if self.method == 'naive':
            return ox, oy  # No adaptation
        
        adapted_ox = []
        adapted_oy = []
        
        # Calculate uncertainty along path
        if path:
            uncertainties = []
            for i in range(0, len(path.x), max(1, len(path.x)//20)):  # Sample points
                x, y, yaw = path.x[i], path.y[i], path.yaw[i]
                u = self.predict_uncertainty(x, y, yaw, ox, oy, gx, gy)
                uncertainties.append(u)
            
            max_uncertainty = max(uncertainties) if uncertainties else 0.5
        else:
            max_uncertainty = 0.5
        
        # Inflate obstacles based on uncertainty
        if self.method == 'ensemble':
            # Uniform inflation for ensemble
            inflation = 0.5 + max_uncertainty * 1.5
            
            for ox_i, oy_i in zip(ox, oy):
                for angle in np.linspace(0, 2*np.pi, 8):
                    adapted_ox.append(ox_i + inflation * np.cos(angle))
                    adapted_oy.append(oy_i + inflation * np.sin(angle))
        
        elif self.method == 'learnable_cp':
            # Adaptive inflation based on local uncertainty
            for ox_i, oy_i in zip(ox, oy):
                # Find local uncertainty near this obstacle
                local_u = self.predict_uncertainty(ox_i, oy_i, 0, ox, oy, gx, gy)
                inflation = 0.3 + local_u * 2.0
                
                for angle in np.linspace(0, 2*np.pi, 12):
                    adapted_ox.append(ox_i + inflation * np.cos(angle))
                    adapted_oy.append(oy_i + inflation * np.sin(angle))
        
        # Add original obstacles too
        adapted_ox.extend(ox)
        adapted_oy.extend(oy)
        
        return adapted_ox, adapted_oy
    
    def plan_with_uncertainty(self, sx, sy, syaw, gx, gy, gyaw, ox, oy):
        """
        Plan path with uncertainty awareness using existing Hybrid A*
        
        Returns:
            path: Path object from existing planner
            uncertainty_map: Uncertainty values along path
        """
        
        # Step 1: Get initial path using existing planner
        try:
            initial_path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, ox, oy, 
                                                self.config.XY_RESO, self.config.YAW_RESO)
        except:
            initial_path = None
        
        if self.method == 'naive':
            # Naive method: just return the initial path
            uncertainty_map = [0.0] * len(initial_path.x) if initial_path else []
            return initial_path, uncertainty_map
        
        # Step 2: Adapt obstacles based on uncertainty
        adapted_ox, adapted_oy = self.adapt_obstacles(ox, oy, initial_path, gx, gy)
        
        # Step 3: Replan with adapted obstacles
        try:
            safe_path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw, 
                                            adapted_ox, adapted_oy, 
                                            self.config.XY_RESO, self.config.YAW_RESO)
        except:
            safe_path = initial_path  # Fallback to initial if replanning fails
        
        # Step 4: Calculate uncertainty along final path
        uncertainty_map = []
        if safe_path:
            for x, y, yaw in zip(safe_path.x, safe_path.y, safe_path.yaw):
                u = self.predict_uncertainty(x, y, yaw, ox, oy, gx, gy)
                uncertainty_map.append(u)
        
        return safe_path, uncertainty_map
    
    def ensemble_plan(self, sx, sy, syaw, gx, gy, gyaw, ox, oy, n_models=5):
        """Ensemble planning with multiple noise realizations"""
        paths = []
        
        for _ in range(n_models):
            # Add noise to obstacles
            noise_level = 0.3
            noisy_ox = [x + np.random.normal(0, noise_level) for x in ox]
            noisy_oy = [y + np.random.normal(0, noise_level) for y in oy]
            
            try:
                path = hybrid_astar_planning(sx, sy, syaw, gx, gy, gyaw,
                                           noisy_ox, noisy_oy,
                                           self.config.XY_RESO, self.config.YAW_RESO)
                if path:
                    paths.append(path)
            except:
                continue
        
        if not paths:
            return None, []
        
        # Use median path
        median_idx = len(paths) // 2
        return paths[median_idx], [0.3] * len(paths[median_idx].x)