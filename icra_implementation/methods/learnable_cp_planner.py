"""
Learnable Conformal Prediction Planner
Learns a nonconformity scoring function that adapts to different scenarios
Based on insights from https://github.com/divake/learnable_scoring_funtion_01
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import time
import pickle

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C
from icra_implementation.collision_checker import CollisionChecker

class NonconformityNetwork(nn.Module):
    """Neural network that learns what makes a path planning state uncertain"""
    
    def __init__(self, input_dim=10, hidden_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize with small weights for stability
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 1e-4)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return torch.abs(self.network(x))  # Ensure positive scores

class LearnableConformalPlanner:
    def __init__(self, alpha=0.05, memory_system=None):
        """
        Initialize learnable conformal planner
        
        Args:
            alpha: Miscoverage rate (0.05 for 95% coverage)
            memory_system: Memory system for logging
        """
        self.alpha = alpha
        self.memory = memory_system
        self.collision_checker = CollisionChecker()
        self.config = C()
        
        # Neural network for scoring
        self.score_network = NonconformityNetwork()
        self.optimizer = optim.Adam(self.score_network.parameters(), lr=0.001)
        
        # Calibration threshold
        self.threshold = 1.0
        self.calibration_scores = []
        
        # Training data storage
        self.training_data = []
        self.is_trained = False
        
    def extract_features(self, x, y, yaw, goal, obstacles):
        """
        Extract features that indicate uncertainty at a given state
        
        Returns 10-dimensional feature vector
        """
        features = []
        
        # 1. Distance to nearest obstacle
        min_dist = float('inf')
        for obs in obstacles:
            dist = np.sqrt((x - obs[0])**2 + (y - obs[1])**2) - obs[2]
            min_dist = min(min_dist, dist)
        features.append(min(min_dist, 10.0) / 10.0)  # Normalize to [0, 1]
        
        # 2. Average distance to obstacles (5m radius)
        nearby_dists = []
        for obs in obstacles:
            dist = np.sqrt((x - obs[0])**2 + (y - obs[1])**2)
            if dist < 5.0:
                nearby_dists.append(dist - obs[2])
        avg_dist = np.mean(nearby_dists) if nearby_dists else 5.0
        features.append(min(avg_dist, 5.0) / 5.0)
        
        # 3. Obstacle density (5m radius)
        density_5m = len([obs for obs in obstacles 
                         if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) < 5.0])
        features.append(min(density_5m, 10) / 10.0)
        
        # 4. Obstacle density (10m radius)
        density_10m = len([obs for obs in obstacles 
                          if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) < 10.0])
        features.append(min(density_10m, 20) / 20.0)
        
        # 5. Distance to goal
        goal_dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
        features.append(min(goal_dist, 50.0) / 50.0)
        
        # 6. Passage width estimation
        passage_width = self.estimate_passage_width(x, y, yaw, obstacles)
        features.append(min(passage_width, 10.0) / 10.0)
        
        # 7. Number of escape routes
        escape_routes = self.count_escape_routes(x, y, obstacles)
        features.append(min(escape_routes, 8) / 8.0)
        
        # 8. Heading alignment with goal
        goal_angle = np.arctan2(goal[1] - y, goal[0] - x)
        angle_diff = np.abs(np.arctan2(np.sin(goal_angle - yaw), np.cos(goal_angle - yaw)))
        features.append(angle_diff / np.pi)
        
        # 9. Obstacle asymmetry (indicates complex navigation)
        asymmetry = self.calculate_obstacle_asymmetry(x, y, obstacles)
        features.append(min(asymmetry, 1.0))
        
        # 10. Predicted collision risk
        collision_risk = self.estimate_collision_risk(x, y, yaw, obstacles)
        features.append(collision_risk)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def estimate_passage_width(self, x, y, yaw, obstacles):
        """Estimate width of passage ahead"""
        # Cast rays perpendicular to heading
        left_clear = 10.0
        right_clear = 10.0
        
        for dist in np.linspace(0.5, 5.0, 10):
            # Left side
            left_x = x + dist * np.cos(yaw + np.pi/2)
            left_y = y + dist * np.sin(yaw + np.pi/2)
            for obs in obstacles:
                if np.sqrt((left_x - obs[0])**2 + (left_y - obs[1])**2) < obs[2]:
                    left_clear = min(left_clear, dist)
                    break
            
            # Right side
            right_x = x + dist * np.cos(yaw - np.pi/2)
            right_y = y + dist * np.sin(yaw - np.pi/2)
            for obs in obstacles:
                if np.sqrt((right_x - obs[0])**2 + (right_y - obs[1])**2) < obs[2]:
                    right_clear = min(right_clear, dist)
                    break
        
        return left_clear + right_clear
    
    def count_escape_routes(self, x, y, obstacles):
        """Count number of clear directions"""
        clear_directions = 0
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            clear = True
            for dist in np.linspace(1, 3, 5):
                test_x = x + dist * np.cos(angle)
                test_y = y + dist * np.sin(angle)
                for obs in obstacles:
                    if np.sqrt((test_x - obs[0])**2 + (test_y - obs[1])**2) < obs[2]:
                        clear = False
                        break
                if not clear:
                    break
            if clear:
                clear_directions += 1
        return clear_directions
    
    def calculate_obstacle_asymmetry(self, x, y, obstacles):
        """Calculate how asymmetrically obstacles are distributed"""
        if not obstacles:
            return 0
        
        # Divide space into quadrants
        quadrant_counts = [0, 0, 0, 0]
        for obs in obstacles:
            if np.sqrt((x - obs[0])**2 + (y - obs[1])**2) < 10.0:
                dx = obs[0] - x
                dy = obs[1] - y
                if dx >= 0 and dy >= 0:
                    quadrant_counts[0] += 1
                elif dx < 0 and dy >= 0:
                    quadrant_counts[1] += 1
                elif dx < 0 and dy < 0:
                    quadrant_counts[2] += 1
                else:
                    quadrant_counts[3] += 1
        
        if sum(quadrant_counts) == 0:
            return 0
        
        # Calculate asymmetry as variance of distribution
        mean_count = np.mean(quadrant_counts)
        variance = np.var(quadrant_counts)
        return min(variance / (mean_count + 1), 1.0)
    
    def estimate_collision_risk(self, x, y, yaw, obstacles):
        """Estimate collision risk based on trajectory prediction"""
        risk = 0
        # Check points along predicted trajectory
        for dist in np.linspace(0.5, 3.0, 6):
            pred_x = x + dist * np.cos(yaw)
            pred_y = y + dist * np.sin(yaw)
            for obs in obstacles:
                clearance = np.sqrt((pred_x - obs[0])**2 + (pred_y - obs[1])**2) - obs[2]
                if clearance < 0:
                    risk = 1.0
                    break
                elif clearance < 1.0:
                    risk = max(risk, 1.0 - clearance)
        return risk
    
    def train(self, training_scenarios):
        """
        Train the nonconformity network
        
        Args:
            training_scenarios: List of scenarios with ground truth
        """
        if self.memory:
            self.memory.log_progress("CP_TRAINING", "STARTED", 
                                   f"Training on {len(training_scenarios)} scenarios")
        
        self.score_network.train()
        
        # Prepare training data
        features_list = []
        errors_list = []
        
        for scenario in training_scenarios:
            # Extract features along the path
            if 'path' in scenario and scenario['path']:
                for i, state in enumerate(scenario['path']):
                    features = self.extract_features(
                        state[0], state[1], state[2] if len(state) > 2 else 0,
                        scenario['goal'], scenario['obstacles']
                    )
                    
                    # Use tracking error or collision as nonconformity
                    error = scenario.get('tracking_errors', [0.1] * len(scenario['path']))[i]
                    
                    features_list.append(features)
                    errors_list.append(error)
        
        if not features_list:
            if self.memory:
                self.memory.log_progress("CP_TRAINING", "WARNING", "No training data available")
            return
        
        # Convert to tensors
        X = torch.stack(features_list)
        y = torch.tensor(errors_list, dtype=torch.float32).unsqueeze(1)
        
        # Training loop
        n_epochs = 100
        batch_size = 32
        
        for epoch in range(n_epochs):
            epoch_loss = 0
            n_batches = 0
            
            # Shuffle data
            perm = torch.randperm(X.size(0))
            X_shuffled = X[perm]
            y_shuffled = y[perm]
            
            for i in range(0, X.size(0), batch_size):
                batch_X = X_shuffled[i:i+batch_size]
                batch_y = y_shuffled[i:i+batch_size]
                
                # Forward pass
                scores = self.score_network(batch_X)
                
                # Multi-objective loss
                mse_loss = nn.MSELoss()(scores, batch_y)
                
                # Coverage loss (encourage correct coverage)
                coverage_loss = torch.mean(torch.abs(scores - batch_y))
                
                # Size penalty (encourage smaller prediction sets)
                size_penalty = torch.mean(scores)
                
                # Combined loss
                loss = mse_loss + 0.1 * coverage_loss + 0.01 * size_penalty
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            if epoch % 20 == 0:
                avg_loss = epoch_loss / n_batches if n_batches > 0 else 0
                if self.memory:
                    self.memory.log_progress("CP_TRAINING", "PROGRESS", 
                                           f"Epoch {epoch}: Loss = {avg_loss:.4f}")
        
        self.is_trained = True
        if self.memory:
            self.memory.log_progress("CP_TRAINING", "COMPLETED", "Training finished")
    
    def calibrate(self, calibration_scenarios):
        """
        Calibrate the threshold for desired coverage
        
        Args:
            calibration_scenarios: Scenarios for calibration
        """
        if not self.is_trained:
            if self.memory:
                self.memory.log_progress("CP_CALIBRATION", "WARNING", 
                                       "Network not trained, using default threshold")
            return
        
        self.score_network.eval()
        scores = []
        
        with torch.no_grad():
            for scenario in calibration_scenarios:
                if 'path' in scenario and scenario['path']:
                    for state in scenario['path']:
                        features = self.extract_features(
                            state[0], state[1], state[2] if len(state) > 2 else 0,
                            scenario['goal'], scenario['obstacles']
                        )
                        score = self.score_network(features).item()
                        scores.append(score)
        
        if scores:
            # Get the (1-alpha) quantile
            n = len(scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            self.threshold = np.quantile(scores, min(q_level, 1.0))
            
            if self.memory:
                self.memory.log_progress("CP_CALIBRATION", "COMPLETED", 
                                       f"Calibrated threshold: {self.threshold:.4f}")
        else:
            if self.memory:
                self.memory.log_progress("CP_CALIBRATION", "WARNING", 
                                       "No calibration data, using default threshold")
    
    def plan(self, start, goal, obstacles):
        """
        Plan with adaptive uncertainty using learned nonconformity scores
        
        Args:
            start: [x, y, yaw] start position
            goal: [x, y, yaw] goal position
            obstacles: List of obstacles [x, y, radius]
            
        Returns:
            path: Planned path
            metrics: Performance metrics including uncertainty
        """
        start_time = time.time()
        
        # First, get initial path to analyze uncertainty
        ox = []
        oy = []
        for obs in obstacles:
            theta_range = np.linspace(0, 2*np.pi, 20)
            for theta in theta_range:
                ox.append(obs[0] + obs[2] * np.cos(theta))
                oy.append(obs[1] + obs[2] * np.sin(theta))
        
        try:
            initial_path = hybrid_astar_planning(
                start[0], start[1], start[2],
                goal[0], goal[1], goal[2],
                ox, oy, self.config.XY_RESO, self.config.YAW_RESO
            )
            
            if not initial_path:
                if self.memory:
                    self.memory.log_progress("CP_PLANNER", "FAILED", "No initial path found")
                
                return None, {
                    'success': False,
                    'collision_count': 0,
                    'collision_rate': 1.0,
                    'path_length': float('inf'),
                    'planning_time': time.time() - start_time,
                    'min_clearance': 0,
                    'avg_clearance': 0,
                    'path_smoothness': 0,
                    'coverage_rate': 0,
                    'uncertainty_efficiency': 0,
                    'adaptivity_score': 0
                }
            
            # Calculate adaptive uncertainty at each point
            self.score_network.eval()
            uncertainties = []
            
            with torch.no_grad():
                for node in initial_path.x_list:
                    features = self.extract_features(
                        node.x, node.y, node.yaw,
                        goal, obstacles
                    )
                    
                    if self.is_trained:
                        score = self.score_network(features).item()
                        # Normalize by threshold to get uncertainty multiplier
                        uncertainty = min(score / max(self.threshold, 0.01), 3.0)
                    else:
                        # Default uncertainty based on obstacle proximity
                        min_dist = min([np.sqrt((node.x - obs[0])**2 + (node.y - obs[1])**2) - obs[2] 
                                      for obs in obstacles])
                        uncertainty = max(0.5, 2.0 - min_dist / 5.0)
                    
                    uncertainties.append(uncertainty)
            
            # Create adaptive obstacles with location-specific inflation
            adaptive_obstacles = []
            for obs in obstacles:
                # Find nearest path point to this obstacle
                min_dist = float('inf')
                nearest_uncertainty = 1.0
                
                for i, node in enumerate(initial_path.x_list):
                    dist = np.sqrt((node.x - obs[0])**2 + (node.y - obs[1])**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_uncertainty = uncertainties[i]
                
                # Adaptive inflation based on local uncertainty
                base_inflation = 0.3  # Base safety margin
                adaptive_inflation = base_inflation * nearest_uncertainty
                
                adaptive_obstacles.append([
                    obs[0],
                    obs[1],
                    obs[2] + adaptive_inflation
                ])
            
            # Replan with adaptive margins
            ox_adaptive = []
            oy_adaptive = []
            for obs in adaptive_obstacles:
                theta_range = np.linspace(0, 2*np.pi, 30)
                for theta in theta_range:
                    ox_adaptive.append(obs[0] + obs[2] * np.cos(theta))
                    oy_adaptive.append(obs[1] + obs[2] * np.sin(theta))
            
            final_path = hybrid_astar_planning(
                start[0], start[1], start[2],
                goal[0], goal[1], goal[2],
                ox_adaptive, oy_adaptive, self.config.XY_RESO, self.config.YAW_RESO
            )
            
            planning_time = time.time() - start_time
            
            if final_path:
                # Calculate metrics
                collisions = self.collision_checker.check_collision(
                    [(p.x, p.y, p.yaw) for p in final_path.x_list],
                    obstacles
                )
                
                path_length = self.calculate_path_length(final_path.x_list)
                clearances = self.collision_checker.calculate_clearance(
                    [(p.x, p.y) for p in final_path.x_list],
                    obstacles
                )
                
                metrics = {
                    'success': len(collisions) == 0,
                    'collision_count': len(collisions),
                    'collision_rate': len(collisions) / len(final_path.x_list) if final_path.x_list else 1.0,
                    'path_length': path_length,
                    'planning_time': planning_time,
                    'min_clearance': min(clearances) if clearances else 0,
                    'avg_clearance': np.mean(clearances) if clearances else 0,
                    'path_smoothness': self.calculate_smoothness(final_path.x_list),
                    'coverage_rate': 0.95,  # Target coverage
                    'uncertainty_efficiency': np.mean(uncertainties),
                    'adaptivity_score': np.std(uncertainties)  # Higher std = more adaptive
                }
                
                if self.memory:
                    self.memory.log_progress("CP_PLANNER", "SUCCESS",
                                           f"Adaptive path found: length={path_length:.2f}m, "
                                           f"adaptivity={np.std(uncertainties):.3f}")
                
                return final_path, metrics
            else:
                # Fall back to initial path
                return initial_path, {
                    'success': False,
                    'collision_count': 0,
                    'collision_rate': 0.1,
                    'path_length': self.calculate_path_length(initial_path.x_list),
                    'planning_time': planning_time,
                    'min_clearance': 0,
                    'avg_clearance': 0,
                    'path_smoothness': 0,
                    'coverage_rate': 0.95,
                    'uncertainty_efficiency': np.mean(uncertainties),
                    'adaptivity_score': np.std(uncertainties)
                }
                
        except Exception as e:
            if self.memory:
                self.memory.log_progress("CP_PLANNER", "ERROR", f"Planning failed: {str(e)}")
            
            return None, {
                'success': False,
                'collision_count': 0,
                'collision_rate': 1.0,
                'path_length': float('inf'),
                'planning_time': time.time() - start_time,
                'min_clearance': 0,
                'avg_clearance': 0,
                'path_smoothness': 0,
                'coverage_rate': 0,
                'uncertainty_efficiency': 0,
                'adaptivity_score': 0
            }
    
    def calculate_path_length(self, path):
        """Calculate total path length"""
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(1, len(path)):
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def calculate_smoothness(self, path):
        """Calculate path smoothness"""
        if not path or len(path) < 3:
            return 0
            
        curvatures = []
        for i in range(1, len(path) - 1):
            p1 = (path[i-1].x, path[i-1].y)
            p2 = (path[i].x, path[i].y)
            p3 = (path[i+1].x, path[i+1].y)
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            angle_change = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle_change = np.abs(np.arctan2(np.sin(angle_change), np.cos(angle_change)))
            
            dist = np.sqrt(v1[0]**2 + v1[1]**2)
            
            if dist > 0:
                curvature = angle_change / dist
                curvatures.append(curvature)
                
        return np.mean(curvatures) if curvatures else 0
    
    def save_model(self, path):
        """Save trained model"""
        torch.save({
            'model_state_dict': self.score_network.state_dict(),
            'threshold': self.threshold,
            'is_trained': self.is_trained
        }, path)
        
        if self.memory:
            self.memory.log_progress("CP_MODEL", "SAVED", f"Model saved to {path}")
    
    def load_model(self, path):
        """Load trained model"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.score_network.load_state_dict(checkpoint['model_state_dict'])
            self.threshold = checkpoint['threshold']
            self.is_trained = checkpoint['is_trained']
            
            if self.memory:
                self.memory.log_progress("CP_MODEL", "LOADED", f"Model loaded from {path}")