#!/usr/bin/env python3
"""
Learnable Conformal Prediction for Path Planning
Adapts Algorithm 1 from classification paper to path planning context
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import sys
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar


class FeatureExtractor:
    """Extract geometric features for any point in the environment"""
    
    @staticmethod
    def extract_features(x: float, y: float, obstacles: List[Tuple], 
                        path: Optional[List] = None, goal: Optional[Tuple] = None) -> np.ndarray:
        """
        Extract 10 features for a given point
        Returns normalized feature vector
        """
        features = []
        
        # 1. Distance to nearest obstacle (normalized 0-1)
        min_dist = float('inf')
        for obs in obstacles:
            # Calculate distance from point to obstacle rectangle
            dx = max(obs[0] - x, 0, x - (obs[0] + obs[2]))
            dy = max(obs[1] - y, 0, y - (obs[1] + obs[3]))
            dist = np.sqrt(dx**2 + dy**2)
            min_dist = min(min_dist, dist)
        features.append(min(min_dist / 10.0, 1.0))  # Normalize by 10 units
        
        # 2. Obstacle density in radius 5
        density_5 = FeatureExtractor._compute_obstacle_density(x, y, obstacles, radius=5.0)
        features.append(density_5)
        
        # 3. Obstacle density in radius 10
        density_10 = FeatureExtractor._compute_obstacle_density(x, y, obstacles, radius=10.0)
        features.append(density_10)
        
        # 4. Passage width estimate (normalized)
        passage_width = FeatureExtractor._estimate_passage_width(x, y, obstacles)
        features.append(min(passage_width / 10.0, 1.0))
        
        # 5. Distance from start (normalized if path provided)
        if path is not None and len(path) > 0:
            start_dist = np.sqrt((x - path[0][0])**2 + (y - path[0][1])**2)
            max_dist = 50.0  # Assuming max environment dimension
            features.append(min(start_dist / max_dist, 1.0))
        else:
            features.append(0.5)  # Default middle value
        
        # 6. Distance to goal (normalized if goal provided)
        if goal is not None:
            goal_dist = np.sqrt((x - goal[0])**2 + (y - goal[1])**2)
            features.append(min(goal_dist / 50.0, 1.0))
        else:
            features.append(0.5)
        
        # 7. Path curvature (if path provided)
        if path is not None and len(path) > 2:
            curvature = FeatureExtractor._compute_path_curvature(x, y, path)
            features.append(curvature)
        else:
            features.append(0.0)
        
        # 8. Number of escape directions (0-8 normalized)
        escape_dirs = FeatureExtractor._count_escape_directions(x, y, obstacles)
        features.append(escape_dirs / 8.0)
        
        # 9. Clearance variance (measures irregularity)
        clearance_var = FeatureExtractor._compute_clearance_variance(x, y, obstacles)
        features.append(clearance_var)
        
        # 10. Is narrow passage indicator (binary)
        is_narrow = 1.0 if passage_width < 3.0 else 0.0
        features.append(is_narrow)
        
        return np.array(features, dtype=np.float32)
    
    @staticmethod
    def _compute_obstacle_density(x: float, y: float, obstacles: List, radius: float) -> float:
        """Compute obstacle density within radius"""
        area = np.pi * radius**2
        occupied_area = 0.0
        
        for obs in obstacles:
            # Check if obstacle intersects with circle
            # Simplified: check if any corner is within radius
            corners = [
                (obs[0], obs[1]),
                (obs[0] + obs[2], obs[1]),
                (obs[0], obs[1] + obs[3]),
                (obs[0] + obs[2], obs[1] + obs[3])
            ]
            
            for cx, cy in corners:
                if np.sqrt((cx - x)**2 + (cy - y)**2) <= radius:
                    # Approximate occupied area (simplified)
                    occupied_area += min(obs[2] * obs[3], area * 0.1)
                    break
        
        return min(occupied_area / area, 1.0)
    
    @staticmethod
    def _estimate_passage_width(x: float, y: float, obstacles: List) -> float:
        """Estimate width of passage at current point"""
        # Cast rays in 8 directions and find minimum clearance
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1), 
                     (0.707, 0.707), (-0.707, 0.707), 
                     (-0.707, -0.707), (0.707, -0.707)]
        
        min_clearance = float('inf')
        
        for dx, dy in directions:
            # Cast ray and find distance to obstacle
            for step in np.arange(0.1, 20.0, 0.1):
                rx = x + dx * step
                ry = y + dy * step
                
                # Check if ray hits obstacle
                for obs in obstacles:
                    if (obs[0] <= rx <= obs[0] + obs[2] and 
                        obs[1] <= ry <= obs[1] + obs[3]):
                        min_clearance = min(min_clearance, step)
                        break
        
        return min_clearance * 2  # Width is 2x the minimum clearance
    
    @staticmethod
    def _compute_path_curvature(x: float, y: float, path: List) -> float:
        """Compute local path curvature near point"""
        # Find nearest point on path
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, (px, py) in enumerate(path):
            dist = np.sqrt((x - px)**2 + (y - py)**2)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        # Compute curvature using 3 points
        if 0 < nearest_idx < len(path) - 1:
            p1 = np.array(path[nearest_idx - 1])
            p2 = np.array(path[nearest_idx])
            p3 = np.array(path[nearest_idx + 1])
            
            # Calculate angle between vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                return angle / np.pi  # Normalize to [0, 1]
        
        return 0.0
    
    @staticmethod
    def _count_escape_directions(x: float, y: float, obstacles: List) -> int:
        """Count number of free directions for escape"""
        directions = [(1, 0), (0, 1), (-1, 0), (0, -1),
                     (0.707, 0.707), (-0.707, 0.707),
                     (-0.707, -0.707), (0.707, -0.707)]
        
        free_dirs = 0
        check_dist = 2.0  # Check 2 units away
        
        for dx, dy in directions:
            tx = x + dx * check_dist
            ty = y + dy * check_dist
            
            # Check if target point is free
            is_free = True
            for obs in obstacles:
                if (obs[0] <= tx <= obs[0] + obs[2] and
                    obs[1] <= ty <= obs[1] + obs[3]):
                    is_free = False
                    break
            
            if is_free:
                free_dirs += 1
        
        return free_dirs
    
    @staticmethod
    def _compute_clearance_variance(x: float, y: float, obstacles: List) -> float:
        """Compute variance in clearances to measure irregularity"""
        clearances = []
        
        for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            # Find clearance in this direction
            for step in np.arange(0.1, 10.0, 0.1):
                rx = x + dx * step
                ry = y + dy * step
                
                for obs in obstacles:
                    if (obs[0] <= rx <= obs[0] + obs[2] and
                        obs[1] <= ry <= obs[1] + obs[3]):
                        clearances.append(step)
                        break
                else:
                    if step >= 9.9:
                        clearances.append(10.0)
        
        if len(clearances) > 1:
            variance = np.var(clearances)
            return min(variance / 25.0, 1.0)  # Normalize by max expected variance
        
        return 0.0


class LearnableScoringNetwork(nn.Module):
    """MLP that predicts local safety margins based on features"""
    
    def __init__(self, input_dim: int = 10, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Regularization
            prev_dim = hidden_dim
        
        # Output layer - single value for tau prediction
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Softplus())  # Ensure positive output
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """Forward pass - predict safety margin for given features"""
        return self.network(x).squeeze()


class LearnableCP:
    """
    Learnable Conformal Prediction for Path Planning
    Implements Algorithm 1 adapted for path planning
    """
    
    def __init__(self, alpha: float = 0.1, max_tau: float = 1.0, 
                 learning_rate: float = 0.001, beta: float = 0.9):
        """
        Args:
            alpha: Miscoverage rate (1 - coverage)
            max_tau: Maximum safety margin
            learning_rate: Learning rate for MLP
            beta: Momentum for tau updates (from Algorithm 1)
        """
        self.alpha = alpha
        self.max_tau = max_tau
        self.beta = beta
        
        # Initialize network
        self.scoring_net = LearnableScoringNetwork()
        self.optimizer = optim.Adam(self.scoring_net.parameters(), lr=learning_rate)
        
        # Dynamic tau (starts at max, will be updated)
        self.tau = max_tau
        
        # Adaptive loss weights
        self.w_coverage = 1.0
        self.w_size = 1.0
        
        # Training history
        self.history = {
            'tau': [],
            'coverage': [],
            'avg_margin': [],
            'path_length': []
        }
    
    def train_epoch(self, training_data: List[Dict], epsilon: float = 0.01):
        """
        Train for one epoch following Algorithm 1
        
        Args:
            training_data: List of trials with paths and obstacles
            epsilon: Tolerance for coverage adaptation
        """
        epoch_losses = []
        epoch_coverage = []
        epoch_margins = []
        
        for trial in training_data:
            # Extract data
            path = trial['path']
            true_obs = trial['true_obstacles']
            perceived_obs = trial['perceived_obstacles']
            goal = path[-1] if path else (45, 15)
            
            # Extract features for each point on path
            features_list = []
            for point in path:
                features = FeatureExtractor.extract_features(
                    point[0], point[1], perceived_obs, path, goal
                )
                features_list.append(features)
            
            if not features_list:
                continue
            
            # Convert to tensor
            features_tensor = torch.FloatTensor(np.array(features_list))
            
            # Get predicted safety margins from network
            self.scoring_net.train()
            predicted_margins = self.scoring_net(features_tensor)
            
            # Update tau with momentum (Line 10 in Algorithm 1)
            if len(predicted_margins) > 0:
                new_tau = torch.quantile(predicted_margins, 1 - self.alpha).item()
                self.tau = self.beta * self.tau + (1 - self.beta) * new_tau
                self.tau = min(self.tau, self.max_tau)
            
            # Apply adaptive margins to obstacles
            local_margins = predicted_margins.detach().numpy() * self.tau
            
            # Check coverage (collision detection)
            collision = False
            for i, point in enumerate(path):
                # Check if point collides with true obstacles
                for obs in true_obs:
                    if (obs[0] <= point[0] <= obs[0] + obs[2] and
                        obs[1] <= point[1] <= obs[1] + obs[3]):
                        collision = True
                        break
                if collision:
                    break
            
            coverage = 0.0 if collision else 1.0
            epoch_coverage.append(coverage)
            
            # Adaptive loss weights (Lines 13-17 in Algorithm 1)
            current_coverage = np.mean(epoch_coverage)
            if current_coverage < 1 - self.alpha - epsilon:
                self.w_coverage = 2.0
                self.w_size = 1.0
            else:
                self.w_coverage = 1.0
                self.w_size = 1.5
            
            # Compute losses
            # Coverage loss: penalize collisions
            coverage_loss = torch.tensor(1.0 - coverage, requires_grad=True)
            
            # Size loss: penalize large margins (inefficiency)
            size_loss = torch.mean(predicted_margins) / self.max_tau
            
            # Regularization
            reg_loss = 0.01 * torch.mean(predicted_margins**2)
            
            # Total loss
            total_loss = (self.w_coverage * coverage_loss + 
                         self.w_size * size_loss + 
                         reg_loss)
            
            # Backpropagation
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            epoch_losses.append(total_loss.item())
            epoch_margins.append(torch.mean(predicted_margins).item())
        
        # Record epoch statistics
        if epoch_losses:
            self.history['tau'].append(self.tau)
            self.history['coverage'].append(np.mean(epoch_coverage))
            self.history['avg_margin'].append(np.mean(epoch_margins))
            
            return {
                'loss': np.mean(epoch_losses),
                'coverage': np.mean(epoch_coverage),
                'tau': self.tau,
                'avg_margin': np.mean(epoch_margins)
            }
        
        return None
    
    def predict_margin(self, x: float, y: float, obstacles: List,
                      path: Optional[List] = None, goal: Optional[Tuple] = None) -> float:
        """
        Predict safety margin for a specific location
        
        Returns:
            Predicted safety margin τ(x,y)
        """
        features = FeatureExtractor.extract_features(x, y, obstacles, path, goal)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        self.scoring_net.eval()
        with torch.no_grad():
            margin = self.scoring_net(features_tensor).item()
        
        return margin * self.tau
    
    def inflate_obstacles_adaptive(self, obstacles: List, 
                                  path: Optional[List] = None,
                                  goal: Optional[Tuple] = None,
                                  resolution: float = 1.0) -> List:
        """
        Inflate obstacles with adaptive margins based on local context
        
        Returns:
            List of inflated obstacles with varying margins
        """
        inflated = []
        
        for obs in obstacles:
            # Sample points around obstacle
            sample_points = []
            for x in np.arange(obs[0] - 2, obs[0] + obs[2] + 2, resolution):
                for y in np.arange(obs[1] - 2, obs[1] + obs[3] + 2, resolution):
                    sample_points.append((x, y))
            
            # Get max margin needed around this obstacle
            max_margin = 0
            for x, y in sample_points:
                margin = self.predict_margin(x, y, obstacles, path, goal)
                max_margin = max(max_margin, margin)
            
            # Inflate obstacle by local margin
            inflated_obs = (
                max(0, obs[0] - max_margin),
                max(0, obs[1] - max_margin),
                obs[2] + 2 * max_margin,
                obs[3] + 2 * max_margin
            )
            inflated.append(inflated_obs)
        
        return inflated
    
    def plan_with_adaptive_cp(self, start: Tuple, goal: Tuple, 
                             perceived_obstacles: List,
                             max_iter: int = 1000) -> Optional[List]:
        """
        Plan path using adaptive conformal prediction
        
        Returns:
            Path if found, None otherwise
        """
        # Initial planning without path context
        initial_inflated = self.inflate_obstacles_adaptive(
            perceived_obstacles, None, goal
        )
        
        # Plan initial path
        planner = RRTStar(start, goal, initial_inflated, max_iter=max_iter//2)
        initial_path = planner.plan()
        
        if initial_path:
            # Refine with path context
            refined_inflated = self.inflate_obstacles_adaptive(
                perceived_obstacles, initial_path, goal
            )
            
            # Replan with refined margins
            planner = RRTStar(start, goal, refined_inflated, max_iter=max_iter//2)
            refined_path = planner.plan()
            
            return refined_path if refined_path else initial_path
        
        return None


def train_learnable_cp(num_epochs: int = 50, 
                       num_trials_per_epoch: int = 100) -> LearnableCP:
    """
    Train Learnable CP model
    
    Returns:
        Trained LearnableCP model
    """
    print("\n" + "="*70)
    print("TRAINING LEARNABLE CP")
    print("="*70)
    
    model = LearnableCP(alpha=0.05, max_tau=1.0, learning_rate=0.001)
    
    for epoch in range(num_epochs):
        # Generate training data for this epoch
        training_data = []
        
        for trial_idx in range(num_trials_per_epoch):
            # Random environment type
            env_types = ['passages', 'open', 'narrow']
            env_type = np.random.choice(env_types)
            env = ContinuousEnvironment(env_type=env_type)
            
            # Add noise
            noise_level = np.random.uniform(0.1, 0.3)
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=noise_level, 
                seed=epoch * 1000 + trial_idx
            )
            
            # Plan path with current model
            if epoch == 0:
                # Use standard CP for initial paths
                cp = ContinuousStandardCP(env.obstacles, "penetration")
                tau = cp.calibrate(
                    ContinuousNoiseModel.add_thinning_noise,
                    {'thin_factor': noise_level},
                    num_samples=50,
                    confidence=0.95
                )
                inflated = cp.inflate_obstacles(perceived)
            else:
                # Use learnable CP
                inflated = model.inflate_obstacles_adaptive(
                    perceived, None, (45, 15)
                )
            
            planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
            path = planner.plan()
            
            if path:
                training_data.append({
                    'path': path,
                    'true_obstacles': env.obstacles,
                    'perceived_obstacles': perceived,
                    'noise_level': noise_level
                })
        
        # Train on this epoch's data
        if training_data:
            stats = model.train_epoch(training_data)
            
            if stats and epoch % 5 == 0:
                print(f"\nEpoch {epoch+1}/{num_epochs}:")
                print(f"  Loss: {stats['loss']:.4f}")
                print(f"  Coverage: {stats['coverage']*100:.1f}%")
                print(f"  Dynamic τ: {stats['tau']:.3f}")
                print(f"  Avg margin: {stats['avg_margin']:.3f}")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    
    return model


def main():
    """Test Learnable CP implementation"""
    
    # Train model
    model = train_learnable_cp(num_epochs=30, num_trials_per_epoch=50)
    
    # Test on different environments
    print("\n" + "="*70)
    print("TESTING LEARNABLE CP")
    print("="*70)
    
    test_envs = ['passages', 'narrow', 'open']
    
    for env_type in test_envs:
        print(f"\nTesting on {env_type} environment:")
        
        env = ContinuousEnvironment(env_type=env_type)
        perceived = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles, thin_factor=0.2, seed=9999
        )
        
        # Plan with learnable CP
        path = model.plan_with_adaptive_cp((5, 15), (45, 15), perceived)
        
        if path:
            # Check collision
            collision = False
            for point in path:
                if env.point_in_obstacle(point[0], point[1]):
                    collision = True
                    break
            
            print(f"  Path found: Length = {len(path)}")
            print(f"  Collision: {'YES' if collision else 'NO'}")
            
            # Show adaptive margins at different points
            sample_points = [path[0], path[len(path)//4], 
                           path[len(path)//2], path[-1]]
            print("  Adaptive margins along path:")
            for i, point in enumerate(sample_points):
                margin = model.predict_margin(
                    point[0], point[1], perceived, path, (45, 15)
                )
                print(f"    Point {i}: τ = {margin:.3f}")
        else:
            print("  No path found")
    
    # Save model
    torch.save(model.scoring_net.state_dict(), 'learnable_cp_model.pth')
    print("\nModel saved to learnable_cp_model.pth")


if __name__ == "__main__":
    main()