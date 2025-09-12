#!/usr/bin/env python3
"""
Learnable CP Scoring Function for Path Planning
Adapts the learnable scoring function approach to predict adaptive safety margins (tau)
based on local geometric and uncertainty features.

Key Innovation: ONE model for ALL environments - learns generalizable safety patterns
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging


class LearnableCPScoringFunction(nn.Module):
    """
    Learnable scoring function that predicts location-specific safety margins (tau).
    
    Core Algorithm:
    1. Extract spatial features from local environment context
    2. MLP processes features to predict adaptive tau
    3. Training: High tau for risky areas, low tau for safe areas
    4. Single model generalizes across all MRPB environments
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the learnable scoring function.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super().__init__()
        
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Feature dimensions - spatial features for path planning
        self.num_features = 15  # Comprehensive spatial features
        
        # Architecture configuration
        if 'learn_cp' not in config:
            # Default architecture for path planning
            hidden_dims = [128, 64, 32]
            dropout_rate = 0.3
            self.tau_min = 0.05  # Minimum safety margin
            self.tau_max = 0.40  # Maximum safety margin
        else:
            hidden_dims = config['learn_cp'].get('hidden_dims', [128, 64, 32])
            dropout_rate = config['learn_cp'].get('dropout', 0.3)
            self.tau_min = config['learn_cp'].get('tau_min', 0.05)
            self.tau_max = config['learn_cp'].get('tau_max', 0.40)
        
        # Build MLP architecture
        layers = []
        prev_dim = self.num_features
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),  # Batch norm for stability
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        # Output layer - single tau value
        layers.append(nn.Linear(prev_dim, 1))
        
        self.scoring_network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
        
        # L2 regularization
        self.l2_lambda = config.get('learn_cp', {}).get('l2_lambda', 0.001)
        
        logging.info(f"LearnableCP initialized with {sum(p.numel() for p in self.parameters())} parameters")
        
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def extract_spatial_features(self, state: np.ndarray, obstacles: np.ndarray, 
                                goal: np.ndarray, bounds: List[float]) -> torch.Tensor:
        """
        Extract comprehensive spatial features for a given location.
        
        Features capture:
        1. Local geometry (clearances, passage width)
        2. Topological context (doorway, corridor, open space)
        3. Path criticality (distance to goal, alternatives)
        4. Uncertainty indicators
        
        Args:
            state: Current position [x, y]
            obstacles: Array of obstacles [N, 4] with [x1, y1, x2, y2] per obstacle
            goal: Goal position [x, y]
            bounds: Environment bounds [xmin, xmax, ymin, ymax]
            
        Returns:
            Feature vector of shape [num_features]
        """
        features = []
        
        # 1. Clearance features (3 features)
        min_clearance = self._compute_min_clearance(state, obstacles)
        avg_clearance_1m = self._compute_avg_clearance(state, obstacles, radius=1.0)
        avg_clearance_2m = self._compute_avg_clearance(state, obstacles, radius=2.0)
        features.extend([min_clearance, avg_clearance_1m, avg_clearance_2m])
        
        # 2. Passage width and density (2 features)
        passage_width = self._compute_passage_width(state, obstacles)
        obstacle_density = self._compute_obstacle_density(state, obstacles, radius=3.0)
        features.extend([passage_width, obstacle_density])
        
        # 3. Topological indicators (3 features)
        is_near_corner = self._detect_corner(state, obstacles)
        is_in_corridor = self._detect_corridor(state, obstacles, bounds)
        is_near_doorway = self._detect_doorway(state, obstacles)
        features.extend([float(is_near_corner), float(is_in_corridor), float(is_near_doorway)])
        
        # 4. Goal-relative features (3 features)
        dist_to_goal = np.linalg.norm(state - goal)
        angle_to_goal = np.arctan2(goal[1] - state[1], goal[0] - state[0])
        goal_visibility = self._check_goal_visibility(state, goal, obstacles)
        features.extend([dist_to_goal / 10.0, angle_to_goal / np.pi, float(goal_visibility)])
        
        # 5. Boundary proximity (2 features)
        dist_to_boundary = min(
            state[0] - bounds[0], bounds[1] - state[0],
            state[1] - bounds[2], bounds[3] - state[1]
        )
        near_boundary = float(dist_to_boundary < 1.0)
        features.extend([dist_to_boundary / 5.0, near_boundary])
        
        # 6. Risk indicators (2 features)
        num_nearby_obstacles = np.sum(self._compute_distances_to_obstacles(state, obstacles) < 2.0)
        in_tight_space = float(passage_width < 1.0 and obstacle_density > 0.3)
        features.extend([num_nearby_obstacles / 5.0, in_tight_space])
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _compute_min_clearance(self, state: np.ndarray, obstacles: np.ndarray) -> float:
        """Compute minimum distance to any obstacle"""
        if len(obstacles) == 0:
            return 10.0  # Large value if no obstacles
        
        distances = self._compute_distances_to_obstacles(state, obstacles)
        return min(10.0, distances.min())
    
    def _compute_avg_clearance(self, state: np.ndarray, obstacles: np.ndarray, radius: float) -> float:
        """Compute average clearance within a radius"""
        if len(obstacles) == 0:
            return radius
        
        # Sample points in a circle around state
        num_samples = 16
        angles = np.linspace(0, 2*np.pi, num_samples, endpoint=False)
        sample_points = state + radius * np.column_stack([np.cos(angles), np.sin(angles)])
        
        total_clearance = 0
        for point in sample_points:
            distances = self._compute_distances_to_obstacles(point, obstacles)
            total_clearance += min(radius, distances.min() if len(distances) > 0 else radius)
        
        return total_clearance / num_samples
    
    def _compute_passage_width(self, state: np.ndarray, obstacles: np.ndarray) -> float:
        """Estimate width of passage at current location"""
        if len(obstacles) == 0:
            return 10.0
        
        # Cast rays in perpendicular directions
        distances = []
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            direction = np.array([np.cos(angle), np.sin(angle)])
            ray_dist = self._cast_ray(state, direction, obstacles, max_dist=5.0)
            distances.append(ray_dist)
        
        # Passage width is minimum of perpendicular distances
        width1 = distances[0] + distances[2]  # Horizontal
        width2 = distances[1] + distances[3]  # Vertical
        return min(width1, width2)
    
    def _compute_obstacle_density(self, state: np.ndarray, obstacles: np.ndarray, radius: float) -> float:
        """Compute obstacle density within radius"""
        if len(obstacles) == 0:
            return 0.0
        
        # Count obstacle area within radius
        total_area = np.pi * radius * radius
        obstacle_area = 0.0
        
        for obs in obstacles:
            # Check if obstacle intersects with radius
            obs_center = np.array([(obs[0] + obs[2])/2, (obs[1] + obs[3])/2])
            if np.linalg.norm(state - obs_center) < radius:
                # Approximate obstacle area contribution
                width = obs[2] - obs[0]
                height = obs[3] - obs[1]
                obstacle_area += width * height
        
        return min(1.0, obstacle_area / total_area)
    
    def _detect_corner(self, state: np.ndarray, obstacles: np.ndarray) -> bool:
        """Detect if near a corner (two perpendicular obstacles close)"""
        if len(obstacles) < 2:
            return False
        
        close_obstacles = []
        for obs in obstacles:
            dist = self._point_to_rectangle_distance(state, obs)
            if dist < 1.5:
                close_obstacles.append(obs)
        
        # Check if we have perpendicular obstacles
        if len(close_obstacles) >= 2:
            return True
        return False
    
    def _detect_corridor(self, state: np.ndarray, obstacles: np.ndarray, bounds: List[float]) -> bool:
        """Detect if in a corridor (narrow passage)"""
        passage_width = self._compute_passage_width(state, obstacles)
        return passage_width < 2.0 and passage_width > 0.5
    
    def _detect_doorway(self, state: np.ndarray, obstacles: np.ndarray) -> bool:
        """Detect if near a doorway (sudden width change)"""
        # Check passage width at current location and nearby
        current_width = self._compute_passage_width(state, obstacles)
        
        # Check width 1m ahead in different directions
        for angle in [0, np.pi/2, np.pi, 3*np.pi/2]:
            direction = np.array([np.cos(angle), np.sin(angle)])
            nearby_state = state + direction * 1.0
            nearby_width = self._compute_passage_width(nearby_state, obstacles)
            
            # Doorway detected if significant width change
            if abs(nearby_width - current_width) > 1.5:
                return True
        
        return False
    
    def _check_goal_visibility(self, state: np.ndarray, goal: np.ndarray, obstacles: np.ndarray) -> bool:
        """Check if goal is visible from current state"""
        # Simple line-of-sight check
        for obs in obstacles:
            if self._line_intersects_rectangle(state, goal, obs):
                return False
        return True
    
    def _compute_distances_to_obstacles(self, state: np.ndarray, obstacles: np.ndarray) -> np.ndarray:
        """Compute distances from state to all obstacles"""
        distances = []
        for obs in obstacles:
            dist = self._point_to_rectangle_distance(state, obs)
            distances.append(dist)
        return np.array(distances) if distances else np.array([10.0])
    
    def _point_to_rectangle_distance(self, point: np.ndarray, rect: np.ndarray) -> float:
        """Compute distance from point to rectangle"""
        x, y = point
        x1, y1, x2, y2 = rect
        
        # Check if point is inside rectangle
        if x1 <= x <= x2 and y1 <= y <= y2:
            return 0.0
        
        # Compute distance to closest edge
        dx = max(x1 - x, 0, x - x2)
        dy = max(y1 - y, 0, y - y2)
        return np.sqrt(dx*dx + dy*dy)
    
    def _cast_ray(self, origin: np.ndarray, direction: np.ndarray, obstacles: np.ndarray, max_dist: float) -> float:
        """Cast a ray and return distance to first obstacle"""
        min_dist = max_dist
        
        for obs in obstacles:
            # Check ray-rectangle intersection
            dist = self._ray_rectangle_intersection(origin, direction, obs)
            if dist is not None and dist < min_dist:
                min_dist = dist
        
        return min_dist
    
    def _ray_rectangle_intersection(self, origin: np.ndarray, direction: np.ndarray, rect: np.ndarray) -> Optional[float]:
        """Calculate ray-rectangle intersection distance"""
        x1, y1, x2, y2 = rect
        
        # Ray parameters
        t_min = 0
        t_max = float('inf')
        
        # Check intersection with each edge
        for i in range(2):
            if abs(direction[i]) < 1e-6:
                # Ray is parallel to slab
                if origin[i] < rect[i*2] or origin[i] > rect[i*2+1]:
                    return None
            else:
                # Compute intersection distances
                t1 = (rect[i*2] - origin[i]) / direction[i]
                t2 = (rect[i*2+1] - origin[i]) / direction[i]
                
                t_min = max(t_min, min(t1, t2))
                t_max = min(t_max, max(t1, t2))
                
                if t_min > t_max or t_max < 0:
                    return None
        
        return t_min if t_min >= 0 else None
    
    def _line_intersects_rectangle(self, p1: np.ndarray, p2: np.ndarray, rect: np.ndarray) -> bool:
        """Check if line segment intersects rectangle"""
        direction = p2 - p1
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            return False
        
        direction = direction / dist
        intersection_dist = self._ray_rectangle_intersection(p1, direction, rect)
        
        return intersection_dist is not None and intersection_dist <= dist
    
    def forward(self, states: torch.Tensor, obstacles_list: List[np.ndarray], 
                goals: torch.Tensor, bounds_list: List[List[float]]) -> torch.Tensor:
        """
        Forward pass to predict adaptive tau values.
        
        Args:
            states: Batch of states [B, 2]
            obstacles_list: List of obstacle arrays for each state
            goals: Batch of goals [B, 2]
            bounds_list: List of bounds for each state
            
        Returns:
            Predicted tau values [B]
        """
        batch_size = states.shape[0]
        all_features = []
        
        # Extract features for each state
        for i in range(batch_size):
            state = states[i].cpu().numpy()
            goal = goals[i].cpu().numpy()
            obstacles = obstacles_list[i]
            bounds = bounds_list[i]
            
            features = self.extract_spatial_features(state, obstacles, goal, bounds)
            all_features.append(features)
        
        # Stack features
        all_features = torch.stack(all_features).to(self.device)
        
        # Forward pass through network
        tau_values = self.scoring_network(all_features)  # [B, 1]
        
        # Apply safety bounds and squeeze
        tau_values = torch.clamp(tau_values.squeeze(-1), min=self.tau_min, max=self.tau_max)
        
        # Add L2 regularization if training
        if self.training:
            l2_reg = sum(torch.sum(param ** 2) for param in self.parameters())
            self.l2_reg = self.l2_lambda * l2_reg
        else:
            self.l2_reg = 0.0
        
        return tau_values
    
    def predict_tau(self, state: np.ndarray, obstacles: np.ndarray, 
                   goal: np.ndarray, bounds: List[float]) -> float:
        """
        Predict tau for a single state (inference mode).
        
        Args:
            state: Current position [x, y]
            obstacles: Obstacle array
            goal: Goal position
            bounds: Environment bounds
            
        Returns:
            Predicted tau value
        """
        self.eval()
        with torch.no_grad():
            states = torch.tensor([state], dtype=torch.float32)
            goals = torch.tensor([goal], dtype=torch.float32)
            
            tau = self.forward(states, [obstacles], goals, [bounds])
            return tau.item()