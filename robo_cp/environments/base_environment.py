"""
Base environment class with uncertainty handling capabilities.
Extends the functionality to support naive, traditional CP, and learnable CP methods.
"""

import numpy as np
import copy
from typing import List, Tuple, Optional, Dict
import sys
import os

# Add python_motion_planning to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

# Note: We'll create our own environment class without depending on python_motion_planning's environment
# This makes our code more self-contained


class UncertaintyEnvironment:
    """Enhanced environment with uncertainty quantification support."""
    
    def __init__(self, width: int = 50, height: int = 50, obstacle_ratio: float = 0.2):
        """
        Initialize environment with uncertainty handling.
        
        Args:
            width: Grid width
            height: Grid height
            obstacle_ratio: Ratio of obstacles to total cells
        """
        self.width = width
        self.height = height
        self.obstacle_ratio = obstacle_ratio
        
        # Original obstacles (ground truth)
        self.original_obstacles = []
        
        # Working obstacles (may be inflated)
        self.obstacles = []
        
        # Base map from python_motion_planning
        self.base_map = None
        
        # Uncertainty parameters
        self.uncertainty_method = 'naive'
        self.fixed_margin = 0.0
        self.adaptive_margins = {}
        
    def generate_random_obstacles(self, num_obstacles: Optional[int] = None):
        """Generate random circular obstacles."""
        if num_obstacles is None:
            num_obstacles = int(self.width * self.height * self.obstacle_ratio / 10)
        
        self.original_obstacles = []
        
        for _ in range(num_obstacles):
            # Random position
            x = np.random.uniform(5, self.width - 5)
            y = np.random.uniform(5, self.height - 5)
            
            # Random radius
            radius = np.random.uniform(1.0, 3.0)
            
            self.original_obstacles.append({
                'center': (x, y),
                'radius': radius,
                'type': 'circle'
            })
        
        self.obstacles = copy.deepcopy(self.original_obstacles)
        return self.original_obstacles
    
    def add_obstacle_cluster(self, center: Tuple[float, float], num_obstacles: int = 5):
        """Add a cluster of obstacles around a center point."""
        for _ in range(num_obstacles):
            offset_x = np.random.normal(0, 3)
            offset_y = np.random.normal(0, 3)
            
            x = np.clip(center[0] + offset_x, 2, self.width - 2)
            y = np.clip(center[1] + offset_y, 2, self.height - 2)
            radius = np.random.uniform(0.5, 2.0)
            
            self.original_obstacles.append({
                'center': (x, y),
                'radius': radius,
                'type': 'circle'
            })
        
        self.obstacles = copy.deepcopy(self.original_obstacles)
    
    def create_narrow_passage(self, start_x: float, y: float, length: float, gap: float):
        """Create a narrow passage scenario."""
        # Top wall
        for x in np.linspace(start_x, start_x + length, int(length)):
            self.original_obstacles.append({
                'center': (x, y + gap/2 + 1),
                'radius': 0.5,
                'type': 'circle'
            })
        
        # Bottom wall
        for x in np.linspace(start_x, start_x + length, int(length)):
            self.original_obstacles.append({
                'center': (x, y - gap/2 - 1),
                'radius': 0.5,
                'type': 'circle'
            })
        
        self.obstacles = copy.deepcopy(self.original_obstacles)
    
    def apply_uncertainty(self, method: str = 'naive', 
                         fixed_margin: float = 2.0,
                         adaptive_model = None,
                         features: Optional[np.ndarray] = None):
        """
        Apply uncertainty quantification method to obstacles.
        
        Args:
            method: 'naive', 'traditional_cp', or 'learnable_cp'
            fixed_margin: Margin for traditional CP (in grid cells)
            adaptive_model: Trained model for learnable CP
            features: Feature vector for learnable CP
        """
        self.uncertainty_method = method
        self.obstacles = copy.deepcopy(self.original_obstacles)
        
        if method == 'naive':
            # No inflation
            self.fixed_margin = 0.0
            
        elif method == 'traditional_cp':
            # Uniform inflation
            self.fixed_margin = fixed_margin
            for obs in self.obstacles:
                obs['radius'] += fixed_margin
                
        elif method == 'learnable_cp':
            # Adaptive inflation based on features
            if adaptive_model is not None and features is not None:
                # Get adaptive margins from model
                margins = adaptive_model.predict(features)
                
                # Apply different margin to each obstacle
                for i, obs in enumerate(self.obstacles):
                    if i < len(margins):
                        obs['radius'] += margins[i]
                    else:
                        obs['radius'] += 0.5  # Default small margin
            else:
                print("Warning: No model provided for learnable CP, using default margin")
                for obs in self.obstacles:
                    obs['radius'] += 0.5
    
    def add_noise_to_obstacles(self, sigma: float = 0.5):
        """
        Add Gaussian noise to obstacle positions for Monte Carlo testing.
        
        Args:
            sigma: Standard deviation of noise
        """
        noisy_obstacles = []
        
        for obs in self.original_obstacles:
            noisy_obs = copy.deepcopy(obs)
            
            # Add noise to position
            noise_x = np.random.normal(0, sigma)
            noise_y = np.random.normal(0, sigma)
            
            noisy_obs['center'] = (
                obs['center'][0] + noise_x,
                obs['center'][1] + noise_y
            )
            
            # Small noise to radius
            noise_r = np.random.normal(0, sigma * 0.2)
            noisy_obs['radius'] = max(0.1, obs['radius'] + noise_r)
            
            noisy_obstacles.append(noisy_obs)
        
        return noisy_obstacles
    
    def check_collision(self, path: List[Tuple[float, float]], 
                       use_original: bool = True,
                       robot_radius: float = 0.5) -> bool:
        """
        Check if path collides with obstacles.
        
        Args:
            path: List of (x, y) points
            use_original: If True, check against original obstacles (ground truth)
            robot_radius: Robot radius for collision checking
        
        Returns:
            True if collision detected
        """
        obstacles_to_check = self.original_obstacles if use_original else self.obstacles
        
        for point in path:
            for obs in obstacles_to_check:
                dist = np.sqrt((point[0] - obs['center'][0])**2 + 
                             (point[1] - obs['center'][1])**2)
                
                if dist < (obs['radius'] + robot_radius):
                    return True
        
        return False
    
    def get_occupancy_grid(self, use_inflated: bool = True) -> np.ndarray:
        """
        Convert obstacles to occupancy grid for planning algorithms.
        
        Args:
            use_inflated: If True, use inflated obstacles
        
        Returns:
            2D numpy array where 1 = occupied, 0 = free
        """
        grid = np.zeros((self.height, self.width))
        obstacles_to_use = self.obstacles if use_inflated else self.original_obstacles
        
        for obs in obstacles_to_use:
            cx, cy = obs['center']
            radius = obs['radius']
            
            # Fill grid cells within obstacle radius
            for i in range(max(0, int(cx - radius - 1)), 
                          min(self.width, int(cx + radius + 1))):
                for j in range(max(0, int(cy - radius - 1)), 
                             min(self.height, int(cy + radius + 1))):
                    if np.sqrt((i - cx)**2 + (j - cy)**2) <= radius:
                        grid[j, i] = 1
        
        return grid
    
    def get_clearance(self, point: Tuple[float, float], 
                     use_original: bool = True) -> float:
        """
        Get minimum clearance from point to nearest obstacle.
        
        Args:
            point: (x, y) position
            use_original: If True, use original obstacles
        
        Returns:
            Minimum distance to nearest obstacle edge
        """
        obstacles_to_check = self.original_obstacles if use_original else self.obstacles
        min_clearance = float('inf')
        
        for obs in obstacles_to_check:
            dist_to_center = np.sqrt((point[0] - obs['center'][0])**2 + 
                                    (point[1] - obs['center'][1])**2)
            clearance = dist_to_center - obs['radius']
            min_clearance = min(min_clearance, clearance)
        
        return max(0, min_clearance)
    
    def save_environment(self, filepath: str):
        """Save environment configuration to file."""
        import pickle
        
        env_data = {
            'width': self.width,
            'height': self.height,
            'original_obstacles': self.original_obstacles,
            'uncertainty_method': self.uncertainty_method,
            'fixed_margin': self.fixed_margin
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(env_data, f)
    
    def load_environment(self, filepath: str):
        """Load environment configuration from file."""
        import pickle
        
        with open(filepath, 'rb') as f:
            env_data = pickle.load(f)
        
        self.width = env_data['width']
        self.height = env_data['height']
        self.original_obstacles = env_data['original_obstacles']
        self.obstacles = copy.deepcopy(self.original_obstacles)
        self.uncertainty_method = env_data.get('uncertainty_method', 'naive')
        self.fixed_margin = env_data.get('fixed_margin', 0.0)