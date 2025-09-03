#!/usr/bin/env python3
"""
Continuous Environment Module
Defines continuous obstacles and environment for path planning
"""

import numpy as np
from typing import List, Tuple, Set
import random


class ContinuousEnvironment:
    """
    Continuous 2D environment with obstacles
    """
    
    def __init__(self, width: float = 50.0, height: float = 30.0):
        """
        Initialize continuous environment
        
        Args:
            width: Environment width
            height: Environment height
        """
        self.width = width
        self.height = height
        self.bounds = (0, width, 0, height)
        
        # Define obstacles as continuous rectangles (x, y, width, height)
        self.obstacles = self._create_obstacles()
        
    def _create_obstacles(self) -> List[Tuple[float, float, float, float]]:
        """
        Create obstacle configuration similar to discrete environment
        """
        obstacles = []
        
        # Boundaries (walls)
        obstacles.append((0, 0, self.width, 0.5))  # Bottom
        obstacles.append((0, self.height-0.5, self.width, 0.5))  # Top
        obstacles.append((0, 0, 0.5, self.height))  # Left
        obstacles.append((self.width-0.5, 0, 0.5, self.height))  # Right
        
        # Internal obstacles (similar to discrete layout)
        # Horizontal wall with gap
        obstacles.append((10, 14.5, 10, 1))  # Horizontal wall at y=15
        
        # Vertical walls creating passages
        obstacles.append((20, 0, 1, 14.5))  # First vertical wall
        obstacles.append((30, 15.5, 1, 14.5))  # Second vertical wall
        obstacles.append((40, 0, 1, 15.5))  # Third vertical wall
        
        return obstacles
    
    def point_in_obstacle(self, x: float, y: float) -> bool:
        """
        Check if point (x, y) is inside any obstacle
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            True if point is in obstacle, False otherwise
        """
        for (ox, oy, width, height) in self.obstacles:
            if ox <= x <= ox + width and oy <= y <= oy + height:
                return True
        return False
    
    def get_passages(self) -> List[Tuple[float, float, float]]:
        """
        Get passage locations (x, y_center, width)
        """
        passages = [
            (20.0, 7.25, 14.5),   # First passage
            (30.0, 15.0, 1.0),    # Second passage (narrow)
            (40.0, 7.75, 15.5),   # Third passage
        ]
        return passages
    
    def sample_free_point(self) -> Tuple[float, float]:
        """
        Sample a random collision-free point
        """
        while True:
            x = random.uniform(1, self.width - 1)
            y = random.uniform(1, self.height - 1)
            if not self.point_in_obstacle(x, y):
                return (x, y)
    
    def get_obstacle_boundaries(self) -> List[Tuple[float, float, float, float]]:
        """
        Get obstacle boundaries for visualization
        """
        return self.obstacles.copy()


class ContinuousNoiseModel:
    """
    Noise models for continuous perception
    """
    
    @staticmethod
    def add_gaussian_noise(obstacles: List[Tuple[float, float, float, float]], 
                          noise_std: float = 0.3,
                          seed: int = None) -> List[Tuple[float, float, float, float]]:
        """
        Add Gaussian noise to obstacle boundaries
        
        Args:
            obstacles: List of (x, y, width, height) rectangles
            noise_std: Standard deviation of noise
            seed: Random seed
            
        Returns:
            Noisy obstacles
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        noisy_obstacles = []
        
        for (x, y, w, h) in obstacles:
            # Keep boundaries unchanged
            if x <= 0.5 or y <= 0.5 or x >= 49.5 or y >= 29.5:
                noisy_obstacles.append((x, y, w, h))
            else:
                # Add noise to position and size
                noise_x = np.random.normal(0, noise_std)
                noise_y = np.random.normal(0, noise_std)
                noise_w = np.random.normal(0, noise_std)
                noise_h = np.random.normal(0, noise_std)
                
                new_x = max(0.5, x + noise_x)
                new_y = max(0.5, y + noise_y)
                new_w = max(0.1, w + noise_w)
                new_h = max(0.1, h + noise_h)
                
                noisy_obstacles.append((new_x, new_y, new_w, new_h))
        
        return noisy_obstacles
    
    @staticmethod
    def add_thinning_noise(obstacles: List[Tuple[float, float, float, float]],
                          thin_factor: float = 0.2,
                          seed: int = None) -> List[Tuple[float, float, float, float]]:
        """
        Make obstacles appear thinner (shrink them)
        
        Args:
            obstacles: List of obstacles
            thin_factor: How much to shrink (0.2 = 20% reduction)
            seed: Random seed
            
        Returns:
            Thinned obstacles
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        thinned = []
        
        for (x, y, w, h) in obstacles:
            # Keep boundaries unchanged
            if x <= 0.5 or y <= 0.5 or x >= 49.5 or y >= 29.5:
                thinned.append((x, y, w, h))
            else:
                # Shrink obstacle
                shrink = random.uniform(0, thin_factor)
                
                new_x = x + shrink * w * 0.25
                new_y = y + shrink * h * 0.25
                new_w = w * (1 - shrink * 0.5)
                new_h = h * (1 - shrink * 0.5)
                
                # Ensure minimum size
                new_w = max(0.1, new_w)
                new_h = max(0.1, new_h)
                
                thinned.append((new_x, new_y, new_w, new_h))
        
        return thinned
    
    @staticmethod
    def add_expansion_noise(obstacles: List[Tuple[float, float, float, float]],
                           expand_factor: float = 0.2,
                           seed: int = None) -> List[Tuple[float, float, float, float]]:
        """
        Make obstacles appear thicker (expand them)
        
        Args:
            obstacles: List of obstacles
            expand_factor: How much to expand (0.2 = 20% expansion)
            seed: Random seed
            
        Returns:
            Expanded obstacles
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        expanded = []
        
        for (x, y, w, h) in obstacles:
            # Keep boundaries unchanged
            if x <= 0.5 or y <= 0.5 or x >= 49.5 or y >= 29.5:
                expanded.append((x, y, w, h))
            else:
                # Expand obstacle
                expand = random.uniform(0, expand_factor)
                
                new_x = max(0.5, x - expand * w * 0.25)
                new_y = max(0.5, y - expand * h * 0.25)
                new_w = min(49 - new_x, w * (1 + expand * 0.5))
                new_h = min(29 - new_y, h * (1 + expand * 0.5))
                
                expanded.append((new_x, new_y, new_w, new_h))
        
        return expanded
    
    @staticmethod
    def add_mixed_noise(obstacles: List[Tuple[float, float, float, float]],
                       gaussian_std: float = 0.1,
                       thin_prob: float = 0.3,
                       expand_prob: float = 0.3,
                       seed: int = None) -> List[Tuple[float, float, float, float]]:
        """
        Mixed noise model combining different types
        
        Args:
            obstacles: List of obstacles
            gaussian_std: Std dev for position noise
            thin_prob: Probability of thinning
            expand_prob: Probability of expansion
            seed: Random seed
            
        Returns:
            Noisy obstacles
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        noisy = []
        
        for (x, y, w, h) in obstacles:
            # Keep boundaries unchanged
            if x <= 0.5 or y <= 0.5 or x >= 49.5 or y >= 29.5:
                noisy.append((x, y, w, h))
            else:
                # Randomly choose noise type
                noise_type = random.random()
                
                if noise_type < thin_prob:
                    # Thin the obstacle
                    shrink = random.uniform(0.1, 0.3)
                    new_x = x + shrink * w * 0.25
                    new_y = y + shrink * h * 0.25
                    new_w = max(0.1, w * (1 - shrink * 0.5))
                    new_h = max(0.1, h * (1 - shrink * 0.5))
                    
                elif noise_type < thin_prob + expand_prob:
                    # Expand the obstacle
                    expand = random.uniform(0.1, 0.3)
                    new_x = max(0.5, x - expand * w * 0.25)
                    new_y = max(0.5, y - expand * h * 0.25)
                    new_w = min(49 - new_x, w * (1 + expand * 0.5))
                    new_h = min(29 - new_y, h * (1 + expand * 0.5))
                    
                else:
                    # Gaussian noise
                    new_x = x + np.random.normal(0, gaussian_std)
                    new_y = y + np.random.normal(0, gaussian_std)
                    new_w = max(0.1, w + np.random.normal(0, gaussian_std))
                    new_h = max(0.1, h + np.random.normal(0, gaussian_std))
                
                noisy.append((new_x, new_y, new_w, new_h))
        
        return noisy