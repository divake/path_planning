#!/usr/bin/env python3
"""
Standard CP Noise Model
Realistic perception noise simulation for MRPB environments
"""

import numpy as np
import yaml
from typing import Dict, Optional, Tuple
import random
import logging

class StandardCPNoiseModel:
    """
    Realistic sensor noise model for Standard CP calibration
    
    Simulates real perception errors:
    - LiDAR measurement noise
    - False negative detections  
    - Localization uncertainty
    - Camera depth estimation errors
    """
    
    def __init__(self, config_path: str = "standard_cp_config.yaml"):
        """
        Initialize noise model from configuration
        
        Args:
            config_path: Path to standard_cp_config.yaml
        """
        self.config = self.load_config(config_path)
        self.noise_config = self.config['noise_model']
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config['debug']['log_level']))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("StandardCPNoiseModel initialized")
        self.logger.info(f"Noise types: {self.noise_config['noise_types']}")
        
    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.error(f"Config file not found: {config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.error(f"Error parsing config file: {e}")
            raise
    
    def add_realistic_noise(self, 
                           occupancy_grid: np.ndarray, 
                           noise_level: float = 0.1,
                           seed: Optional[int] = None) -> np.ndarray:
        """
        Add realistic perception noise to occupancy grid
        
        Args:
            occupancy_grid: Original occupancy grid (0=free, 1=occupied)
            noise_level: Noise intensity scale factor (0.05-0.20 typical)
            seed: Random seed for reproducibility
            
        Returns:
            Noisy occupancy grid with same shape
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            
        self.logger.debug(f"Adding noise level {noise_level} with seed {seed}")
        
        # Start with copy of original grid
        noisy_grid = occupancy_grid.copy()
        height, width = occupancy_grid.shape
        
        # Apply each noise type based on configuration
        for noise_type in self.noise_config['noise_types']:
            if noise_type == "measurement_noise":
                noisy_grid = self._add_measurement_noise(noisy_grid, noise_level)
            elif noise_type == "false_negatives":
                noisy_grid = self._add_false_negatives(noisy_grid, noise_level)
            elif noise_type == "localization_drift":
                noisy_grid = self._add_localization_drift(noisy_grid, noise_level)
        
        # Validate result
        self._validate_noisy_grid(occupancy_grid, noisy_grid)
        
        return noisy_grid
    
    def _add_measurement_noise(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add LiDAR/camera measurement noise
        
        Simulates:
        - Gaussian noise on obstacle boundaries
        - Random erosion/dilation of obstacles
        """
        lidar_config = self.noise_config['lidar']
        camera_config = self.noise_config['camera']
        
        noisy_grid = grid.copy()
        height, width = grid.shape
        
        # Scale noise by grid resolution (0.05m per pixel)
        noise_std_pixels = (lidar_config['measurement_std'] * noise_level) / 0.05
        
        # Find obstacle boundaries for targeted noise
        obstacle_boundaries = self._find_obstacle_boundaries(grid)
        
        for (i, j) in obstacle_boundaries:
            # Gaussian noise on boundary pixels
            noise_magnitude = abs(np.random.normal(0, noise_std_pixels))
            
            if noise_magnitude > 0.5:  # Significant noise
                # Randomly erode or dilate
                if np.random.random() < 0.5:
                    # Erode: remove obstacle pixel (make free)
                    noisy_grid[i, j] = 0
                else:
                    # Dilate: add obstacle pixels in neighborhood
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                if np.random.random() < noise_level:
                                    noisy_grid[ni, nj] = 100  # MRPB occupied value
        
        # Camera depth estimation errors (additional false negatives)
        if np.random.random() < camera_config['failure_rate'] * noise_level:
            # Random patch failures
            patch_size = np.random.randint(5, 15)
            patch_x = np.random.randint(0, max(1, width - patch_size))
            patch_y = np.random.randint(0, max(1, height - patch_size))
            
            # Remove obstacles in patch (depth estimation failure)
            noisy_grid[patch_y:patch_y+patch_size, patch_x:patch_x+patch_size] = 0
        
        self.logger.debug(f"Applied measurement noise: {noise_std_pixels:.2f} pixels std")
        return noisy_grid
    
    def _add_false_negatives(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add false negative detections (missing obstacles)
        
        Simulates:
        - Obstacles outside sensor range
        - Sensor occlusion
        - Detection algorithm failures
        """
        lidar_config = self.noise_config['lidar']
        
        noisy_grid = grid.copy()
        height, width = grid.shape
        
        # Scale false negative rate by noise level
        false_negative_rate = lidar_config['false_negative_rate'] * noise_level
        
        # Apply false negatives randomly to occupied cells
        occupied_cells = np.where(grid == 100)  # MRPB occupied value
        num_occupied = len(occupied_cells[0])
        
        if num_occupied > 0:
            # Select random subset for false negatives
            num_false_negatives = int(num_occupied * false_negative_rate)
            false_negative_indices = np.random.choice(
                num_occupied, 
                size=min(num_false_negatives, num_occupied), 
                replace=False
            )
            
            # Remove selected obstacles
            for idx in false_negative_indices:
                i, j = occupied_cells[0][idx], occupied_cells[1][idx]
                noisy_grid[i, j] = 0
        
        self.logger.debug(f"Applied false negatives: {false_negative_rate:.3f} rate")
        return noisy_grid
    
    def _add_localization_drift(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add localization uncertainty (map drift)
        
        Simulates:
        - Robot position uncertainty
        - Map registration errors
        - SLAM drift
        """
        loc_config = self.noise_config['localization']
        
        # Scale position uncertainty by noise level and convert to pixels
        position_std_pixels = (loc_config['position_std'] * noise_level) / 0.05
        orientation_std = loc_config['orientation_std'] * noise_level
        
        # Random translation
        shift_x = int(np.random.normal(0, position_std_pixels))
        shift_y = int(np.random.normal(0, position_std_pixels))
        
        # Apply translation if significant
        if abs(shift_x) > 0 or abs(shift_y) > 0:
            noisy_grid = np.roll(grid, (shift_y, shift_x), axis=(0, 1))
            
            # Fill boundary with zeros (unknown space)
            if shift_x > 0:
                noisy_grid[:, :shift_x] = 0
            elif shift_x < 0:
                noisy_grid[:, shift_x:] = 0
            if shift_y > 0:
                noisy_grid[:shift_y, :] = 0
            elif shift_y < 0:
                noisy_grid[shift_y:, :] = 0
        else:
            noisy_grid = grid.copy()
        
        # TODO: Add rotation noise if needed (more complex)
        
        self.logger.debug(f"Applied localization drift: ({shift_x}, {shift_y}) pixels")
        return noisy_grid
    
    def _find_obstacle_boundaries(self, grid: np.ndarray) -> list:
        """Find pixels on obstacle boundaries for targeted noise"""
        boundaries = []
        height, width = grid.shape
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                if grid[i, j] == 100:  # MRPB occupied cell
                    # Check if it's on boundary (adjacent to free space)
                    neighbors = [
                        grid[i-1, j], grid[i+1, j], 
                        grid[i, j-1], grid[i, j+1]
                    ]
                    if 0 in neighbors:  # Adjacent to free space
                        boundaries.append((i, j))
        
        return boundaries
    
    def _validate_noisy_grid(self, original: np.ndarray, noisy: np.ndarray):
        """Validate that noisy grid is reasonable"""
        # Check shape preservation
        assert original.shape == noisy.shape, "Grid shape changed during noise application"
        
        # Check values are valid for MRPB grids (0=free, 50=unknown, 100=occupied)
        unique_values = np.unique(noisy)
        valid_values = set([0, 50, 100])
        assert set(unique_values).issubset(valid_values), f"Invalid values in noisy grid: {unique_values}"
        
        # Check noise is not too extreme
        diff_ratio = np.sum(original != noisy) / original.size
        if diff_ratio > 0.5:  # More than 50% changed
            self.logger.warning(f"Very high noise level: {diff_ratio:.2f} of pixels changed")
        
        self.logger.debug(f"Noise validation passed: {diff_ratio:.3f} pixels changed")
    
    def generate_noise_sequence(self, 
                               base_grid: np.ndarray, 
                               num_samples: int,
                               noise_level: float = 0.1,
                               base_seed: int = 42) -> list:
        """
        Generate sequence of noisy grids for calibration
        
        Args:
            base_grid: Original clean occupancy grid
            num_samples: Number of noisy samples to generate
            noise_level: Noise intensity
            base_seed: Base seed for reproducible sequence
            
        Returns:
            List of noisy grids
        """
        self.logger.info(f"Generating {num_samples} noisy samples at level {noise_level}")
        
        noisy_grids = []
        
        for i in range(num_samples):
            # Use deterministic seed sequence
            sample_seed = base_seed + i
            noisy_grid = self.add_realistic_noise(base_grid, noise_level, sample_seed)
            noisy_grids.append(noisy_grid)
            
            if (i + 1) % 10 == 0:
                self.logger.debug(f"Generated {i + 1}/{num_samples} noisy samples")
        
        return noisy_grids
    
    def analyze_noise_impact(self, original: np.ndarray, noisy: np.ndarray) -> Dict:
        """
        Analyze the impact of noise on the grid
        
        Returns metrics for validation and debugging
        """
        # Basic statistics
        total_pixels = original.size
        changed_pixels = np.sum(original != noisy)
        change_ratio = changed_pixels / total_pixels
        
        # Obstacle-specific changes
        original_obstacles = np.sum(original == 100)  # MRPB occupied
        noisy_obstacles = np.sum(noisy == 100)
        obstacle_change = (noisy_obstacles - original_obstacles) / max(original_obstacles, 1)
        
        # False negatives and positives
        false_negatives = np.sum((original == 100) & (noisy == 0))
        false_positives = np.sum((original == 0) & (noisy == 100))
        
        analysis = {
            'total_pixels': total_pixels,
            'changed_pixels': changed_pixels,
            'change_ratio': change_ratio,
            'original_obstacles': original_obstacles,
            'noisy_obstacles': noisy_obstacles,
            'obstacle_change_ratio': obstacle_change,
            'false_negatives': false_negatives,
            'false_positives': false_positives,
            'false_negative_rate': false_negatives / max(original_obstacles, 1),
            'false_positive_rate': false_positives / max(np.sum(original == 0), 1)
        }
        
        self.logger.debug(f"Noise analysis: {change_ratio:.3f} change ratio, "
                         f"{false_negatives} false negatives, {false_positives} false positives")
        
        return analysis


def test_noise_model():
    """Simple test of noise model functionality"""
    print("Testing StandardCP Noise Model...")
    
    # Create test grid (MRPB format)
    test_grid = np.zeros((100, 100), dtype=int)
    test_grid[40:60, 40:60] = 100  # Square obstacle (MRPB occupied value)
    
    # Initialize noise model
    noise_model = StandardCPNoiseModel()
    
    # Test noise application
    noisy_grid = noise_model.add_realistic_noise(test_grid, noise_level=0.15, seed=42)
    
    # Analyze results
    analysis = noise_model.analyze_noise_impact(test_grid, noisy_grid)
    
    print(f"Original obstacles: {analysis['original_obstacles']}")
    print(f"Noisy obstacles: {analysis['noisy_obstacles']}")
    print(f"Change ratio: {analysis['change_ratio']:.3f}")
    print(f"False negative rate: {analysis['false_negative_rate']:.3f}")
    
    print("Noise model test completed successfully!")


if __name__ == "__main__":
    test_noise_model()