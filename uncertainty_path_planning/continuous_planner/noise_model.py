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

class NoiseModel:
    """
    Realistic sensor noise model for Standard CP calibration
    
    Simulates real perception errors:
    - LiDAR measurement noise
    - False negative detections  
    - Localization uncertainty
    - Camera depth estimation errors
    """
    
    def __init__(self, config_path: str = "config/standard_cp_config.yaml"):
        """
        Initialize noise model from configuration
        
        Args:
            config_path: Path to config/standard_cp_config.yaml
        """
        self.config = self.load_config(config_path)
        self.noise_config = self.config['noise_model']
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config['debug']['log_level']))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("NoiseModel initialized")
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
            elif noise_type == "transparency_noise":
                noisy_grid = self._add_transparency_noise(noisy_grid, noise_level)
            elif noise_type == "reflectance_noise":
                noisy_grid = self._add_reflectance_noise(noisy_grid, noise_level)
            elif noise_type == "occlusion_noise":
                noisy_grid = self._add_occlusion_noise(noisy_grid, noise_level)
        
        # Validate result
        self._validate_noisy_grid(occupancy_grid, noisy_grid)
        
        return noisy_grid
    
    def _add_measurement_noise(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add LiDAR/camera measurement noise
        
        Simulates obstacles appearing at wrong distances by shifting boundaries
        """
        lidar_config = self.noise_config['lidar']
        camera_config = self.noise_config['camera']
        
        noisy_grid = grid.copy()
        height, width = grid.shape
        
        # Scale noise by grid resolution (0.05m per pixel)
        noise_std_pixels = (lidar_config['measurement_std'] * noise_level) / 0.05
        
        # Apply morphological operations to shift obstacle boundaries
        from scipy import ndimage
        
        # Create structuring element for morphological ops
        struct_size = max(3, int(noise_std_pixels * 2) | 1)  # Ensure odd size
        structure = ndimage.generate_binary_structure(2, 2)
        
        # Extract obstacle mask as boolean
        obstacle_mask = (grid == 100)
        
        # Random erosion to make obstacles appear farther (70% of the time)
        if np.random.random() < 0.7:
            # Erode obstacles - they appear smaller/farther
            erosion_iterations = max(1, int(noise_std_pixels))
            eroded_mask = ndimage.binary_erosion(obstacle_mask, structure, iterations=erosion_iterations)
            # Clear eroded pixels (obstacles that disappeared)
            noisy_grid[obstacle_mask & ~eroded_mask] = 0
        else:
            # Dilate obstacles - they appear closer (30% of the time)
            dilation_iterations = max(1, int(noise_std_pixels * 0.5))
            dilated_mask = ndimage.binary_dilation(obstacle_mask, structure, iterations=dilation_iterations)
            # Add dilated pixels (new obstacle areas)
            noisy_grid[dilated_mask & ~obstacle_mask] = 100
        
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
    
    def _add_transparency_noise(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add transparency noise - simulates glass/transparent obstacles that LiDAR passes through
        
        Makes random sections of obstacles transparent (appear as free space)
        This creates underestimation of danger as robot thinks it can pass through
        """
        noisy_grid = grid.copy()
        height, width = grid.shape
        
        # Find all obstacle pixels
        obstacle_pixels = np.where(grid == 100)
        num_obstacles = len(obstacle_pixels[0])
        
        if num_obstacles > 0:
            # Make 15-30% of obstacle pixels transparent based on noise level
            transparency_rate = 0.15 + (0.15 * noise_level)
            
            # Select contiguous regions to make transparent (more realistic than random pixels)
            from scipy import ndimage
            
            # Label connected components
            labeled, num_features = ndimage.label(grid == 100)
            
            # For each obstacle component, potentially make parts transparent
            for label_id in range(1, num_features + 1):
                if np.random.random() < transparency_rate:
                    # Make this entire small obstacle transparent
                    component_mask = (labeled == label_id)
                    component_size = np.sum(component_mask)
                    
                    if component_size < 50:  # Small obstacles become fully transparent
                        noisy_grid[component_mask] = 0
                    else:
                        # For larger obstacles, create transparent windows/sections
                        component_coords = np.where(component_mask)
                        num_coords = len(component_coords[0])
                        
                        # Create transparent patches
                        num_patches = max(1, int(num_coords * transparency_rate))
                        for _ in range(num_patches):
                            # Pick random center point
                            idx = np.random.randint(num_coords)
                            cy, cx = component_coords[0][idx], component_coords[1][idx]
                            
                            # Create transparent patch around this point
                            patch_size = np.random.randint(2, 5)
                            y_min = max(0, cy - patch_size)
                            y_max = min(height, cy + patch_size + 1)
                            x_min = max(0, cx - patch_size)
                            x_max = min(width, cx + patch_size + 1)
                            
                            # Make patch transparent
                            noisy_grid[y_min:y_max, x_min:x_max] = 0
        
        self.logger.debug(f"Applied transparency noise with rate {transparency_rate:.2f}")
        return noisy_grid
    
    def _add_reflectance_noise(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add reflectance noise - simulates black/absorptive surfaces and mirror reflections
        
        Black surfaces: LiDAR gets weak/no returns, appears as free space
        Mirrors: Create shifted/displaced obstacles due to indirect reflections
        """
        noisy_grid = grid.copy()
        height, width = grid.shape
        
        # Find obstacle boundaries
        boundaries = self._find_obstacle_boundaries(grid)
        
        if boundaries:
            # Black surface absorption (40% of boundaries)
            black_surface_rate = 0.4 * noise_level
            
            for (i, j) in boundaries:
                if np.random.random() < black_surface_rate:
                    # Black surface - remove obstacle pixels in a gradient
                    # Stronger absorption at edges, weaker towards center
                    absorption_radius = np.random.randint(1, 4)
                    for di in range(-absorption_radius, absorption_radius + 1):
                        for dj in range(-absorption_radius, absorption_radius + 1):
                            ni, nj = i + di, j + dj
                            if 0 <= ni < height and 0 <= nj < width:
                                # Absorption probability decreases with distance
                                dist = max(abs(di), abs(dj))
                                absorption_prob = (1.0 - dist / (absorption_radius + 1)) * noise_level
                                if np.random.random() < absorption_prob:
                                    noisy_grid[ni, nj] = 0
            
            # Mirror reflections (20% of boundaries) - shift obstacle positions
            mirror_rate = 0.2 * noise_level
            
            # Create shifted obstacles from mirrors
            from scipy import ndimage
            obstacle_mask = (grid == 100)
            
            if np.random.random() < mirror_rate:
                # Random shift direction and magnitude
                shift_y = np.random.randint(-3, 4) * noise_level
                shift_x = np.random.randint(-3, 4) * noise_level
                
                # Shift some obstacles to simulate mirror reflections
                shifted_mask = ndimage.shift(obstacle_mask.astype(float), 
                                            [shift_y, shift_x], 
                                            order=0, mode='constant', cval=0)
                
                # Combine: remove some original, add shifted versions
                # This simulates seeing obstacle at wrong position due to reflection
                erosion_mask = np.random.random(grid.shape) < 0.3 * noise_level
                noisy_grid[obstacle_mask & erosion_mask] = 0  # Remove some original
                noisy_grid[shifted_mask.astype(bool) & (grid == 0)] = 100  # Add shifted
        
        self.logger.debug(f"Applied reflectance noise (black surfaces + mirrors)")
        return noisy_grid
    
    def _add_occlusion_noise(self, grid: np.ndarray, noise_level: float) -> np.ndarray:
        """
        Add occlusion noise - simulates partial visibility of obstacles
        
        Only parts of obstacles are visible, underestimating their true size
        This is different from false negatives which remove entire obstacles
        """
        noisy_grid = grid.copy()
        height, width = grid.shape
        
        from scipy import ndimage
        
        # Label connected components (individual obstacles)
        labeled, num_features = ndimage.label(grid == 100)
        
        for label_id in range(1, num_features + 1):
            component_mask = (labeled == label_id)
            
            # Apply occlusion with probability based on noise level
            if np.random.random() < 0.5 + (0.3 * noise_level):
                # Determine occlusion type
                occlusion_type = np.random.choice(['edge', 'corner', 'partial'])
                
                if occlusion_type == 'edge':
                    # Remove one edge of the obstacle (simulate viewing from side)
                    coords = np.where(component_mask)
                    if len(coords[0]) > 0:
                        # Find bounding box
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        
                        # Choose which edge to occlude
                        edge = np.random.choice(['top', 'bottom', 'left', 'right'])
                        occlusion_depth = max(1, int((y_max - y_min) * 0.3 * noise_level))
                        
                        if edge == 'top':
                            noisy_grid[y_min:y_min + occlusion_depth, x_min:x_max + 1] = 0
                        elif edge == 'bottom':
                            noisy_grid[y_max - occlusion_depth + 1:y_max + 1, x_min:x_max + 1] = 0
                        elif edge == 'left':
                            occlusion_depth = max(1, int((x_max - x_min) * 0.3 * noise_level))
                            noisy_grid[y_min:y_max + 1, x_min:x_min + occlusion_depth] = 0
                        else:  # right
                            occlusion_depth = max(1, int((x_max - x_min) * 0.3 * noise_level))
                            noisy_grid[y_min:y_max + 1, x_max - occlusion_depth + 1:x_max + 1] = 0
                
                elif occlusion_type == 'corner':
                    # Remove a corner section (simulate partial view)
                    coords = np.where(component_mask)
                    if len(coords[0]) > 0:
                        y_min, y_max = coords[0].min(), coords[0].max()
                        x_min, x_max = coords[1].min(), coords[1].max()
                        
                        corner = np.random.choice(['tl', 'tr', 'bl', 'br'])
                        occlusion_size = max(2, int(min(y_max - y_min, x_max - x_min) * 0.4 * noise_level))
                        
                        if corner == 'tl':
                            noisy_grid[y_min:y_min + occlusion_size, x_min:x_min + occlusion_size] = 0
                        elif corner == 'tr':
                            noisy_grid[y_min:y_min + occlusion_size, x_max - occlusion_size + 1:x_max + 1] = 0
                        elif corner == 'bl':
                            noisy_grid[y_max - occlusion_size + 1:y_max + 1, x_min:x_min + occlusion_size] = 0
                        else:  # br
                            noisy_grid[y_max - occlusion_size + 1:y_max + 1, x_max - occlusion_size + 1:x_max + 1] = 0
                
                else:  # partial
                    # Random patches occluded (simulate complex occlusion)
                    component_coords = np.where(component_mask)
                    num_pixels = len(component_coords[0])
                    
                    if num_pixels > 0:
                        # Remove 20-40% of the obstacle
                        occlusion_ratio = 0.2 + (0.2 * noise_level)
                        num_occluded = int(num_pixels * occlusion_ratio)
                        
                        # Select pixels to occlude (in clusters for realism)
                        occluded_indices = np.random.choice(num_pixels, size=min(num_occluded, num_pixels), replace=False)
                        
                        for idx in occluded_indices:
                            y, x = component_coords[0][idx], component_coords[1][idx]
                            # Occlude small region around selected pixel
                            for dy in range(-1, 2):
                                for dx in range(-1, 2):
                                    ny, nx = y + dy, x + dx
                                    if 0 <= ny < height and 0 <= nx < width:
                                        if np.random.random() < 0.7:  # Probabilistic for softer edges
                                            noisy_grid[ny, nx] = 0
        
        self.logger.debug(f"Applied occlusion noise to partially hide obstacles")
        return noisy_grid


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