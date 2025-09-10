#!/usr/bin/env python3
"""
Standard CP Nonconformity Scores
Implementation of perception error measurement for conformal prediction
"""

import numpy as np
import yaml
from typing import List, Tuple, Dict, Optional
import logging
from scipy.ndimage import distance_transform_edt
import time

class StandardCPNonconformity:
    """
    Nonconformity score calculation for Standard CP
    
    Implements the corrected approach:
    Score = Maximum perception underestimation of safety clearance along path
    
    This measures "how much did perception underestimate danger?"
    """
    
    def __init__(self, config_path: str = "standard_cp_config.yaml"):
        """
        Initialize nonconformity calculator
        
        Args:
            config_path: Path to standard_cp_config.yaml
        """
        self.config = self.load_config(config_path)
        self.nc_config = self.config['conformal_prediction']['nonconformity']
        self.robot_config = self.config['robot']
        
        # Setup logging
        logging.basicConfig(level=getattr(logging, self.config['debug']['log_level']))
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("StandardCP Nonconformity initialized")
        self.logger.info(f"Method: {self.nc_config['method']}")
        self.logger.info(f"Robot radius: {self.robot_config['radius']}m")
        
        # Cache for distance transforms (optimization)
        self._distance_cache = {}
        
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
    
    def compute_nonconformity_score(self,
                                   true_grid: np.ndarray,
                                   perceived_grid: np.ndarray, 
                                   path: List[Tuple[float, float]],
                                   path_id: str = "unknown") -> float:
        """
        Compute nonconformity score for a path
        
        Args:
            true_grid: Ground truth occupancy grid
            perceived_grid: Perceived occupancy grid (with noise)
            path: List of (x, y) waypoints in world coordinates
            path_id: Identifier for logging/debugging
            
        Returns:
            Nonconformity score (meters) - higher = more dangerous
        """
        start_time = time.time()
        
        if not path or len(path) == 0:
            score = self.nc_config['penalty_planning_failure']
            self.logger.debug(f"Planning failure penalty: {score}m")
            return score
        
        self.logger.debug(f"Computing nonconformity for path {path_id} with {len(path)} waypoints")
        
        # Select method
        if self.nc_config['method'] == "clearance_error":
            score = self._compute_clearance_error_score(true_grid, perceived_grid, path)
        elif self.nc_config['method'] == "collision_depth":
            score = self._compute_collision_depth_score(true_grid, perceived_grid, path)
        elif self.nc_config['method'] == "hausdorff":
            score = self._compute_hausdorff_score(true_grid, perceived_grid, path)
        else:
            raise ValueError(f"Unknown nonconformity method: {self.nc_config['method']}")
        
        computation_time = time.time() - start_time
        self.logger.debug(f"Nonconformity score: {score:.4f}m (computed in {computation_time:.3f}s)")
        
        return score
    
    def _compute_clearance_error_score(self,
                                      true_grid: np.ndarray,
                                      perceived_grid: np.ndarray,
                                      path: List[Tuple[float, float]]) -> float:
        """
        Compute maximum clearance underestimation along path
        
        This is the main method - measures perception error directly
        """
        robot_radius = self.robot_config['radius']
        max_underestimation = 0.0
        
        # Compute distance transforms for efficiency
        true_distance_map = self._get_distance_transform(true_grid)
        perceived_distance_map = self._get_distance_transform(perceived_grid)
        
        for i, waypoint in enumerate(path):
            # Convert world coordinates to grid coordinates
            grid_x, grid_y = self._world_to_grid(waypoint, true_grid)
            
            # Skip if outside grid bounds
            if not self._is_valid_grid_position(grid_x, grid_y, true_grid):
                continue
            
            # Get clearances from distance maps (in meters)
            true_clearance = true_distance_map[grid_y, grid_x] * 0.05  # Convert to meters
            perceived_clearance = perceived_distance_map[grid_y, grid_x] * 0.05
            
            # Compute robot safety margins
            true_safety_margin = true_clearance - robot_radius
            perceived_safety_margin = perceived_clearance - robot_radius
            
            # Perception underestimation = how much we thought we were safer than reality
            underestimation = max(0, perceived_safety_margin - true_safety_margin)
            
            # Special penalty for actual collisions
            if true_safety_margin < 0:  # Robot actually collides in true environment
                collision_penalty = abs(true_safety_margin) + self.nc_config['penalty_collision']
                underestimation = max(underestimation, collision_penalty)
                
                self.logger.debug(f"Collision at waypoint {i}: true_margin={true_safety_margin:.3f}m, "
                                f"penalty={collision_penalty:.3f}m")
            
            max_underestimation = max(max_underestimation, underestimation)
        
        return max_underestimation
    
    def _compute_collision_depth_score(self,
                                      true_grid: np.ndarray,
                                      perceived_grid: np.ndarray,
                                      path: List[Tuple[float, float]]) -> float:
        """
        Alternative method: compute maximum collision depth
        
        Less preferred but included for comparison
        """
        robot_radius = self.robot_config['radius']
        max_collision_depth = 0.0
        
        true_distance_map = self._get_distance_transform(true_grid)
        
        for waypoint in path:
            grid_x, grid_y = self._world_to_grid(waypoint, true_grid)
            
            if not self._is_valid_grid_position(grid_x, grid_y, true_grid):
                continue
            
            true_clearance = true_distance_map[grid_y, grid_x] * 0.05
            safety_margin = true_clearance - robot_radius
            
            if safety_margin < 0:  # Collision
                collision_depth = abs(safety_margin)
                max_collision_depth = max(max_collision_depth, collision_depth)
        
        return max_collision_depth
    
    def _compute_hausdorff_score(self,
                                true_grid: np.ndarray,
                                perceived_grid: np.ndarray,
                                path: List[Tuple[float, float]]) -> float:
        """
        Alternative method: Hausdorff distance between obstacle sets
        
        Measures overall shape difference, not path-specific
        """
        # Find obstacle centroids
        true_obstacles = np.argwhere(true_grid == 1)
        perceived_obstacles = np.argwhere(perceived_grid == 1)
        
        if len(true_obstacles) == 0 or len(perceived_obstacles) == 0:
            return 0.0
        
        # Compute Hausdorff distance
        max_dist = 0.0
        
        for true_obs in true_obstacles:
            min_dist = float('inf')
            for perc_obs in perceived_obstacles:
                dist = np.linalg.norm(true_obs - perc_obs) * 0.05  # Convert to meters
                min_dist = min(min_dist, dist)
            max_dist = max(max_dist, min_dist)
        
        return max_dist
    
    def _get_distance_transform(self, grid: np.ndarray) -> np.ndarray:
        """
        Compute distance transform with caching for efficiency
        
        Returns distance to nearest obstacle for each pixel
        """
        # Create cache key from grid hash
        grid_hash = hash(grid.tobytes())
        
        if grid_hash in self._distance_cache:
            return self._distance_cache[grid_hash]
        
        # Compute distance transform
        # distance_transform_edt gives distance to nearest 'True' pixel
        # We want distance to nearest obstacle (100 in MRPB), so invert
        free_space_mask = (grid != 100)  # Everything except obstacles
        distance_map = distance_transform_edt(free_space_mask)
        
        # Cache result
        self._distance_cache[grid_hash] = distance_map
        
        # Limit cache size
        if len(self._distance_cache) > 50:
            # Remove oldest entry
            oldest_key = next(iter(self._distance_cache))
            del self._distance_cache[oldest_key]
        
        return distance_map
    
    def _world_to_grid(self, world_pos: Tuple[float, float], grid: np.ndarray) -> Tuple[int, int]:
        """
        Convert world coordinates to grid coordinates
        
        Assumes:
        - Grid resolution: 0.05m per pixel
        - Grid origin: (-grid_width*0.05/2, -grid_height*0.05/2)
        """
        x, y = world_pos
        height, width = grid.shape
        
        # Convert to grid coordinates (assuming centered grid)
        grid_x = int((x + width * 0.05 / 2) / 0.05)
        grid_y = int((y + height * 0.05 / 2) / 0.05)
        
        return grid_x, grid_y
    
    def _is_valid_grid_position(self, grid_x: int, grid_y: int, grid: np.ndarray) -> bool:
        """Check if grid coordinates are within bounds"""
        height, width = grid.shape
        return 0 <= grid_x < width and 0 <= grid_y < height
    
    def compute_batch_scores(self,
                           true_grid: np.ndarray,
                           perceived_grids: List[np.ndarray],
                           paths: List[List[Tuple[float, float]]]) -> List[float]:
        """
        Compute nonconformity scores for multiple paths efficiently
        
        Args:
            true_grid: Ground truth grid
            perceived_grids: List of noisy grids
            paths: List of paths (one per perceived grid)
            
        Returns:
            List of nonconformity scores
        """
        self.logger.info(f"Computing batch nonconformity scores for {len(paths)} paths")
        
        scores = []
        
        for i, (perceived_grid, path) in enumerate(zip(perceived_grids, paths)):
            score = self.compute_nonconformity_score(
                true_grid, perceived_grid, path, path_id=f"batch_{i}"
            )
            scores.append(score)
            
            if (i + 1) % 10 == 0:
                self.logger.debug(f"Computed {i + 1}/{len(paths)} scores")
        
        return scores
    
    def analyze_score_distribution(self, scores: List[float]) -> Dict:
        """
        Analyze distribution of nonconformity scores
        
        Useful for calibration validation and debugging
        """
        if not scores:
            return {}
        
        scores_array = np.array(scores)
        
        analysis = {
            'count': len(scores),
            'mean': np.mean(scores_array),
            'std': np.std(scores_array),
            'min': np.min(scores_array),
            'max': np.max(scores_array),
            'median': np.median(scores_array),
            'percentiles': {
                '25%': np.percentile(scores_array, 25),
                '50%': np.percentile(scores_array, 50),
                '75%': np.percentile(scores_array, 75),
                '90%': np.percentile(scores_array, 90),
                '95%': np.percentile(scores_array, 95),
                '99%': np.percentile(scores_array, 99)
            },
            'zero_scores': np.sum(scores_array == 0),
            'high_scores': np.sum(scores_array > 0.5),  # Scores > 50cm
            'extreme_scores': np.sum(scores_array > 1.0)  # Scores > 1m
        }
        
        self.logger.info(f"Score distribution: mean={analysis['mean']:.4f}m, "
                        f"std={analysis['std']:.4f}m, range=[{analysis['min']:.4f}, {analysis['max']:.4f}]m")
        
        return analysis
    
    def validate_scores(self, scores: List[float]) -> bool:
        """
        Validate that nonconformity scores are reasonable
        
        Returns:
            True if scores pass validation, False otherwise
        """
        if not scores:
            self.logger.warning("Empty scores list")
            return False
        
        scores_array = np.array(scores)
        
        # Check for negative scores
        if np.any(scores_array < 0):
            self.logger.error("Found negative nonconformity scores")
            return False
        
        # Check for extremely high scores (> 2m suggests implementation error)
        extreme_scores = scores_array > 2.0
        if np.sum(extreme_scores) > len(scores) * 0.01:  # More than 1% extreme
            self.logger.warning(f"Found {np.sum(extreme_scores)} extreme scores (>2m)")
        
        # Check for all-zero scores (suggests no noise or implementation issue)
        if np.all(scores_array == 0):
            self.logger.warning("All nonconformity scores are zero")
            return False
        
        # Check for reasonable distribution
        if np.std(scores_array) < 0.001:  # Very low variance
            self.logger.warning(f"Very low score variance: {np.std(scores_array):.6f}")
        
        self.logger.debug("Nonconformity score validation passed")
        return True


def test_nonconformity_calculator():
    """Simple test of nonconformity calculator"""
    print("Testing StandardCP Nonconformity Calculator...")
    
    # Create test grids (MRPB format)
    true_grid = np.zeros((100, 100), dtype=int)
    true_grid[40:60, 40:60] = 100  # Square obstacle (MRPB occupied value)
    
    # Create noisy version (remove some obstacles)
    perceived_grid = true_grid.copy()
    perceived_grid[45:55, 45:55] = 0  # Remove center part
    
    # Create test path
    path = [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (2.5, 2.5)]  # Diagonal path
    
    # Initialize calculator
    calc = StandardCPNonconformity()
    
    # Compute score
    score = calc.compute_nonconformity_score(true_grid, perceived_grid, path)
    
    print(f"Nonconformity score: {score:.4f}m")
    
    # Test batch computation
    scores = calc.compute_batch_scores(
        true_grid, 
        [perceived_grid, perceived_grid], 
        [path, path]
    )
    
    print(f"Batch scores: {scores}")
    
    # Analyze distribution
    analysis = calc.analyze_score_distribution(scores)
    print(f"Score analysis: {analysis}")
    
    print("Nonconformity calculator test completed successfully!")


if __name__ == "__main__":
    test_nonconformity_calculator()