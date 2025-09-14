#!/usr/bin/env python3
"""
Feature Extractor for Learnable Conformal Prediction
Extracts meaningful features from each waypoint for tau prediction
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import cv2
from scipy.spatial.distance import cdist
from scipy.ndimage import distance_transform_edt


class WaypointFeatureExtractor:
    """
    Extracts features from waypoints that are meaningful for safety margin prediction.
    
    Features are designed to capture:
    1. Local geometry (clearances, passage width)
    2. Noise characteristics (transparency, occlusion, localization)
    3. Path context (position along path, curvature)
    4. Environment type (corridor, open space, corner)
    """
    
    def __init__(self, 
                 robot_radius: float = 0.17,
                 resolution: float = 0.05,
                 safe_distance: float = 0.3):
        """
        Initialize feature extractor.
        
        Args:
            robot_radius: Robot radius in meters
            resolution: Map resolution in meters/pixel
            safe_distance: Safe distance threshold in meters
        """
        self.robot_radius = robot_radius
        self.resolution = resolution
        self.safe_distance = safe_distance
        
        # Feature configuration
        self.num_features = 20  # Total number of features
        self.feature_names = [
            # Geometric features (7)
            'min_clearance',           # Minimum distance to obstacles
            'avg_clearance_1m',         # Average clearance within 1m
            'avg_clearance_2m',         # Average clearance within 2m
            'passage_width',            # Width of narrowest passage
            'obstacle_density_1m',      # Obstacle density within 1m
            'obstacle_density_2m',      # Obstacle density within 2m
            'num_obstacles_nearby',     # Number of obstacles within 2m
            
            # Noise-specific features (3)
            'transparency_indicator',   # Likelihood of transparent obstacles
            'occlusion_ratio',         # Ratio of potentially occluded area
            'position_uncertainty',     # Estimated localization uncertainty
            
            # Path context features (5)
            'path_progress',           # Position along path (0 to 1)
            'distance_to_goal',        # Distance to goal
            'path_curvature',          # Local path curvature
            'velocity_magnitude',       # Expected velocity at waypoint
            'heading_change',          # Change in heading at waypoint
            
            # Environment type features (5)
            'is_corridor',             # Binary: in corridor
            'is_open_space',           # Binary: in open space
            'is_near_corner',          # Binary: near corner
            'is_near_doorway',         # Binary: near doorway
            'boundary_distance'        # Distance to map boundary
        ]
        
        assert len(self.feature_names) == self.num_features
    
    def extract_features(self, 
                         waypoint: np.ndarray,
                         path: List[np.ndarray],
                         waypoint_idx: int,
                         occupancy_grid: np.ndarray,
                         origin: np.ndarray,
                         noise_type: Optional[str] = None) -> np.ndarray:
        """
        Extract features for a single waypoint.
        
        Args:
            waypoint: Current waypoint [x, y] in world coordinates
            path: Full path as list of waypoints
            waypoint_idx: Index of current waypoint in path
            occupancy_grid: Occupancy grid (0=free, 100=occupied)
            origin: Map origin in world coordinates
            noise_type: Type of noise applied (for noise-specific features)
            
        Returns:
            Feature vector of shape [num_features]
        """
        features = []
        
        # Convert waypoint to grid coordinates
        grid_pos = self._world_to_grid(waypoint, origin)
        
        # 1. Geometric features
        features.extend(self._extract_geometric_features(
            grid_pos, occupancy_grid
        ))
        
        # 2. Noise-specific features
        features.extend(self._extract_noise_features(
            grid_pos, occupancy_grid, noise_type
        ))
        
        # 3. Path context features
        features.extend(self._extract_path_features(
            waypoint, path, waypoint_idx
        ))
        
        # 4. Environment type features
        features.extend(self._extract_environment_features(
            grid_pos, occupancy_grid
        ))
        
        # Normalize features to similar scales
        features = np.array(features, dtype=np.float32)
        features = self._normalize_features(features)
        
        return features
    
    def _world_to_grid(self, point: np.ndarray, origin: np.ndarray) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        grid_x = int((point[0] - origin[0]) / self.resolution)
        grid_y = int((point[1] - origin[1]) / self.resolution)
        return (grid_x, grid_y)
    
    def _extract_geometric_features(self, 
                                   grid_pos: Tuple[int, int],
                                   occupancy_grid: np.ndarray) -> List[float]:
        """Extract geometry-based features"""
        features = []
        
        # Create distance transform for efficient clearance computation
        obstacle_mask = (occupancy_grid > 50).astype(np.uint8)
        distance_map = distance_transform_edt(1 - obstacle_mask) * self.resolution
        
        x, y = grid_pos
        h, w = occupancy_grid.shape
        
        # Min clearance
        min_clearance = distance_map[min(y, h-1), min(x, w-1)]
        features.append(min_clearance)
        
        # Average clearance within different radii
        for radius_m in [1.0, 2.0]:
            radius_px = int(radius_m / self.resolution)
            avg_clearance = self._compute_avg_clearance_in_radius(
                grid_pos, distance_map, radius_px
            )
            features.append(avg_clearance)
        
        # Passage width
        passage_width = self._compute_passage_width(grid_pos, distance_map)
        features.append(passage_width)
        
        # Obstacle density within different radii
        for radius_m in [1.0, 2.0]:
            radius_px = int(radius_m / self.resolution)
            density = self._compute_obstacle_density(
                grid_pos, obstacle_mask, radius_px
            )
            features.append(density)
        
        # Number of nearby obstacles
        num_obstacles = self._count_nearby_obstacles(grid_pos, obstacle_mask, 2.0)
        features.append(num_obstacles)
        
        return features
    
    def _extract_noise_features(self,
                               grid_pos: Tuple[int, int],
                               occupancy_grid: np.ndarray,
                               noise_type: Optional[str]) -> List[float]:
        """Extract noise-specific features"""
        features = []
        
        # Transparency indicator (glass-like surfaces)
        transparency_score = 0.0
        if noise_type == 'transparency':
            transparency_score = 1.0
        elif noise_type == 'combined':
            transparency_score = 0.33
        features.append(transparency_score)
        
        # Occlusion ratio (hidden areas)
        occlusion_ratio = self._compute_occlusion_ratio(grid_pos, occupancy_grid)
        if noise_type == 'occlusion':
            occlusion_ratio *= 2.0  # Amplify when occlusion noise is active
        elif noise_type == 'combined':
            occlusion_ratio *= 1.33
        features.append(min(occlusion_ratio, 1.0))
        
        # Position uncertainty
        position_uncertainty = 0.0
        if noise_type == 'localization':
            position_uncertainty = 1.0
        elif noise_type == 'combined':
            position_uncertainty = 0.33
        features.append(position_uncertainty)
        
        return features
    
    def _extract_path_features(self,
                              waypoint: np.ndarray,
                              path: List[np.ndarray],
                              waypoint_idx: int) -> List[float]:
        """Extract path-related features"""
        features = []
        
        # Path progress (0 to 1)
        path_progress = waypoint_idx / max(len(path) - 1, 1)
        features.append(path_progress)
        
        # Distance to goal
        goal = path[-1] if path else waypoint
        # Ensure both are numpy arrays
        waypoint_arr = np.array(waypoint) if isinstance(waypoint, (list, tuple)) else waypoint
        goal_arr = np.array(goal) if isinstance(goal, (list, tuple)) else goal
        dist_to_goal = np.linalg.norm(waypoint_arr - goal_arr)
        features.append(dist_to_goal / 10.0)  # Normalize by 10m
        
        # Path curvature
        curvature = self._compute_path_curvature(path, waypoint_idx)
        features.append(curvature)
        
        # Velocity magnitude (based on path segment length)
        if waypoint_idx > 0:
            prev_waypoint = path[waypoint_idx - 1]
            prev_arr = np.array(prev_waypoint) if isinstance(prev_waypoint, (list, tuple)) else prev_waypoint
            segment_length = np.linalg.norm(waypoint_arr - prev_arr)
            velocity = segment_length / 0.1  # Assume 0.1s between waypoints
        else:
            velocity = 0.0
        features.append(min(velocity / 1.0, 1.0))  # Normalize by 1 m/s
        
        # Heading change
        heading_change = self._compute_heading_change(path, waypoint_idx)
        features.append(heading_change / np.pi)  # Normalize by pi
        
        return features
    
    def _extract_environment_features(self,
                                     grid_pos: Tuple[int, int],
                                     occupancy_grid: np.ndarray) -> List[float]:
        """Extract environment type features"""
        features = []
        
        # Detect corridor
        is_corridor = self._detect_corridor(grid_pos, occupancy_grid)
        features.append(float(is_corridor))
        
        # Detect open space
        is_open_space = self._detect_open_space(grid_pos, occupancy_grid)
        features.append(float(is_open_space))
        
        # Detect corner
        is_near_corner = self._detect_corner(grid_pos, occupancy_grid)
        features.append(float(is_near_corner))
        
        # Detect doorway
        is_near_doorway = self._detect_doorway(grid_pos, occupancy_grid)
        features.append(float(is_near_doorway))
        
        # Distance to boundary
        h, w = occupancy_grid.shape
        x, y = grid_pos
        boundary_dist = min(x, y, w - x - 1, h - y - 1) * self.resolution
        features.append(boundary_dist / 5.0)  # Normalize by 5m
        
        return features
    
    def _compute_avg_clearance_in_radius(self,
                                        grid_pos: Tuple[int, int],
                                        distance_map: np.ndarray,
                                        radius_px: int) -> float:
        """Compute average clearance within radius"""
        x, y = grid_pos
        h, w = distance_map.shape
        
        # Create mask for circular region
        y_min = max(0, y - radius_px)
        y_max = min(h, y + radius_px + 1)
        x_min = max(0, x - radius_px)
        x_max = min(w, x + radius_px + 1)
        
        if y_max <= y_min or x_max <= x_min:
            return 0.0
        
        region = distance_map[y_min:y_max, x_min:x_max]
        return np.mean(region)
    
    def _compute_passage_width(self,
                              grid_pos: Tuple[int, int],
                              distance_map: np.ndarray) -> float:
        """Compute width of passage at current position"""
        x, y = grid_pos
        h, w = distance_map.shape
        
        if y >= h or x >= w:
            return 0.0
        
        # Look in 4 directions and find minimum opening
        directions = [(1, 0), (0, 1), (1, 1), (-1, 1)]
        min_width = float('inf')
        
        for dx, dy in directions:
            width = 0
            for step in range(1, 50):  # Check up to 2.5m
                px = x + step * dx
                py = y + step * dy
                nx = x - step * dx
                ny = y - step * dy
                
                if (px < 0 or px >= w or py < 0 or py >= h or
                    nx < 0 or nx >= w or ny < 0 or ny >= h):
                    break
                
                if distance_map[py, px] < self.robot_radius or distance_map[ny, nx] < self.robot_radius:
                    width = step * self.resolution * 2
                    break
            
            if width > 0:
                min_width = min(min_width, width)
        
        return min_width if min_width < float('inf') else 5.0
    
    def _compute_obstacle_density(self,
                                 grid_pos: Tuple[int, int],
                                 obstacle_mask: np.ndarray,
                                 radius_px: int) -> float:
        """Compute obstacle density within radius"""
        x, y = grid_pos
        h, w = obstacle_mask.shape
        
        y_min = max(0, y - radius_px)
        y_max = min(h, y + radius_px + 1)
        x_min = max(0, x - radius_px)
        x_max = min(w, x + radius_px + 1)
        
        if y_max <= y_min or x_max <= x_min:
            return 0.0
        
        region = obstacle_mask[y_min:y_max, x_min:x_max]
        return np.mean(region)
    
    def _count_nearby_obstacles(self,
                               grid_pos: Tuple[int, int],
                               obstacle_mask: np.ndarray,
                               radius_m: float) -> float:
        """Count number of obstacle clusters nearby"""
        radius_px = int(radius_m / self.resolution)
        x, y = grid_pos
        h, w = obstacle_mask.shape
        
        y_min = max(0, y - radius_px)
        y_max = min(h, y + radius_px + 1)
        x_min = max(0, x - radius_px)
        x_max = min(w, x + radius_px + 1)
        
        if y_max <= y_min or x_max <= x_min:
            return 0.0
        
        region = obstacle_mask[y_min:y_max, x_min:x_max].astype(np.uint8)
        
        # Count connected components (obstacle clusters)
        num_labels, _ = cv2.connectedComponents(region)
        return float(min(num_labels - 1, 10)) / 10.0  # Normalize by 10
    
    def _compute_occlusion_ratio(self,
                                grid_pos: Tuple[int, int],
                                occupancy_grid: np.ndarray) -> float:
        """Estimate ratio of potentially occluded area"""
        # Simplified: check for obstacles that might hide other areas
        x, y = grid_pos
        h, w = occupancy_grid.shape
        radius_px = int(2.0 / self.resolution)
        
        occluded_pixels = 0
        total_pixels = 0
        
        for angle in np.linspace(0, 2*np.pi, 16):
            for r in range(5, radius_px):
                px = int(x + r * np.cos(angle))
                py = int(y + r * np.sin(angle))
                
                if 0 <= px < w and 0 <= py < h:
                    total_pixels += 1
                    # Check if line of sight is blocked
                    if self._is_line_blocked(grid_pos, (px, py), occupancy_grid):
                        occluded_pixels += 1
        
        return occluded_pixels / max(total_pixels, 1)
    
    def _is_line_blocked(self,
                        start: Tuple[int, int],
                        end: Tuple[int, int],
                        occupancy_grid: np.ndarray) -> bool:
        """Check if line between two points is blocked"""
        # Bresenham's line algorithm
        x0, y0 = start
        x1, y1 = end
        
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        while (x0, y0) != (x1, y1):
            if occupancy_grid[y0, x0] > 50:
                return True
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        
        return False
    
    def _compute_path_curvature(self,
                               path: List[np.ndarray],
                               idx: int) -> float:
        """Compute local path curvature"""
        if len(path) < 3 or idx == 0 or idx >= len(path) - 1:
            return 0.0
        
        # Use three points to estimate curvature
        p0 = np.array(path[idx - 1]) if isinstance(path[idx - 1], (list, tuple)) else path[idx - 1]
        p1 = np.array(path[idx]) if isinstance(path[idx], (list, tuple)) else path[idx]
        p2 = np.array(path[idx + 1]) if isinstance(path[idx + 1], (list, tuple)) else path[idx + 1]
        
        # Compute angle change
        v1 = p1 - p0
        v2 = p2 - p1
        
        if np.linalg.norm(v1) < 1e-6 or np.linalg.norm(v2) < 1e-6:
            return 0.0
        
        v1_norm = v1 / np.linalg.norm(v1)
        v2_norm = v2 / np.linalg.norm(v2)
        
        # Angle between vectors
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # Approximate curvature
        avg_length = (np.linalg.norm(v1) + np.linalg.norm(v2)) / 2
        curvature = angle / max(avg_length, 0.1)
        
        return min(curvature, 10.0) / 10.0  # Normalize
    
    def _compute_heading_change(self,
                               path: List[np.ndarray],
                               idx: int) -> float:
        """Compute heading change at waypoint"""
        if len(path) < 2 or idx == 0 or idx >= len(path) - 1:
            return 0.0
        
        # Compute heading before and after
        curr_pt = np.array(path[idx]) if isinstance(path[idx], (list, tuple)) else path[idx]
        prev_pt = np.array(path[idx - 1]) if isinstance(path[idx - 1], (list, tuple)) else path[idx - 1]
        next_pt = np.array(path[idx + 1]) if idx < len(path) - 1 and isinstance(path[idx + 1], (list, tuple)) else (path[idx + 1] if idx < len(path) - 1 else curr_pt)
        
        v_before = curr_pt - prev_pt
        v_after = next_pt - curr_pt if idx < len(path) - 1 else v_before
        
        if np.linalg.norm(v_before) < 1e-6 or np.linalg.norm(v_after) < 1e-6:
            return 0.0
        
        heading_before = np.arctan2(v_before[1], v_before[0])
        heading_after = np.arctan2(v_after[1], v_after[0])
        
        # Compute angle difference
        heading_change = heading_after - heading_before
        
        # Normalize to [-pi, pi]
        while heading_change > np.pi:
            heading_change -= 2 * np.pi
        while heading_change < -np.pi:
            heading_change += 2 * np.pi
        
        return abs(heading_change)
    
    def _detect_corridor(self,
                        grid_pos: Tuple[int, int],
                        occupancy_grid: np.ndarray) -> bool:
        """Detect if position is in a corridor"""
        x, y = grid_pos
        h, w = occupancy_grid.shape
        
        # Check for walls on opposite sides
        check_dist = int(2.0 / self.resolution)
        
        # Check horizontal corridor
        left_wall = False
        right_wall = False
        for dy in range(-2, 3):
            if x - check_dist >= 0 and occupancy_grid[min(y + dy, h-1), x - check_dist] > 50:
                left_wall = True
            if x + check_dist < w and occupancy_grid[min(y + dy, h-1), x + check_dist] > 50:
                right_wall = True
        
        # Check vertical corridor
        top_wall = False
        bottom_wall = False
        for dx in range(-2, 3):
            if y - check_dist >= 0 and occupancy_grid[y - check_dist, min(x + dx, w-1)] > 50:
                top_wall = True
            if y + check_dist < h and occupancy_grid[y + check_dist, min(x + dx, w-1)] > 50:
                bottom_wall = True
        
        return (left_wall and right_wall) or (top_wall and bottom_wall)
    
    def _detect_open_space(self,
                          grid_pos: Tuple[int, int],
                          occupancy_grid: np.ndarray) -> bool:
        """Detect if position is in open space"""
        x, y = grid_pos
        h, w = occupancy_grid.shape
        
        # Check if large area around is free
        check_radius = int(3.0 / self.resolution)
        
        y_min = max(0, y - check_radius)
        y_max = min(h, y + check_radius + 1)
        x_min = max(0, x - check_radius)
        x_max = min(w, x + check_radius + 1)
        
        if y_max <= y_min or x_max <= x_min:
            return False
        
        region = occupancy_grid[y_min:y_max, x_min:x_max]
        obstacle_ratio = np.mean(region > 50)
        
        return obstacle_ratio < 0.1  # Less than 10% obstacles
    
    def _detect_corner(self,
                      grid_pos: Tuple[int, int],
                      occupancy_grid: np.ndarray) -> bool:
        """Detect if position is near a corner"""
        x, y = grid_pos
        h, w = occupancy_grid.shape
        
        # Check for L-shaped obstacle configuration
        check_dist = int(1.5 / self.resolution)
        
        # Check all four corner configurations
        corners = [
            # Top-left corner
            [(x - check_dist, y), (x, y - check_dist)],
            # Top-right corner
            [(x + check_dist, y), (x, y - check_dist)],
            # Bottom-left corner
            [(x - check_dist, y), (x, y + check_dist)],
            # Bottom-right corner
            [(x + check_dist, y), (x, y + check_dist)]
        ]
        
        for corner_points in corners:
            walls_found = 0
            for px, py in corner_points:
                if 0 <= px < w and 0 <= py < h:
                    if occupancy_grid[int(py), int(px)] > 50:
                        walls_found += 1
            
            if walls_found >= 2:
                return True
        
        return False
    
    def _detect_doorway(self,
                       grid_pos: Tuple[int, int],
                       occupancy_grid: np.ndarray) -> bool:
        """Detect if position is near a doorway"""
        x, y = grid_pos
        h, w = occupancy_grid.shape
        
        # Look for narrow passage between rooms
        passage_width = self._compute_passage_width(grid_pos, 
                                                   distance_transform_edt(1 - (occupancy_grid > 50)) * self.resolution)
        
        # Doorway is typically 0.8-1.5m wide
        return 0.8 <= passage_width <= 1.5
    
    def _normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to similar scales"""
        # Most features are already normalized to [0, 1] or similar range
        # Clip extreme values
        features = np.clip(features, -5.0, 5.0)
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names
    
    def get_num_features(self) -> int:
        """Get number of features"""
        return self.num_features