"""
Feature extraction module for uncertainty quantification.
Extracts relevant features from the environment for learnable CP.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial import KDTree


class FeatureExtractor:
    """Extract features for uncertainty prediction."""
    
    def __init__(self, environment):
        """
        Initialize feature extractor.
        
        Args:
            environment: UncertaintyEnvironment instance
        """
        self.env = environment
        self.obstacle_centers = None
        self.kdtree = None
        self._build_kdtree()
    
    def _build_kdtree(self):
        """Build KD-tree for efficient nearest neighbor queries."""
        if self.env.original_obstacles:
            self.obstacle_centers = np.array([obs['center'] 
                                             for obs in self.env.original_obstacles])
            self.kdtree = KDTree(self.obstacle_centers)
    
    def extract_point_features(self, point: Tuple[float, float]) -> np.ndarray:
        """
        Extract features for a single point.
        
        Args:
            point: (x, y) position
        
        Returns:
            Feature vector of shape (10,)
        """
        features = []
        
        # 1. Distance to nearest obstacle
        dist_nearest = self._distance_to_nearest_obstacle(point)
        features.append(dist_nearest / max(self.env.width, self.env.height))
        
        # 2. Obstacle density within 5 units
        density_5 = self._obstacle_density(point, radius=5.0)
        features.append(density_5)
        
        # 3. Obstacle density within 10 units
        density_10 = self._obstacle_density(point, radius=10.0)
        features.append(density_10)
        
        # 4. Passage width at current location
        passage_width = self._estimate_passage_width(point)
        features.append(passage_width / max(self.env.width, self.env.height))
        
        # 5. Distance from boundary
        dist_boundary = self._distance_from_boundary(point)
        features.append(dist_boundary / max(self.env.width, self.env.height))
        
        # 6. Number of obstacles in quadrants around point
        quadrant_counts = self._quadrant_obstacle_count(point)
        features.extend(quadrant_counts)  # Adds 4 features
        
        # 7. Asymmetry measure
        asymmetry = self._obstacle_asymmetry(point)
        features.append(asymmetry)
        
        return np.array(features, dtype=np.float32)
    
    def extract_path_features(self, path: List[Tuple[float, float]]) -> np.ndarray:
        """
        Extract features along entire path.
        
        Args:
            path: List of (x, y) points
        
        Returns:
            Feature matrix of shape (len(path), 10)
        """
        if not path:
            return np.array([])
        
        features = []
        for point in path:
            features.append(self.extract_point_features(point))
        
        return np.array(features)
    
    def extract_global_features(self) -> np.ndarray:
        """
        Extract global environment features.
        
        Returns:
            Global feature vector
        """
        features = []
        
        # Total number of obstacles
        num_obstacles = len(self.env.original_obstacles)
        features.append(num_obstacles / 100.0)  # Normalize
        
        # Average obstacle radius
        if self.env.original_obstacles:
            avg_radius = np.mean([obs['radius'] for obs in self.env.original_obstacles])
            features.append(avg_radius / 5.0)  # Normalize assuming max radius ~5
        else:
            features.append(0.0)
        
        # Obstacle coverage ratio
        total_area = self.env.width * self.env.height
        obstacle_area = sum([np.pi * obs['radius']**2 
                           for obs in self.env.original_obstacles])
        features.append(obstacle_area / total_area)
        
        # Clustering measure (average distance between obstacles)
        if len(self.obstacle_centers) > 1:
            distances = []
            for i in range(len(self.obstacle_centers)):
                dists, _ = self.kdtree.query(self.obstacle_centers[i], k=2)
                distances.append(dists[1])  # Distance to nearest neighbor
            avg_spacing = np.mean(distances)
            features.append(avg_spacing / max(self.env.width, self.env.height))
        else:
            features.append(1.0)
        
        return np.array(features, dtype=np.float32)
    
    def _distance_to_nearest_obstacle(self, point: Tuple[float, float]) -> float:
        """Calculate distance to nearest obstacle edge."""
        if self.kdtree is None:
            return float('inf')
        
        dist, idx = self.kdtree.query(point)
        obstacle = self.env.original_obstacles[idx]
        
        # Distance to obstacle edge (not center)
        edge_dist = max(0, dist - obstacle['radius'])
        return edge_dist
    
    def _obstacle_density(self, point: Tuple[float, float], radius: float) -> float:
        """Calculate obstacle density within radius of point."""
        if self.kdtree is None:
            return 0.0
        
        indices = self.kdtree.query_ball_point(point, radius)
        
        # Calculate total obstacle area within radius
        area_sum = 0
        for idx in indices:
            obs = self.env.original_obstacles[idx]
            area_sum += np.pi * obs['radius']**2
        
        # Normalize by search area
        search_area = np.pi * radius**2
        density = area_sum / search_area
        
        return min(1.0, density)  # Cap at 1.0
    
    def _estimate_passage_width(self, point: Tuple[float, float]) -> float:
        """Estimate width of passage at current point."""
        if self.kdtree is None:
            return max(self.env.width, self.env.height)
        
        # Find nearest obstacles
        dists, indices = self.kdtree.query(point, k=min(5, len(self.obstacle_centers)))
        
        if len(indices) < 2:
            return max(self.env.width, self.env.height)
        
        # Estimate passage width using nearest obstacles
        min_gap = float('inf')
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                obs1 = self.env.original_obstacles[indices[i]]
                obs2 = self.env.original_obstacles[indices[j]]
                
                # Distance between obstacle edges
                center_dist = np.linalg.norm(np.array(obs1['center']) - 
                                            np.array(obs2['center']))
                gap = center_dist - obs1['radius'] - obs2['radius']
                
                min_gap = min(min_gap, gap)
        
        return max(0, min_gap)
    
    def _distance_from_boundary(self, point: Tuple[float, float]) -> float:
        """Calculate minimum distance to environment boundary."""
        x, y = point
        
        distances = [
            x,  # Distance to left boundary
            self.env.width - x,  # Distance to right boundary
            y,  # Distance to bottom boundary
            self.env.height - y  # Distance to top boundary
        ]
        
        return min(distances)
    
    def _quadrant_obstacle_count(self, point: Tuple[float, float]) -> List[float]:
        """Count obstacles in four quadrants around point."""
        if not self.obstacle_centers.any():
            return [0.0, 0.0, 0.0, 0.0]
        
        x, y = point
        counts = [0, 0, 0, 0]  # NE, NW, SW, SE
        
        for obs_center in self.obstacle_centers:
            ox, oy = obs_center
            
            if ox >= x and oy >= y:
                counts[0] += 1  # NE
            elif ox < x and oy >= y:
                counts[1] += 1  # NW
            elif ox < x and oy < y:
                counts[2] += 1  # SW
            else:
                counts[3] += 1  # SE
        
        # Normalize
        total = sum(counts)
        if total > 0:
            counts = [c / total for c in counts]
        
        return counts
    
    def _obstacle_asymmetry(self, point: Tuple[float, float]) -> float:
        """Measure asymmetry of obstacle distribution around point."""
        if not self.obstacle_centers.any():
            return 0.0
        
        # Calculate center of mass of obstacles relative to point
        relative_positions = self.obstacle_centers - np.array(point)
        
        if len(relative_positions) == 0:
            return 0.0
        
        com = np.mean(relative_positions, axis=0)
        
        # Asymmetry is distance of COM from origin (normalized)
        asymmetry = np.linalg.norm(com) / max(self.env.width, self.env.height)
        
        return min(1.0, asymmetry)