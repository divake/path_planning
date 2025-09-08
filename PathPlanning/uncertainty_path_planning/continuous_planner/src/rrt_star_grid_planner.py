#!/usr/bin/env python3
"""
RRT* planner that works directly with occupancy grids (for MRPB maps)
"""

import numpy as np
import random
from typing import List, Tuple, Optional
import math


class RRTStarGrid:
    """
    RRT* planner for occupancy grid maps
    """
    
    def __init__(self, 
                 start: Tuple[float, float],
                 goal: Tuple[float, float],
                 occupancy_grid: np.ndarray,
                 origin: Tuple[float, float, float],
                 resolution: float,
                 robot_radius: float = 0.17,
                 step_size: float = 2.0,
                 max_iter: int = 3000,
                 goal_threshold: float = 1.0,
                 search_radius: float = 5.0,
                 seed: Optional[int] = None):
        """
        Initialize RRT* planner for grid maps
        
        Args:
            start: Start position (x, y) in meters
            goal: Goal position (x, y) in meters
            occupancy_grid: 2D numpy array (0=free, 100=occupied)
            origin: Map origin [x, y, theta]
            resolution: Meters per pixel
            robot_radius: Robot radius in meters
            step_size: Maximum step size for extending tree
            max_iter: Maximum iterations
            goal_threshold: Distance threshold to consider goal reached
            search_radius: Radius for rewiring
            seed: Random seed for reproducibility
        """
        self.start = start
        self.goal = goal
        self.occupancy_grid = occupancy_grid
        self.origin = origin[:2]  # Just x, y
        self.resolution = resolution
        self.robot_radius = robot_radius
        self.step_size = step_size
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold
        self.search_radius = search_radius
        
        # Calculate bounds from grid
        self.height_pixels, self.width_pixels = occupancy_grid.shape
        self.width_meters = self.width_pixels * resolution
        self.height_meters = self.height_pixels * resolution
        
        self.x_min = self.origin[0]
        self.x_max = self.origin[0] + self.width_meters
        self.y_min = self.origin[1]
        self.y_max = self.origin[1] + self.height_meters
        
        # Robot radius in pixels for collision checking
        self.robot_radius_pixels = int(np.ceil(robot_radius / resolution))
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        # RRT* tree
        self.nodes = [start]
        self.parents = {0: None}
        self.costs = {0: 0.0}
        
    def world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to grid indices"""
        grid_x = int((x - self.origin[0]) / self.resolution)
        grid_y = int((y - self.origin[1]) / self.resolution)
        return grid_x, grid_y
    
    def is_collision_free(self, x: float, y: float) -> bool:
        """Check if a point is collision-free considering robot radius"""
        grid_x, grid_y = self.world_to_grid(x, y)
        
        # Check bounds
        if (grid_x - self.robot_radius_pixels < 0 or 
            grid_x + self.robot_radius_pixels >= self.width_pixels or
            grid_y - self.robot_radius_pixels < 0 or 
            grid_y + self.robot_radius_pixels >= self.height_pixels):
            return False
        
        # Check circular area around robot position
        for dx in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
            for dy in range(-self.robot_radius_pixels, self.robot_radius_pixels + 1):
                # Check if within robot radius
                if dx*dx + dy*dy <= self.robot_radius_pixels * self.robot_radius_pixels:
                    check_x = grid_x + dx
                    check_y = grid_y + dy
                    
                    # Check occupancy
                    if self.occupancy_grid[check_y, check_x] > 50:  # Occupied or unknown
                        return False
        
        return True
    
    def is_path_collision_free(self, p1: Tuple[float, float], 
                               p2: Tuple[float, float]) -> bool:
        """Check if path between two points is collision-free"""
        dist = self.distance(p1, p2)
        steps = int(dist / (self.resolution * 2)) + 1  # Check every 2 pixels
        
        for i in range(steps + 1):
            t = i / float(steps)
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            
            if not self.is_collision_free(x, y):
                return False
        
        return True
    
    def distance(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        """Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def sample_random(self) -> Tuple[float, float]:
        """Sample random point in free space"""
        # 10% chance to sample goal
        if random.random() < 0.1:
            return self.goal
        
        # Sample until we find a collision-free point
        for _ in range(100):
            x = random.uniform(self.x_min + self.robot_radius, 
                             self.x_max - self.robot_radius)
            y = random.uniform(self.y_min + self.robot_radius, 
                             self.y_max - self.robot_radius)
            
            if self.is_collision_free(x, y):
                return (x, y)
        
        # If we can't find a free point, return goal
        return self.goal
    
    def nearest_node(self, point: Tuple[float, float]) -> int:
        """Find nearest node in tree to given point"""
        min_dist = float('inf')
        nearest_idx = 0
        
        for i, node in enumerate(self.nodes):
            dist = self.distance(node, point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx
    
    def steer(self, from_point: Tuple[float, float], 
             to_point: Tuple[float, float]) -> Tuple[float, float]:
        """Steer from one point toward another with step size limit"""
        dist = self.distance(from_point, to_point)
        
        if dist <= self.step_size:
            return to_point
        
        # Move step_size distance toward to_point
        ratio = self.step_size / dist
        x = from_point[0] + ratio * (to_point[0] - from_point[0])
        y = from_point[1] + ratio * (to_point[1] - from_point[1])
        
        return (x, y)
    
    def near_nodes(self, point: Tuple[float, float]) -> List[int]:
        """Find nodes within search radius"""
        near = []
        for i, node in enumerate(self.nodes):
            if self.distance(node, point) <= self.search_radius:
                near.append(i)
        return near
    
    def plan(self) -> Optional[List[Tuple[float, float]]]:
        """
        Execute RRT* planning
        
        Returns:
            Path from start to goal if found, None otherwise
        """
        # Check if start and goal are valid
        if not self.is_collision_free(self.start[0], self.start[1]):
            print(f"  ERROR: Start position {self.start} is in collision!")
            return None
        
        if not self.is_collision_free(self.goal[0], self.goal[1]):
            print(f"  ERROR: Goal position {self.goal} is in collision!")
            return None
        
        goal_reached = False
        goal_node_idx = None
        
        for iteration in range(self.max_iter):
            # Sample random point
            rand_point = self.sample_random()
            
            # Find nearest node
            nearest_idx = self.nearest_node(rand_point)
            nearest_point = self.nodes[nearest_idx]
            
            # Steer toward random point
            new_point = self.steer(nearest_point, rand_point)
            
            # Check collision
            if not self.is_collision_free(new_point[0], new_point[1]):
                continue
            
            if not self.is_path_collision_free(nearest_point, new_point):
                continue
            
            # Find near nodes for rewiring
            near_indices = self.near_nodes(new_point)
            
            # Choose best parent
            min_cost = self.costs[nearest_idx] + self.distance(nearest_point, new_point)
            best_parent_idx = nearest_idx
            
            for near_idx in near_indices:
                near_node = self.nodes[near_idx]
                if self.is_path_collision_free(near_node, new_point):
                    cost = self.costs[near_idx] + self.distance(near_node, new_point)
                    if cost < min_cost:
                        min_cost = cost
                        best_parent_idx = near_idx
            
            # Add new node
            new_idx = len(self.nodes)
            self.nodes.append(new_point)
            self.parents[new_idx] = best_parent_idx
            self.costs[new_idx] = min_cost
            
            # Rewire tree
            for near_idx in near_indices:
                near_node = self.nodes[near_idx]
                new_cost = min_cost + self.distance(new_point, near_node)
                
                if (new_cost < self.costs[near_idx] and 
                    self.is_path_collision_free(new_point, near_node)):
                    self.parents[near_idx] = new_idx
                    self.costs[near_idx] = new_cost
            
            # Check if goal reached
            if self.distance(new_point, self.goal) <= self.goal_threshold:
                goal_reached = True
                goal_node_idx = new_idx
                
                # Try to connect directly to goal
                if self.is_path_collision_free(new_point, self.goal):
                    final_idx = len(self.nodes)
                    self.nodes.append(self.goal)
                    self.parents[final_idx] = new_idx
                    self.costs[final_idx] = min_cost + self.distance(new_point, self.goal)
                    goal_node_idx = final_idx
                
                print(f"  Goal reached at iteration {iteration}")
                break
            
            # Progress update
            if iteration % 500 == 0:
                print(f"  Iteration {iteration}/{self.max_iter}, nodes: {len(self.nodes)}")
        
        if not goal_reached:
            print(f"  Failed to reach goal after {self.max_iter} iterations")
            return None
        
        # Extract path
        path = []
        current_idx = goal_node_idx
        
        while current_idx is not None:
            path.append(self.nodes[current_idx])
            current_idx = self.parents[current_idx]
        
        path.reverse()
        return path
    
    def compute_path_length(self, path: List[Tuple[float, float]]) -> float:
        """Compute total path length"""
        if not path or len(path) < 2:
            return 0.0
        
        length = 0.0
        for i in range(len(path) - 1):
            length += self.distance(path[i], path[i + 1])
        
        return length