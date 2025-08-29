"""
Wrapper to integrate python_motion_planning planners with our environment.
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Optional

# Add python_motion_planning to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

from python_motion_planning.utils import Grid
from python_motion_planning.global_planner.graph_search import AStar as AStarBase


class PlannerWrapper:
    """Wrapper to use python_motion_planning planners with our environment."""
    
    def __init__(self, environment):
        """
        Initialize planner wrapper.
        
        Args:
            environment: UncertaintyEnvironment instance
        """
        self.env = environment
        
    def plan_astar(self, start: Tuple[int, int], goal: Tuple[int, int],
                   use_inflated: bool = True) -> Tuple[List[Tuple[int, int]], float]:
        """
        Plan using A* algorithm.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            use_inflated: If True, use inflated obstacles
            
        Returns:
            Tuple of (path, cost) where path is list of (x, y) points
        """
        # Get occupancy grid
        occ_grid = self.env.get_occupancy_grid(use_inflated=use_inflated)
        
        # Create Grid environment for planner
        grid_env = Grid(self.env.width, self.env.height)
        
        # Set obstacles in Grid environment
        grid_env.obs = set()
        for i in range(self.env.width):
            for j in range(self.env.height):
                if occ_grid[j, i] > 0:  # Note: occupancy grid is (y, x)
                    grid_env.obs.add((i, j))
        
        # Initialize grid (adds boundary obstacles)
        grid_env.init()
        
        try:
            # Create planner
            planner = AStarBase(start, goal, grid_env)
            
            # Plan
            cost, path, _ = planner.plan()
            
            return path, cost
            
        except Exception as e:
            print(f"Planning failed: {e}")
            return [], float('inf')
    
    def check_path_validity(self, path: List[Tuple[int, int]], 
                           robot_radius: float = 0.5) -> bool:
        """
        Check if path is valid (collision-free).
        
        Args:
            path: List of (x, y) points
            robot_radius: Robot radius for collision checking
            
        Returns:
            True if path is valid
        """
        if not path:
            return False
            
        return not self.env.check_collision(path, use_original=True, 
                                           robot_radius=robot_radius)