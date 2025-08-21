"""
Naive Planner - No uncertainty handling
Simply runs Hybrid A* with observed obstacles
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C
from icra_implementation.collision_checker import CollisionChecker

class NaivePlanner:
    def __init__(self, memory_system=None):
        self.memory = memory_system
        self.collision_checker = CollisionChecker()
        self.config = C()  # Use Hybrid A* configuration
        
    def plan(self, start, goal, obstacles):
        """
        Plan path using naive approach (no uncertainty)
        
        Args:
            start: [x, y, yaw] start position
            goal: [x, y, yaw] goal position  
            obstacles: List of obstacles [x, y, radius]
            
        Returns:
            path: List of states
            metrics: Dictionary of performance metrics
        """
        start_time = time.time()
        
        # Convert obstacles to Hybrid A* format
        ox = []
        oy = []
        for obs in obstacles:
            # Create circle of points for each obstacle
            theta_range = np.linspace(0, 2*np.pi, 20)
            for theta in theta_range:
                ox.append(obs[0] + obs[2] * np.cos(theta))
                oy.append(obs[1] + obs[2] * np.sin(theta))
        
        # Run Hybrid A* planning
        try:
            path = hybrid_astar_planning(
                start[0], start[1], start[2],
                goal[0], goal[1], goal[2],
                ox, oy, self.config.XY_RESO, self.config.YAW_RESO
            )
            
            planning_time = time.time() - start_time
            
            if path:
                # Calculate metrics
                collisions = self.collision_checker.check_collision(
                    [(p.x, p.y, p.yaw) for p in path.x_list],
                    obstacles
                )
                
                path_length = self.calculate_path_length(path.x_list)
                clearances = self.collision_checker.calculate_clearance(
                    [(p.x, p.y) for p in path.x_list],
                    obstacles
                )
                
                metrics = {
                    'success': len(collisions) == 0,
                    'collision_count': len(collisions),
                    'collision_rate': len(collisions) / len(path.x_list) if path.x_list else 1.0,
                    'path_length': path_length,
                    'planning_time': planning_time,
                    'min_clearance': min(clearances) if clearances else 0,
                    'avg_clearance': np.mean(clearances) if clearances else 0,
                    'path_smoothness': self.calculate_smoothness(path.x_list)
                }
                
                if self.memory:
                    self.memory.log_progress("NAIVE_PLANNER", "SUCCESS", 
                                           f"Path found: length={path_length:.2f}m, collisions={len(collisions)}")
                
                return path, metrics
            else:
                if self.memory:
                    self.memory.log_progress("NAIVE_PLANNER", "FAILED", "No path found")
                    
                return None, {
                    'success': False,
                    'collision_count': 0,
                    'collision_rate': 1.0,
                    'path_length': float('inf'),
                    'planning_time': planning_time,
                    'min_clearance': 0,
                    'avg_clearance': 0,
                    'path_smoothness': 0
                }
                
        except Exception as e:
            if self.memory:
                self.memory.log_progress("NAIVE_PLANNER", "ERROR", f"Planning failed: {str(e)}")
                
            return None, {
                'success': False,
                'collision_count': 0,
                'collision_rate': 1.0,
                'path_length': float('inf'),
                'planning_time': time.time() - start_time,
                'min_clearance': 0,
                'avg_clearance': 0,
                'path_smoothness': 0
            }
    
    def calculate_path_length(self, path):
        """Calculate total path length"""
        if not path or len(path) < 2:
            return 0
            
        length = 0
        for i in range(1, len(path)):
            dx = path[i].x - path[i-1].x
            dy = path[i].y - path[i-1].y
            length += np.sqrt(dx**2 + dy**2)
        return length
    
    def calculate_smoothness(self, path):
        """Calculate path smoothness (lower is smoother)"""
        if not path or len(path) < 3:
            return 0
            
        curvatures = []
        for i in range(1, len(path) - 1):
            # Calculate curvature using three points
            p1 = (path[i-1].x, path[i-1].y)
            p2 = (path[i].x, path[i].y)
            p3 = (path[i+1].x, path[i+1].y)
            
            # Vectors
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # Angle change
            angle_change = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle_change = np.abs(np.arctan2(np.sin(angle_change), np.cos(angle_change)))
            
            # Distance
            dist = np.sqrt(v1[0]**2 + v1[1]**2)
            
            if dist > 0:
                curvature = angle_change / dist
                curvatures.append(curvature)
                
        return np.mean(curvatures) if curvatures else 0