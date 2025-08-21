"""
Ensemble Planner - Uses multiple models with different noise realizations
Provides uncertainty through ensemble disagreement
"""

import numpy as np
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from HybridAstarPlanner.hybrid_astar import hybrid_astar_planning, C
from icra_implementation.collision_checker import CollisionChecker

class EnsemblePlanner:
    def __init__(self, n_models=5, memory_system=None):
        self.n_models = n_models
        self.memory = memory_system
        self.collision_checker = CollisionChecker()
        self.config = C()
        
    def plan(self, start, goal, obstacles, noise_level=0.3):
        """
        Plan using ensemble of models with different noise realizations
        
        Args:
            start: [x, y, yaw] start position
            goal: [x, y, yaw] goal position
            obstacles: List of obstacles [x, y, radius]
            noise_level: Noise standard deviation for ensemble diversity
            
        Returns:
            path: Best path from ensemble
            metrics: Performance metrics including uncertainty
        """
        start_time = time.time()
        paths = []
        
        # Generate multiple paths with different noise realizations
        for model_idx in range(self.n_models):
            # Add noise to obstacle positions to simulate uncertainty
            noisy_obstacles = []
            for obs in obstacles:
                noise_x = np.random.normal(0, noise_level)
                noise_y = np.random.normal(0, noise_level)
                noise_r = np.random.normal(0, noise_level * 0.3)  # Less noise on radius
                
                noisy_obs = [
                    obs[0] + noise_x,
                    obs[1] + noise_y,
                    max(0.1, obs[2] + noise_r)  # Ensure positive radius
                ]
                noisy_obstacles.append(noisy_obs)
            
            # Convert to Hybrid A* format
            ox = []
            oy = []
            for obs in noisy_obstacles:
                theta_range = np.linspace(0, 2*np.pi, 20)
                for theta in theta_range:
                    ox.append(obs[0] + obs[2] * np.cos(theta))
                    oy.append(obs[1] + obs[2] * np.sin(theta))
            
            # Run planning
            try:
                path = hybrid_astar_planning(
                    start[0], start[1], start[2],
                    goal[0], goal[1], goal[2],
                    ox, oy, self.config.XY_RESO, self.config.YAW_RESO
                )
                
                if path:
                    paths.append(path)
                    
            except Exception as e:
                if self.memory:
                    self.memory.log_progress("ENSEMBLE_PLANNER", "WARNING", 
                                           f"Model {model_idx} failed: {str(e)}")
        
        planning_time = time.time() - start_time
        
        if not paths:
            if self.memory:
                self.memory.log_progress("ENSEMBLE_PLANNER", "FAILED", "No paths found by ensemble")
            
            return None, {
                'success': False,
                'collision_count': 0,
                'collision_rate': 1.0,
                'path_length': float('inf'),
                'planning_time': planning_time,
                'min_clearance': 0,
                'avg_clearance': 0,
                'path_smoothness': 0,
                'uncertainty_mean': 0,
                'uncertainty_std': 0
            }
        
        # Calculate uncertainty from ensemble disagreement
        uncertainties = self.calculate_ensemble_uncertainty(paths)
        
        # Create safety margins based on uncertainty
        safety_factor = 2.0  # 2-sigma confidence
        inflated_obstacles = self.inflate_obstacles(obstacles, 
                                                   safety_factor * np.mean(uncertainties))
        
        # Replan with inflated obstacles for safety
        ox_safe = []
        oy_safe = []
        for obs in inflated_obstacles:
            theta_range = np.linspace(0, 2*np.pi, 30)
            for theta in theta_range:
                ox_safe.append(obs[0] + obs[2] * np.cos(theta))
                oy_safe.append(obs[1] + obs[2] * np.sin(theta))
        
        try:
            safe_path = hybrid_astar_planning(
                start[0], start[1], start[2],
                goal[0], goal[1], goal[2],
                ox_safe, oy_safe, self.config.XY_RESO, self.config.YAW_RESO
            )
            
            if safe_path:
                # Calculate metrics
                collisions = self.collision_checker.check_collision(
                    [(p.x, p.y, p.yaw) for p in safe_path.x_list],
                    obstacles
                )
                
                path_length = self.calculate_path_length(safe_path.x_list)
                clearances = self.collision_checker.calculate_clearance(
                    [(p.x, p.y) for p in safe_path.x_list],
                    obstacles
                )
                
                metrics = {
                    'success': len(collisions) == 0,
                    'collision_count': len(collisions),
                    'collision_rate': len(collisions) / len(safe_path.x_list) if safe_path.x_list else 1.0,
                    'path_length': path_length,
                    'planning_time': planning_time,
                    'min_clearance': min(clearances) if clearances else 0,
                    'avg_clearance': np.mean(clearances) if clearances else 0,
                    'path_smoothness': self.calculate_smoothness(safe_path.x_list),
                    'uncertainty_mean': np.mean(uncertainties),
                    'uncertainty_std': np.std(uncertainties),
                    'ensemble_size': len(paths)
                }
                
                if self.memory:
                    self.memory.log_progress("ENSEMBLE_PLANNER", "SUCCESS",
                                           f"Safe path found: length={path_length:.2f}m, "
                                           f"uncertainty={np.mean(uncertainties):.3f}")
                
                return safe_path, metrics
                
        except Exception as e:
            if self.memory:
                self.memory.log_progress("ENSEMBLE_PLANNER", "ERROR", f"Safe planning failed: {str(e)}")
        
        # Fallback to best path from ensemble
        best_path = paths[0]
        for path in paths[1:]:
            if self.calculate_path_length(path.x_list) < self.calculate_path_length(best_path.x_list):
                best_path = path
                
        collisions = self.collision_checker.check_collision(
            [(p.x, p.y, p.yaw) for p in best_path.x_list],
            obstacles
        )
        
        path_length = self.calculate_path_length(best_path.x_list)
        clearances = self.collision_checker.calculate_clearance(
            [(p.x, p.y) for p in best_path.x_list],
            obstacles
        )
        
        return best_path, {
            'success': len(collisions) == 0,
            'collision_count': len(collisions),
            'collision_rate': len(collisions) / len(best_path.x_list) if best_path.x_list else 1.0,
            'path_length': path_length,
            'planning_time': planning_time,
            'min_clearance': min(clearances) if clearances else 0,
            'avg_clearance': np.mean(clearances) if clearances else 0,
            'path_smoothness': self.calculate_smoothness(best_path.x_list),
            'uncertainty_mean': np.mean(uncertainties),
            'uncertainty_std': np.std(uncertainties),
            'ensemble_size': len(paths)
        }
    
    def calculate_ensemble_uncertainty(self, paths):
        """Calculate uncertainty as ensemble disagreement"""
        if len(paths) < 2:
            return [0.1]  # Default small uncertainty
        
        # Find minimum path length for comparison
        min_len = min(len(path.x_list) for path in paths)
        
        uncertainties = []
        for i in range(min_len):
            # Get positions at this point from all paths
            positions = [(path.x_list[i].x, path.x_list[i].y) for path in paths]
            
            # Calculate variance
            positions_array = np.array(positions)
            variance = np.var(positions_array, axis=0)
            uncertainty = np.sqrt(np.sum(variance))
            uncertainties.append(uncertainty)
            
        return uncertainties
    
    def inflate_obstacles(self, obstacles, inflation_amount):
        """Inflate obstacles for conservative planning"""
        inflated = []
        for obs in obstacles:
            inflated.append([
                obs[0],
                obs[1],
                obs[2] + inflation_amount
            ])
        return inflated
    
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
        """Calculate path smoothness"""
        if not path or len(path) < 3:
            return 0
            
        curvatures = []
        for i in range(1, len(path) - 1):
            p1 = (path[i-1].x, path[i-1].y)
            p2 = (path[i].x, path[i].y)
            p3 = (path[i+1].x, path[i+1].y)
            
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            angle_change = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])
            angle_change = np.abs(np.arctan2(np.sin(angle_change), np.cos(angle_change)))
            
            dist = np.sqrt(v1[0]**2 + v1[1]**2)
            
            if dist > 0:
                curvature = angle_change / dist
                curvatures.append(curvature)
                
        return np.mean(curvatures) if curvatures else 0