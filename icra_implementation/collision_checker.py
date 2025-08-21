import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class CollisionChecker:
    """Analytical collision checking for path planning"""
    
    def __init__(self, vehicle_width=2.0, vehicle_length=4.5):
        self.vehicle_width = vehicle_width
        self.vehicle_length = vehicle_length
        self.memory = None
        
    def set_memory_system(self, memory):
        self.memory = memory
        
    def check_collision(self, path, obstacles, safety_margin=0.0):
        """
        Check if path collides with obstacles
        Returns collision details
        """
        collisions = []
        
        for i, state in enumerate(path):
            if len(state) >= 3:
                x, y, theta = state[0], state[1], state[2]
            else:
                x, y = state[0], state[1]
                theta = 0
                
            # Get vehicle corners
            corners = self.get_vehicle_corners(x, y, theta)
            
            for obs_idx, obs in enumerate(obstacles):
                # Check if any corner is inside obstacle
                for corner in corners:
                    if hasattr(obs, 'x') and hasattr(obs, 'y') and hasattr(obs, 'radius'):
                        distance = np.sqrt((corner[0] - obs.x)**2 + (corner[1] - obs.y)**2)
                        if distance < (obs.radius + safety_margin):
                            collisions.append({
                                'path_index': i,
                                'obstacle_index': obs_idx,
                                'distance': distance,
                                'position': (x, y),
                                'obstacle_center': (obs.x, obs.y),
                                'obstacle_radius': obs.radius
                            })
                    elif isinstance(obs, (list, tuple)) and len(obs) >= 3:
                        # Handle different obstacle format (x, y, radius)
                        ox, oy, oradius = obs[0], obs[1], obs[2] if len(obs) > 2 else 1.0
                        distance = np.sqrt((corner[0] - ox)**2 + (corner[1] - oy)**2)
                        if distance < (oradius + safety_margin):
                            collisions.append({
                                'path_index': i,
                                'obstacle_index': obs_idx,
                                'distance': distance,
                                'position': (x, y),
                                'obstacle_center': (ox, oy),
                                'obstacle_radius': oradius
                            })
                            
        return collisions
    
    def get_vehicle_corners(self, x, y, theta):
        """Get four corners of the vehicle rectangle"""
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Half dimensions
        half_length = self.vehicle_length / 2
        half_width = self.vehicle_width / 2
        
        # Corners in local frame
        corners_local = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width)
        ]
        
        # Transform to global frame
        corners_global = []
        for lx, ly in corners_local:
            gx = x + lx * cos_theta - ly * sin_theta
            gy = y + lx * sin_theta + ly * cos_theta
            corners_global.append((gx, gy))
            
        return corners_global
    
    def calculate_clearance(self, path, obstacles):
        """Calculate minimum clearance for each point in path"""
        clearances = []
        
        for state in path:
            if len(state) >= 2:
                x, y = state[0], state[1]
            else:
                continue
                
            min_clearance = float('inf')
            
            for obs in obstacles:
                if hasattr(obs, 'x') and hasattr(obs, 'y') and hasattr(obs, 'radius'):
                    distance = np.sqrt((x - obs.x)**2 + (y - obs.y)**2) - obs.radius
                elif isinstance(obs, (list, tuple)) and len(obs) >= 2:
                    ox, oy = obs[0], obs[1]
                    oradius = obs[2] if len(obs) > 2 else 1.0
                    distance = np.sqrt((x - ox)**2 + (y - oy)**2) - oradius
                else:
                    continue
                    
                min_clearance = min(min_clearance, distance)
                
            clearances.append(min_clearance)
            
        return clearances
    
    def visualize_collision_check(self, path, obstacles, collisions, save_path=None):
        """Visualize path with collision points highlighted"""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot obstacles
        for obs in obstacles:
            if hasattr(obs, 'x') and hasattr(obs, 'y') and hasattr(obs, 'radius'):
                circle = Circle((obs.x, obs.y), obs.radius, color='gray', alpha=0.5)
                ax.add_patch(circle)
            elif isinstance(obs, (list, tuple)) and len(obs) >= 2:
                ox, oy = obs[0], obs[1]
                oradius = obs[2] if len(obs) > 2 else 1.0
                circle = Circle((ox, oy), oradius, color='gray', alpha=0.5)
                ax.add_patch(circle)
        
        # Plot path
        if len(path) > 0 and len(path[0]) >= 2:
            path_x = [state[0] for state in path]
            path_y = [state[1] for state in path]
            ax.plot(path_x, path_y, 'b-', linewidth=2, label='Path')
        
        # Highlight collision points
        if collisions:
            collision_x = [c['position'][0] for c in collisions]
            collision_y = [c['position'][1] for c in collisions]
            ax.scatter(collision_x, collision_y, c='red', s=100, marker='x', 
                      linewidth=3, label=f'Collisions ({len(collisions)})')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_title('Collision Check Visualization')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            if self.memory:
                self.memory.log_progress('VISUALIZATION', 'SAVED', f'Collision check saved to {save_path}')
        
        plt.close()
        
        return fig