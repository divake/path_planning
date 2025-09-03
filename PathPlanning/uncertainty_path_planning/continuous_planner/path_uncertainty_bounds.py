#!/usr/bin/env python3
"""
Path uncertainty bounds visualization - showing upper/lower quantiles
Similar to ICRA 2024 Visual Odometry paper with confidence bands
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from typing import List, Tuple, Optional
import sys
sys.path.append('continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar

class PathUncertaintyBounds:
    """Visualize paths with upper/lower uncertainty bounds"""
    
    def __init__(self):
        self.colors = {
            'path': '#2ECC71',      # Green for main path
            'upper': '#E74C3C',     # Red for upper bound
            'lower': '#3498DB',     # Blue for lower bound
            'fill': '#95A5A6',      # Gray for uncertainty band
            'obstacles': '#34495E',  # Dark gray for obstacles
            'perceived': '#F39C12',  # Orange for perceived
            'cp_inflated': '#9B59B6' # Purple for CP inflated
        }
    
    def compute_path_bounds(self, path, tau, confidence=0.95):
        """
        Compute upper and lower bounds for a path based on tau
        
        Returns:
            upper_path: Path shifted by +tau perpendicular to direction
            lower_path: Path shifted by -tau perpendicular to direction
        """
        path_array = np.array(path)
        n_points = len(path)
        
        upper_path = []
        lower_path = []
        
        for i in range(n_points):
            # Get current point
            current = path_array[i]
            
            # Calculate direction vector
            if i == 0:
                # First point: use direction to next
                direction = path_array[i+1] - path_array[i]
            elif i == n_points - 1:
                # Last point: use direction from previous
                direction = path_array[i] - path_array[i-1]
            else:
                # Middle points: average of forward and backward directions
                direction = path_array[i+1] - path_array[i-1]
            
            # Normalize direction
            if np.linalg.norm(direction) > 0:
                direction = direction / np.linalg.norm(direction)
            else:
                direction = np.array([1, 0])
            
            # Get perpendicular vector (rotate 90 degrees)
            perpendicular = np.array([-direction[1], direction[0]])
            
            # Create upper and lower bounds
            # Scale tau based on confidence level for visualization
            bound_width = tau * (1 + (1 - confidence) * 2)  # Wider bounds for lower confidence
            
            upper_point = current + perpendicular * bound_width
            lower_point = current - perpendicular * bound_width
            
            upper_path.append(upper_point)
            lower_path.append(lower_point)
        
        return np.array(upper_path), np.array(lower_path)
    
    def create_path_with_bounds_visualization(self):
        """
        Create main visualization showing path with uncertainty bounds
        Similar to ICRA 2024 paper style
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Different scenarios
        scenarios = [
            ('Low Noise (0.1)', 0.1, 0.95),
            ('Medium Noise (0.2)', 0.2, 0.95),
            ('High Noise (0.3)', 0.3, 0.95),
            ('90% Confidence', 0.2, 0.90),
            ('95% Confidence', 0.2, 0.95),
            ('99% Confidence', 0.2, 0.99)
        ]
        
        for idx, (title, noise_level, confidence) in enumerate(scenarios):
            ax = axes[idx // 3, idx % 3]
            
            # Create environment and plan path
            env = ContinuousEnvironment(env_type='passages')
            
            # Calibrate CP for this scenario
            cp = ContinuousStandardCP(env.obstacles, "penetration")
            tau = cp.calibrate(
                ContinuousNoiseModel.add_thinning_noise,
                {'thin_factor': noise_level},
                num_samples=200,
                confidence=confidence
            )
            
            # Get perceived obstacles and plan path
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=noise_level, seed=42
            )
            inflated = cp.inflate_obstacles(perceived)
            
            planner = RRTStar((5, 15), (45, 15), inflated, max_iter=1000)
            path = planner.plan()
            
            if path:
                # Compute uncertainty bounds
                upper_path, lower_path = self.compute_path_bounds(path, tau, confidence)
                
                # Plot obstacles (faded)
                for obs in env.obstacles:
                    rect = Rectangle((obs[0], obs[1]), obs[2], obs[3],
                                   facecolor=self.colors['obstacles'], 
                                   alpha=0.2, edgecolor='none')
                    ax.add_patch(rect)
                
                # Plot uncertainty band (filled region between bounds)
                # Create polygon from upper and lower paths
                polygon_points = np.vstack([upper_path, lower_path[::-1]])
                polygon = Polygon(polygon_points, facecolor=self.colors['fill'], 
                                alpha=0.2, edgecolor='none', label='Uncertainty Band')
                ax.add_patch(polygon)
                
                # Plot the three trajectories
                path_array = np.array(path)
                
                # Upper bound (dashed red)
                ax.plot(upper_path[:, 0], upper_path[:, 1], '--', 
                       color=self.colors['upper'], linewidth=2, 
                       label=f'Upper Bound (+{tau:.3f})', alpha=0.8)
                
                # Main path (solid green, thicker)
                ax.plot(path_array[:, 0], path_array[:, 1], '-', 
                       color=self.colors['path'], linewidth=3, 
                       label='Planned Path', zorder=10)
                
                # Lower bound (dashed blue)
                ax.plot(lower_path[:, 0], lower_path[:, 1], '--', 
                       color=self.colors['lower'], linewidth=2, 
                       label=f'Lower Bound (-{tau:.3f})', alpha=0.8)
                
                # Add markers at regular intervals to show correspondence
                step = max(1, len(path) // 10)
                for i in range(0, len(path), step):
                    # Vertical lines connecting bounds
                    ax.plot([upper_path[i, 0], lower_path[i, 0]], 
                           [upper_path[i, 1], lower_path[i, 1]], 
                           'k-', alpha=0.1, linewidth=0.5)
                    
                    # Points on each trajectory
                    ax.plot(upper_path[i, 0], upper_path[i, 1], 'o', 
                           color=self.colors['upper'], markersize=3)
                    ax.plot(path_array[i, 0], path_array[i, 1], 'o', 
                           color=self.colors['path'], markersize=4)
                    ax.plot(lower_path[i, 0], lower_path[i, 1], 'o', 
                           color=self.colors['lower'], markersize=3)
            
            # Start and goal
            ax.plot(5, 15, 'o', color='green', markersize=12, 
                   label='Start', zorder=20, markeredgecolor='white', markeredgewidth=2)
            ax.plot(45, 15, '*', color='red', markersize=15, 
                   label='Goal', zorder=20, markeredgecolor='white', markeredgewidth=2)
            
            # Formatting
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 30)
            ax.set_aspect('equal')
            ax.set_title(f'{title}\nτ = {tau:.3f}', fontsize=12, weight='bold')
            ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
            ax.grid(True, alpha=0.2, linestyle=':')
            
            # Add margin annotation
            ax.text(0.98, 0.02, f'Avg. interval\nlength: {2*tau:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   ha='right', va='bottom',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                   color='red', weight='bold')
        
        plt.suptitle('Path Planning with Uncertainty Bounds - Conformal Prediction\n' +
                    '(Similar to ICRA 2024 Visual Odometry)', 
                    fontsize=16, weight='bold', y=1.02)
        plt.tight_layout()
        
        return fig
    
    def create_3d_path_bounds_visualization(self):
        """
        Create 3D visualization with path bounds over time
        """
        fig = plt.figure(figsize=(16, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Setup environment and path
        env = ContinuousEnvironment(env_type='passages')
        
        cp = ContinuousStandardCP(env.obstacles, "penetration")
        tau = cp.calibrate(
            ContinuousNoiseModel.add_thinning_noise,
            {'thin_factor': 0.2},
            num_samples=200,
            confidence=0.95
        )
        
        perceived = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles, thin_factor=0.2, seed=42
        )
        inflated = cp.inflate_obstacles(perceived)
        
        planner = RRTStar((5, 15), (45, 15), inflated, max_iter=1000)
        path = planner.plan()
        
        if path:
            # Compute bounds
            upper_path, lower_path = self.compute_path_bounds(path, tau)
            path_array = np.array(path)
            
            # Create time axis
            t = np.linspace(0, 10, len(path))
            
            # Plot main trajectory with bounds
            ax.plot(path_array[:, 0], path_array[:, 1], t, 
                   color=self.colors['path'], linewidth=4, 
                   label='Predicted Path', zorder=10)
            
            ax.plot(upper_path[:, 0], upper_path[:, 1], t, 
                   color=self.colors['upper'], linewidth=2, linestyle='--',
                   label=f'Upper Bound (τ={tau:.3f})', alpha=0.7)
            
            ax.plot(lower_path[:, 0], lower_path[:, 1], t, 
                   color=self.colors['lower'], linewidth=2, linestyle='--',
                   label=f'Lower Bound (τ={tau:.3f})', alpha=0.7)
            
            # Create surface between bounds for better visualization
            for i in range(len(path) - 1):
                # Create quad surface between bounds
                x_quad = [upper_path[i, 0], upper_path[i+1, 0], 
                         lower_path[i+1, 0], lower_path[i, 0]]
                y_quad = [upper_path[i, 1], upper_path[i+1, 1], 
                         lower_path[i+1, 1], lower_path[i, 1]]
                z_quad = [t[i], t[i+1], t[i+1], t[i]]
                
                # Create vertices for polygon
                from mpl_toolkits.mplot3d.art3d import Poly3DCollection
                verts = [list(zip(x_quad, y_quad, z_quad))]
                poly = Poly3DCollection(verts, alpha=0.1, 
                                       facecolor=self.colors['fill'],
                                       edgecolor='none')
                ax.add_collection3d(poly)
            
            # Add markers
            ax.scatter([path_array[0, 0]], [path_array[0, 1]], [t[0]], 
                      color='green', s=100, marker='o', label='Start', zorder=15)
            ax.scatter([path_array[-1, 0]], [path_array[-1, 1]], [t[-1]], 
                      color='red', s=100, marker='*', label='Goal', zorder=15)
        
        # Labels and formatting
        ax.set_xlabel('X coordinate', fontsize=12, labelpad=10)
        ax.set_ylabel('Y coordinate', fontsize=12, labelpad=10)
        ax.set_zlabel('Time', fontsize=12, labelpad=10)
        
        ax.set_title('3D Path with Uncertainty Bounds Over Time\n' +
                    f'Average Interval Length: {2*tau:.3f}',
                    fontsize=14, weight='bold', pad=20)
        
        # Add text annotation similar to ICRA 2024
        ax.text2D(0.05, 0.95, f'Avg. interval length: {2*tau:.3f}',
                 transform=ax.transAxes, fontsize=14, color='red',
                 weight='bold', verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.legend(loc='upper right', fontsize=10)
        ax.view_init(elev=25, azim=45)
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    def create_comparative_bounds_visualization(self):
        """
        Compare Naive vs CP approach with uncertainty bounds
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        env = ContinuousEnvironment(env_type='passages')
        noise_level = 0.2
        
        for idx, (method, use_cp) in enumerate([('Naive', False), ('CP', True)]):
            ax = axes[idx]
            
            # Get perceived obstacles
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=noise_level, seed=42
            )
            
            if use_cp:
                cp = ContinuousStandardCP(env.obstacles, "penetration")
                tau = cp.calibrate(
                    ContinuousNoiseModel.add_thinning_noise,
                    {'thin_factor': noise_level},
                    num_samples=200,
                    confidence=0.95
                )
                obstacles_to_use = cp.inflate_obstacles(perceived)
            else:
                tau = 0  # No safety margin for naive
                obstacles_to_use = perceived
            
            # Plan path
            planner = RRTStar((5, 15), (45, 15), obstacles_to_use, max_iter=1000)
            path = planner.plan()
            
            # Plot true obstacles (very faded)
            for obs in env.obstacles:
                rect = Rectangle((obs[0], obs[1]), obs[2], obs[3],
                               facecolor='gray', alpha=0.1, 
                               edgecolor='black', linewidth=0.5, linestyle=':')
                ax.add_patch(rect)
            
            # Plot perceived/inflated obstacles
            for obs in obstacles_to_use:
                color = self.colors['cp_inflated'] if use_cp else self.colors['perceived']
                rect = Rectangle((obs[0], obs[1]), obs[2], obs[3],
                               facecolor=color, alpha=0.3,
                               edgecolor='black', linewidth=1)
                ax.add_patch(rect)
            
            if path:
                path_array = np.array(path)
                
                if use_cp:
                    # Show uncertainty bounds for CP
                    upper_path, lower_path = self.compute_path_bounds(path, tau)
                    
                    # Plot uncertainty band
                    polygon_points = np.vstack([upper_path, lower_path[::-1]])
                    polygon = Polygon(polygon_points, 
                                    facecolor=self.colors['fill'], 
                                    alpha=0.2, edgecolor='none')
                    ax.add_patch(polygon)
                    
                    # Plot bounds
                    ax.plot(upper_path[:, 0], upper_path[:, 1], '--',
                           color=self.colors['upper'], linewidth=1.5, alpha=0.7)
                    ax.plot(lower_path[:, 0], lower_path[:, 1], '--',
                           color=self.colors['lower'], linewidth=1.5, alpha=0.7)
                
                # Check for collisions with true obstacles
                collisions = []
                for p in path:
                    if env.point_in_obstacle(p[0], p[1]):
                        collisions.append(p)
                
                # Color path based on safety
                if collisions:
                    path_color = '#E74C3C'  # Red for colliding path
                    path_label = f'{method} Path (COLLISION!)'
                else:
                    path_color = self.colors['path']
                    path_label = f'{method} Path (Safe)'
                
                # Plot main path
                ax.plot(path_array[:, 0], path_array[:, 1], '-',
                       color=path_color, linewidth=3,
                       label=path_label, zorder=10)
                
                # Mark collision points
                if collisions:
                    collision_array = np.array(collisions)
                    ax.scatter(collision_array[:, 0], collision_array[:, 1],
                             color='red', s=100, marker='x', 
                             label=f'Collisions ({len(collisions)})',
                             zorder=15, linewidths=3)
            
            # Start and goal
            ax.plot(5, 15, 'o', color='green', markersize=12,
                   label='Start', zorder=20, markeredgecolor='white', markeredgewidth=2)
            ax.plot(45, 15, '*', color='red', markersize=15,
                   label='Goal', zorder=20, markeredgecolor='white', markeredgewidth=2)
            
            # Formatting
            ax.set_xlim(0, 50)
            ax.set_ylim(0, 30)
            ax.set_aspect('equal')
            
            title = f'{method} Method'
            if use_cp:
                title += f'\nSafety Margin τ = {tau:.3f}'
            else:
                title += '\nNo Safety Margin'
            
            ax.set_title(title, fontsize=14, weight='bold')
            ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.2, linestyle=':')
            
            # Add safety annotation
            if use_cp:
                ax.text(0.98, 0.02, f'Interval: {2*tau:.3f}\nGuarantee: 95%',
                       transform=ax.transAxes, fontsize=11,
                       ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9),
                       weight='bold')
            else:
                ax.text(0.98, 0.02, 'No guarantee',
                       transform=ax.transAxes, fontsize=11,
                       ha='right', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.9),
                       weight='bold')
        
        plt.suptitle('Uncertainty Bounds: Naive vs Conformal Prediction',
                    fontsize=16, weight='bold')
        plt.tight_layout()
        
        return fig
    
    def create_time_series_bounds(self):
        """
        Create time-series plot showing how bounds evolve along the path
        """
        fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
        
        # Setup
        env = ContinuousEnvironment(env_type='narrow')
        
        cp = ContinuousStandardCP(env.obstacles, "penetration")
        tau = cp.calibrate(
            ContinuousNoiseModel.add_thinning_noise,
            {'thin_factor': 0.2},
            num_samples=200,
            confidence=0.95
        )
        
        perceived = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles, thin_factor=0.2, seed=42
        )
        inflated = cp.inflate_obstacles(perceived)
        
        planner = RRTStar((5, 15), (45, 15), inflated, max_iter=1000)
        path = planner.plan()
        
        if path:
            path_array = np.array(path)
            upper_path, lower_path = self.compute_path_bounds(path, tau)
            
            # Create path progress axis (0 to 1)
            progress = np.linspace(0, 1, len(path))
            
            # Plot X coordinates
            ax = axes[0]
            ax.fill_between(progress, upper_path[:, 0], lower_path[:, 0],
                          color=self.colors['fill'], alpha=0.3,
                          label='Uncertainty Band')
            ax.plot(progress, upper_path[:, 0], '--', 
                   color=self.colors['upper'], linewidth=1.5, alpha=0.7)
            ax.plot(progress, path_array[:, 0], '-', 
                   color=self.colors['path'], linewidth=3,
                   label='Predicted X')
            ax.plot(progress, lower_path[:, 0], '--', 
                   color=self.colors['lower'], linewidth=1.5, alpha=0.7)
            ax.set_ylabel('X Coordinate', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_title('Path with Uncertainty Bounds - X Component', fontsize=12, weight='bold')
            
            # Plot Y coordinates
            ax = axes[1]
            ax.fill_between(progress, upper_path[:, 1], lower_path[:, 1],
                          color=self.colors['fill'], alpha=0.3)
            ax.plot(progress, upper_path[:, 1], '--', 
                   color=self.colors['upper'], linewidth=1.5, alpha=0.7,
                   label='Upper Bound')
            ax.plot(progress, path_array[:, 1], '-', 
                   color=self.colors['path'], linewidth=3,
                   label='Predicted Y')
            ax.plot(progress, lower_path[:, 1], '--', 
                   color=self.colors['lower'], linewidth=1.5, alpha=0.7,
                   label='Lower Bound')
            ax.set_ylabel('Y Coordinate', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_title('Path with Uncertainty Bounds - Y Component', fontsize=12, weight='bold')
            
            # Plot interval width
            ax = axes[2]
            interval_width = np.sqrt((upper_path[:, 0] - lower_path[:, 0])**2 + 
                                    (upper_path[:, 1] - lower_path[:, 1])**2)
            ax.fill_between(progress, 0, interval_width,
                          color=self.colors['fill'], alpha=0.5)
            ax.plot(progress, interval_width, '-', 
                   color='purple', linewidth=2,
                   label='Interval Width')
            ax.axhline(y=2*tau, color='red', linestyle='--', 
                      linewidth=2, alpha=0.7,
                      label=f'Expected Width (2τ = {2*tau:.3f})')
            ax.set_ylabel('Interval Width', fontsize=12)
            ax.set_xlabel('Path Progress (0=Start, 1=Goal)', fontsize=12)
            ax.legend(loc='upper right', fontsize=10)
            ax.grid(True, alpha=0.2)
            ax.set_title('Uncertainty Interval Width Along Path', fontsize=12, weight='bold')
            
            # Add average interval annotation
            avg_width = np.mean(interval_width)
            ax.text(0.02, 0.95, f'Avg. interval: {avg_width:.3f}',
                   transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                   weight='bold', va='top')
        
        plt.suptitle('Time-Series Analysis of Path Uncertainty\n' +
                    '(Similar to ICRA 2024 Visual Odometry Bounds)',
                    fontsize=16, weight='bold', y=1.01)
        plt.tight_layout()
        
        return fig


def main():
    """Generate path uncertainty bounds visualizations"""
    
    print("\n" + "="*70)
    print("GENERATING PATH UNCERTAINTY BOUNDS VISUALIZATIONS")
    print("(ICRA 2024 Visual Odometry Style)")
    print("="*70)
    
    visualizer = PathUncertaintyBounds()
    
    # 1. Main visualization with bounds
    print("\n1. Creating path with uncertainty bounds...")
    fig = visualizer.create_path_with_bounds_visualization()
    fig.savefig('results/path_uncertainty_bounds.png', dpi=300, bbox_inches='tight')
    print("   Saved: path_uncertainty_bounds.png")
    
    # 2. 3D visualization
    print("\n2. Creating 3D path bounds visualization...")
    fig, ax = visualizer.create_3d_path_bounds_visualization()
    fig.savefig('results/path_bounds_3d.png', dpi=300, bbox_inches='tight')
    print("   Saved: path_bounds_3d.png")
    
    # 3. Comparative visualization
    print("\n3. Creating comparative bounds (Naive vs CP)...")
    fig = visualizer.create_comparative_bounds_visualization()
    fig.savefig('results/comparative_bounds.png', dpi=300, bbox_inches='tight')
    print("   Saved: comparative_bounds.png")
    
    # 4. Time-series bounds
    print("\n4. Creating time-series bounds visualization...")
    fig = visualizer.create_time_series_bounds()
    fig.savefig('results/time_series_bounds.png', dpi=300, bbox_inches='tight')
    print("   Saved: time_series_bounds.png")
    
    print("\n" + "="*70)
    print("PATH UNCERTAINTY BOUNDS VISUALIZATIONS COMPLETE")
    print("="*70)
    print("\nGenerated ICRA 2024-style visualizations:")
    print("  ✓ Path with upper/lower uncertainty bounds")
    print("  ✓ Three parallel trajectories (upper, predicted, lower)")
    print("  ✓ Filled uncertainty bands between bounds")
    print("  ✓ Time-series analysis of bounds")
    print("  ✓ Comparative visualization (Naive vs CP)")
    print("\nKey features shown:")
    print("  - Average interval length (similar to your ICRA 2024)")
    print("  - Confidence-based bound width")
    print("  - Adaptive bounds based on noise level")
    print("  - Clear safety guarantees visualization")


if __name__ == "__main__":
    main()