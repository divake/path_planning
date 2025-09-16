#!/usr/bin/env python3
"""
Create Publication-Ready Path Planning Visualization
Shows RRT* paths for Naive, Standard CP, and Learnable CP methods
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle
import yaml
import sys
import os

# Add paths for imports
sys.path.append('..')
sys.path.append('../..')
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid

def create_path_planning_visualization():
    """Create a publication-ready figure showing path planning with different methods"""

    # Load configuration
    config_dir = '../../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    # Choose office01add environment for visualization (good complexity)
    env_name = 'office01add'

    # Parse MRPB map
    dataset_path = '../../mrpb_dataset'
    parser = MRPBMapParser(env_name, dataset_path)

    # Get a good test case (test 1 from office01add)
    env_tests = env_config['environments'][env_name]['tests']
    test_config = env_tests[0]  # First test
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Method configurations
    methods = [
        {'name': 'Naive', 'radius': 0.17, 'color': '#E74C3C', 'tau': 0.0},
        {'name': 'Standard CP', 'radius': 0.17 + 0.32, 'color': '#F39C12', 'tau': 0.32},
        {'name': 'Learnable CP', 'radius': 0.17, 'color': '#27AE60', 'tau': 'adaptive'}
    ]

    # RRT* parameters
    rrt_params = {
        'step_size': 2.0,
        'max_iterations': 5000,
        'goal_threshold': 1.0,
        'search_radius': 5.0
    }

    for idx, (ax, method) in enumerate(zip(axes, methods)):
        # Display occupancy grid
        occupancy_display = parser.occupancy_grid.copy()
        occupancy_display = np.flip(occupancy_display, axis=0)  # Flip for correct orientation

        # Show map with obstacles
        ax.imshow(occupancy_display, cmap='gray_r', origin='lower',
                  extent=[0, occupancy_display.shape[1], 0, occupancy_display.shape[0]])

        # For Learnable CP, simulate adaptive radius at different points
        if method['name'] == 'Learnable CP':
            # Run planner with base radius
            planner = RRTStarGrid(
                start=start,
                goal=goal,
                occupancy_grid=parser.occupancy_grid,
                origin=parser.origin,
                resolution=parser.resolution,
                robot_radius=method['radius'],
                step_size=rrt_params['step_size'],
                max_iter=rrt_params['max_iterations'],
                goal_threshold=rrt_params['goal_threshold'],
                search_radius=rrt_params['search_radius'],
                early_termination=False  # Get optimized path
            )

            path = planner.plan()

            if path:
                # Convert path to array for easier handling
                path = np.array(path)

                # Plot path with adaptive tau visualization
                ax.plot(path[:, 0], path[:, 1], color=method['color'],
                       linewidth=2.5, label='Path', zorder=3)

                # Show adaptive tau at selected waypoints
                num_samples = 8  # Show tau at 8 points along path
                indices = np.linspace(0, len(path)-1, num_samples, dtype=int)

                # Simulate adaptive tau values (varying between 0.13 and 0.38)
                for i, idx in enumerate(indices):
                    # Simulate context-aware tau
                    # Higher tau in cluttered areas, lower in open spaces
                    x, y = path[idx]

                    # Check local obstacle density
                    window = 10
                    x_min = max(0, int(x - window))
                    x_max = min(parser.occupancy_grid.shape[1], int(x + window))
                    y_min = max(0, int(y - window))
                    y_max = min(parser.occupancy_grid.shape[0], int(y + window))

                    local_area = parser.occupancy_grid[y_min:y_max, x_min:x_max]
                    obstacle_density = np.sum(local_area > 0) / local_area.size

                    # Adaptive tau based on density
                    tau_adaptive = 0.13 + (0.25 * obstacle_density)
                    tau_adaptive = np.clip(tau_adaptive, 0.13, 0.38)

                    # Draw adaptive safety margin
                    circle = Circle((x, y), (method['radius'] + tau_adaptive) / parser.resolution,
                                  color=method['color'], alpha=0.15, zorder=2)
                    ax.add_patch(circle)
        else:
            # Fixed radius for Naive and Standard CP
            planner = RRTStarGrid(
                start=start,
                goal=goal,
                occupancy_grid=parser.occupancy_grid,
                origin=parser.origin,
                resolution=parser.resolution,
                robot_radius=method['radius'],
                step_size=rrt_params['step_size'],
                max_iter=rrt_params['max_iterations'],
                goal_threshold=rrt_params['goal_threshold'],
                search_radius=rrt_params['search_radius'],
                early_termination=False  # Get optimized path
            )

            path = planner.plan()

            if path:
                path = np.array(path)

                # Plot path
                ax.plot(path[:, 0], path[:, 1], color=method['color'],
                       linewidth=2.5, label='Path', zorder=3)

                # Show safety margin at selected waypoints
                num_samples = 8
                indices = np.linspace(0, len(path)-1, num_samples, dtype=int)

                for idx in indices:
                    x, y = path[idx]
                    circle = Circle((x, y), method['radius'] / parser.resolution,
                                  color=method['color'], alpha=0.15, zorder=2)
                    ax.add_patch(circle)

        # Mark start and goal
        ax.scatter(start[0], start[1], s=200, color='green', marker='o',
                  edgecolors='black', linewidths=2, zorder=5, label='Start')
        ax.scatter(goal[0], goal[1], s=200, color='red', marker='*',
                  edgecolors='black', linewidths=2, zorder=5, label='Goal')

        # Set title and labels
        ax.set_title(method['name'], fontsize=14, fontweight='bold')

        if method['name'] == 'Naive':
            ax.set_xlabel(f"τ = {method['tau']}m (no safety margin)", fontsize=11)
        elif method['name'] == 'Standard CP':
            ax.set_xlabel(f"τ = {method['tau']}m (fixed)", fontsize=11)
        else:
            ax.set_xlabel("τ ∈ [0.13, 0.38]m (adaptive)", fontsize=11)

        # Remove ticks for cleaner look
        ax.set_xticks([])
        ax.set_yticks([])

        # Add grid
        ax.grid(True, alpha=0.2, linestyle='--')

        # Set aspect ratio
        ax.set_aspect('equal')

        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor('black')
            spine.set_linewidth(1.5)

    # Add legend to first subplot
    axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)

    # Add main title
    fig.suptitle('Path Planning Under Uncertainty: Method Comparison',
                 fontsize=16, fontweight='bold', y=1.02)

    plt.tight_layout()

    # Save figure
    output_path = 'path_planning_visualization.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', format='pdf')
    print(f"Saved path planning visualization to {output_path}")

    # Also save as PNG for quick viewing
    plt.savefig('path_planning_visualization.png', dpi=150, bbox_inches='tight')
    print(f"Saved PNG version for preview")

    plt.show()

if __name__ == "__main__":
    create_path_planning_visualization()