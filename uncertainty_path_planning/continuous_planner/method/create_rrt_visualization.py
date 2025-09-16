#!/usr/bin/env python3
"""
Create Publication-Ready RRT* Path Planning Visualization
Shows tree expansion and final paths for Naive, Standard CP, and Learnable CP
Inspired by Sampling_based_Planning visualization style
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D
import yaml
import sys
import os
import random

# Add paths for imports
sys.path.append('..')
sys.path.append('../..')
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid

def create_beautiful_rrt_visualization():
    """Create publication-ready RRT* visualization with tree expansion"""

    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Load configuration
    config_dir = '../config'
    with open(os.path.join(config_dir, 'config_env.yaml'), 'r') as f:
        env_config = yaml.safe_load(f)

    # Choose office01add environment
    env_name = 'office01add'

    # Parse MRPB map
    dataset_path = '../mrpb_dataset'
    parser = MRPBMapParser(env_name, dataset_path)

    # Get test configuration
    env_tests = env_config['environments'][env_name]['tests']
    test_config = env_tests[0]  # First test
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Color scheme
    colors = {
        'naive': '#2E86C1',      # Blue
        'standard': '#E67E22',   # Orange
        'learnable': '#27AE60',  # Green
        'obstacles': '#34495E',  # Dark gray
        'tree': '#BDC3C7',       # Light gray for RRT tree
        'path': None,            # Will use method color
        'start': '#16A085',      # Teal
        'goal': '#E74C3C'        # Red
    }

    # Method configurations
    methods = [
        {
            'name': 'Naive',
            'radius': 0.17,
            'color': colors['naive'],
            'tau': 0.0,
            'title': 'Naive Planning\n(No Safety Margin)'
        },
        {
            'name': 'Standard CP',
            'radius': 0.17 + 0.32,
            'color': colors['standard'],
            'tau': 0.32,
            'title': 'Standard CP\n(Fixed τ = 0.32m)'
        },
        {
            'name': 'Learnable CP',
            'radius': 0.17,  # Will vary adaptively
            'color': colors['learnable'],
            'tau': 'adaptive',
            'title': 'Learnable CP\n(Adaptive τ ∈ [0.13, 0.38]m)'
        }
    ]

    # RRT* parameters
    rrt_params = {
        'step_size': 2.0,
        'max_iterations': 3000,
        'goal_threshold': 1.0,
        'search_radius': 5.0
    }

    for idx, (ax, method) in enumerate(zip(axes, methods)):

        # Create custom occupancy grid for visualization
        occupancy_display = parser.occupancy_grid.copy()

        # Draw obstacles with nice styling
        for y in range(occupancy_display.shape[0]):
            for x in range(occupancy_display.shape[1]):
                if occupancy_display[y, x] > 0:
                    # Draw obstacle blocks
                    rect = Rectangle((x-0.5, y-0.5), 1, 1,
                                   facecolor=colors['obstacles'],
                                   edgecolor='none',
                                   alpha=0.9)
                    ax.add_patch(rect)

        # For Learnable CP, use adaptive radius
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
                early_termination=False
            )
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
                early_termination=False
            )

        # Plan and get tree
        path = planner.plan()

        # Draw RRT tree (sampled edges for visual clarity)
        if hasattr(planner, 'nodes'):
            # Sample tree edges for cleaner visualization
            num_edges_to_show = min(500, len(planner.nodes) // 3)
            sampled_nodes = random.sample(planner.nodes,
                                        min(num_edges_to_show, len(planner.nodes)))

            for node in sampled_nodes:
                if hasattr(node, 'parent') and node.parent is not None:
                    # Draw tree edge
                    ax.plot([node.parent.x, node.x],
                           [node.parent.y, node.y],
                           color=colors['tree'],
                           linewidth=0.3,
                           alpha=0.4,
                           zorder=1)

        # Draw the optimal path
        if path:
            path = np.array(path)

            # Draw path with thicker line
            ax.plot(path[:, 0], path[:, 1],
                   color=method['color'],
                   linewidth=3.5,
                   alpha=0.9,
                   zorder=4,
                   solid_capstyle='round',
                   solid_joinstyle='round')

            # For Learnable CP, show adaptive safety margins
            if method['name'] == 'Learnable CP':
                # Show adaptive tau at key waypoints
                num_samples = 6
                indices = np.linspace(0, len(path)-1, num_samples, dtype=int)

                for i, idx in enumerate(indices):
                    x, y = path[idx]

                    # Calculate local obstacle density
                    window = 8
                    x_min = max(0, int(x - window))
                    x_max = min(parser.occupancy_grid.shape[1], int(x + window))
                    y_min = max(0, int(y - window))
                    y_max = min(parser.occupancy_grid.shape[0], int(y + window))

                    local_area = parser.occupancy_grid[y_min:y_max, x_min:x_max]
                    obstacle_density = np.sum(local_area > 0) / local_area.size

                    # Adaptive tau based on density
                    tau_adaptive = 0.13 + (0.25 * obstacle_density)
                    tau_adaptive = np.clip(tau_adaptive, 0.13, 0.38)

                    # Draw adaptive safety bubble
                    circle = Circle((x, y),
                                  (method['radius'] + tau_adaptive) / parser.resolution,
                                  color=method['color'],
                                  alpha=0.2,
                                  linewidth=1,
                                  linestyle='--',
                                  fill=True,
                                  zorder=2)
                    ax.add_patch(circle)

            else:
                # For Naive and Standard CP, show fixed safety margin at key points
                if method['name'] == 'Standard CP':
                    num_samples = 6
                    indices = np.linspace(0, len(path)-1, num_samples, dtype=int)

                    for idx in indices:
                        x, y = path[idx]
                        circle = Circle((x, y),
                                      method['radius'] / parser.resolution,
                                      color=method['color'],
                                      alpha=0.2,
                                      linewidth=1,
                                      linestyle='--',
                                      fill=True,
                                      zorder=2)
                        ax.add_patch(circle)

        # Draw start and goal with fancy markers
        # Start point
        start_circle = Circle(start, 1.5,
                             facecolor=colors['start'],
                             edgecolor='white',
                             linewidth=2,
                             zorder=5)
        ax.add_patch(start_circle)
        ax.text(start[0], start[1], 'S',
               color='white',
               fontsize=12,
               fontweight='bold',
               ha='center',
               va='center',
               zorder=6)

        # Goal point
        goal_circle = Circle(goal, 1.5,
                           facecolor=colors['goal'],
                           edgecolor='white',
                           linewidth=2,
                           zorder=5)
        ax.add_patch(goal_circle)
        ax.text(goal[0], goal[1], 'G',
               color='white',
               fontsize=12,
               fontweight='bold',
               ha='center',
               va='center',
               zorder=6)

        # Set title
        ax.set_title(method['title'],
                    fontsize=13,
                    fontweight='bold',
                    pad=10)

        # Set axis properties
        ax.set_xlim([0, parser.occupancy_grid.shape[1]])
        ax.set_ylim([0, parser.occupancy_grid.shape[0]])
        ax.set_aspect('equal')

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Add subtle grid
        ax.grid(True, alpha=0.1, linestyle='-', linewidth=0.5)

        # Style the frame
        for spine in ax.spines.values():
            spine.set_edgecolor('#2C3E50')
            spine.set_linewidth(2)

        # Add path statistics
        if path is not None:
            path_length = planner.compute_path_length(path)
            stats_text = f"Path Length: {path_length:.1f}m\nWaypoints: {len(path)}"
            ax.text(0.02, 0.98, stats_text,
                   transform=ax.transAxes,
                   fontsize=9,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.3',
                            facecolor='white',
                            edgecolor='gray',
                            alpha=0.8))

    # Add legend
    legend_elements = [
        Line2D([0], [0], color=colors['tree'], linewidth=1, alpha=0.5, label='RRT* Tree'),
        Line2D([0], [0], color=colors['naive'], linewidth=3, label='Optimal Path'),
        patches.Circle((0, 0), 0.1, facecolor=colors['start'], edgecolor='white', label='Start'),
        patches.Circle((0, 0), 0.1, facecolor=colors['goal'], edgecolor='white', label='Goal'),
    ]
    axes[1].legend(handles=legend_elements,
                  loc='upper center',
                  bbox_to_anchor=(0.5, -0.05),
                  ncol=4,
                  frameon=True,
                  fancybox=True,
                  shadow=True,
                  fontsize=10)

    # Main title
    fig.suptitle('RRT* Path Planning: Safety Margin Comparison',
                fontsize=16,
                fontweight='bold',
                y=1.02)

    plt.tight_layout()

    # Save figures
    output_dir = './'
    plt.savefig(os.path.join(output_dir, 'rrt_visualization_comparison.pdf'),
               dpi=300, bbox_inches='tight', format='pdf')
    plt.savefig(os.path.join(output_dir, 'rrt_visualization_comparison.png'),
               dpi=150, bbox_inches='tight')

    print("Saved RRT* visualization to rrt_visualization_comparison.pdf/png")

    plt.show()

if __name__ == "__main__":
    create_beautiful_rrt_visualization()