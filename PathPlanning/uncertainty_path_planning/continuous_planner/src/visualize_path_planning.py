#!/usr/bin/env python3
"""
Path Planning Visualization Script

Visualizes all 12 successful path planning results showing:
- Original RRT* waypoints
- Smooth quintic polynomial trajectory
- Environment obstacles
- Start/goal positions
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('../../../CurvesGenerator')
from quintic_polynomial import QuinticPolynomial

from mrpb_map_parser import MRPBMapParser
from test_runner import generate_realistic_trajectory


def plot_single_environment(env_name, test_id, result_data, save_dir="../plots/map_planning"):
    """
    Plot a single environment with path planning results
    
    Args:
        env_name: Environment name
        test_id: Test ID
        result_data: Result data from JSON
        save_dir: Directory to save plots
    """
    
    # Load map
    parser = MRPBMapParser(env_name, '../mrpb_dataset')
    
    # Get path from results
    path = result_data['naive']['path']
    if not path:
        print(f"No path found for {env_name}-{test_id}")
        return
    
    path = np.array(path)
    
    # Generate smooth trajectory
    smooth_trajectory = generate_realistic_trajectory(path, parser)
    
    # Extract smooth trajectory points
    smooth_x = [data.x for data in smooth_trajectory]
    smooth_y = [data.y for data in smooth_trajectory]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Original RRT* waypoints
    ax1.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower',
               extent=[parser.origin[0], parser.origin[0] + parser.width_pixels * parser.resolution,
                      parser.origin[1], parser.origin[1] + parser.height_pixels * parser.resolution])
    
    # Plot RRT* waypoints
    ax1.plot(path[:, 0], path[:, 1], 'ro-', linewidth=2, markersize=6, 
             label=f'RRT* Path ({len(path)} waypoints)', alpha=0.8)
    
    # Start and goal
    ax1.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax1.plot(path[-1, 0], path[-1, 1], 'bo', markersize=12, label='Goal', markeredgecolor='black')
    
    ax1.set_title(f'{env_name.upper()} Test-{test_id}\nRRT* Waypoints ({len(path)} points)', 
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Smooth Quintic Polynomial Trajectory
    ax2.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower',
               extent=[parser.origin[0], parser.origin[0] + parser.width_pixels * parser.resolution,
                      parser.origin[1], parser.origin[1] + parser.height_pixels * parser.resolution])
    
    # Plot smooth trajectory
    ax2.plot(smooth_x, smooth_y, 'b-', linewidth=3, label=f'Smooth Trajectory ({len(smooth_x)} points)', alpha=0.8)
    
    # Plot original waypoints as reference
    ax2.plot(path[:, 0], path[:, 1], 'ro', markersize=4, label='RRT* Waypoints', alpha=0.6)
    
    # Start and goal
    ax2.plot(path[0, 0], path[0, 1], 'go', markersize=12, label='Start', markeredgecolor='black')
    ax2.plot(path[-1, 0], path[-1, 1], 'bo', markersize=12, label='Goal', markeredgecolor='black')
    
    ax2.set_title(f'{env_name.upper()} Test-{test_id}\nSmooth Trajectory ({len(smooth_x)} points)', 
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    # Add metrics info
    metrics = result_data['naive'].get('metrics', {})
    path_length = result_data['naive'].get('path_length', 0)
    planning_time = result_data['naive'].get('planning_time', 0)
    
    # Extract key metrics
    d_0 = metrics.get('safety', {}).get('d_0', 0)
    d_avg = metrics.get('safety', {}).get('d_avg', 0)
    p_0 = metrics.get('safety', {}).get('p_0', 0)
    T = metrics.get('efficiency', {}).get('T', 0)
    f_vs = metrics.get('smoothness', {}).get('f_vs', 0)
    
    # Add text box with metrics
    info_text = f"""Path Metrics:
Length: {path_length:.1f}m
Plan Time: {planning_time:.1f}s
Travel Time: {T:.1f}s
d_0: {d_0:.3f}m
d_avg: {d_avg:.3f}m  
p_0: {p_0:.1f}%
f_vs: {f_vs:.2f} m/s²"""
    
    ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{save_dir}/{env_name}_test{test_id}_path_planning.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {filename}")


def visualize_all_paths():
    """Visualize all 12 successful path planning results"""
    
    # Load latest results
    results_file = "../results/results_20250909_152644.json"
    try:
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    except FileNotFoundError:
        print(f"Results file not found: {results_file}")
        return
    
    print("Generating path planning visualizations for all environments...")
    print("=" * 70)
    
    plot_count = 0
    
    # Generate plots for each successful test
    for env_name, env_results in all_results.items():
        for test_name, test_results in env_results.items():
            if 'naive' in test_results and test_results['naive'].get('success', False):
                test_id = int(test_name.split('_')[1])
                print(f"Processing {env_name} Test-{test_id}...")
                
                try:
                    plot_single_environment(env_name, test_id, test_results)
                    plot_count += 1
                except Exception as e:
                    print(f"Error plotting {env_name}-{test_id}: {e}")
    
    print("=" * 70)
    print(f"Generated {plot_count} path planning visualizations")
    print(f"Saved to: ../plots/map_planning/")
    
    # Create summary comparison plot
    create_summary_comparison(all_results)


def create_summary_comparison(all_results):
    """Create a summary plot showing all 12 paths in a grid"""
    
    # Determine grid size
    total_tests = sum(len([t for t in tests.keys() if 'naive' in tests[t] and tests[t]['naive'].get('success', False)]) 
                     for tests in all_results.values())
    
    cols = 4
    rows = (total_tests + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 15))
    axes = axes.flatten() if rows > 1 else [axes] if rows == 1 else axes
    
    plot_idx = 0
    
    for env_name, env_results in all_results.items():
        for test_name, test_results in env_results.items():
            if 'naive' in test_results and test_results['naive'].get('success', False):
                if plot_idx >= len(axes):
                    break
                    
                ax = axes[plot_idx]
                test_id = int(test_name.split('_')[1])
                
                try:
                    # Load map
                    parser = MRPBMapParser(env_name, '../mrpb_dataset')
                    path = np.array(test_results['naive']['path'])
                    
                    # Generate smooth trajectory  
                    smooth_trajectory = generate_realistic_trajectory(path, parser)
                    smooth_x = [data.x for data in smooth_trajectory]
                    smooth_y = [data.y for data in smooth_trajectory]
                    
                    # Plot
                    ax.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower',
                             extent=[parser.origin[0], parser.origin[0] + parser.width_pixels * parser.resolution,
                                    parser.origin[1], parser.origin[1] + parser.height_pixels * parser.resolution])
                    
                    ax.plot(smooth_x, smooth_y, 'b-', linewidth=2, alpha=0.8)
                    ax.plot(path[:, 0], path[:, 1], 'ro', markersize=3, alpha=0.6)
                    ax.plot(path[0, 0], path[0, 1], 'go', markersize=8, markeredgecolor='black')
                    ax.plot(path[-1, 0], path[-1, 1], 'bo', markersize=8, markeredgecolor='black')
                    
                    ax.set_title(f'{env_name}-{test_id}', fontsize=10, fontweight='bold')
                    ax.set_aspect('equal')
                    ax.grid(True, alpha=0.3)
                    
                except Exception as e:
                    ax.text(0.5, 0.5, f'Error: {env_name}-{test_id}', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'{env_name}-{test_id} (Error)', fontsize=10)
                
                plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('All 12 RRT* Path Planning Results with Quintic Polynomial Smoothing', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    summary_file = "../plots/map_planning/ALL_paths_summary.png"
    plt.savefig(summary_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary plot saved: {summary_file}")


if __name__ == "__main__":
    visualize_all_paths()