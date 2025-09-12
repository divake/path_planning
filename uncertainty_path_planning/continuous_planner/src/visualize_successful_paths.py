#!/usr/bin/env python3
"""Visualize successful paths from the test results"""

import json
import numpy as np
import matplotlib.pyplot as plt
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
import yaml

def visualize_successful_path(env_name, test_id, path_data=None):
    """Visualize a successful path test"""
    
    # Load environment config
    with open('config_env.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Get test configuration
    test_config = None
    for test in env_config['environments'][env_name]['tests']:
        if test['id'] == test_id:
            test_config = test
            break
    
    if not test_config:
        return None
    
    # Parse map
    parser = MRPBMapParser(env_name, '../mrpb_dataset')
    
    # Get start and goal
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])
    
    # If no path provided, run planner
    if path_data is None:
        print(f"Running planner for {env_name} test {test_id}...")
        planner = RRTStarGrid(
            start=start,
            goal=goal,
            occupancy_grid=parser.occupancy_grid,
            origin=parser.origin,
            resolution=parser.resolution,
            robot_radius=0.17,
            step_size=2.0,
            max_iter=1500,
            goal_threshold=1.0,
            search_radius=5.0
        )
        path = planner.plan()
        if not path:
            print(f"  Failed to find path")
            return None
        print(f"  Found path with {len(path)} waypoints")
    else:
        path = path_data
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Left: Full map with occupancy grid
    ax1.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower', 
               extent=[parser.origin[0], parser.origin[0] + parser.width_meters,
                      parser.origin[1], parser.origin[1] + parser.height_meters])
    ax1.set_title(f'{env_name.upper()} - Test {test_id}\nOccupancy Grid View', fontsize=12)
    ax1.set_xlabel('X (meters)')
    ax1.set_ylabel('Y (meters)')
    
    # Plot path
    if path:
        path_array = np.array(path)
        ax1.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, 
                label=f'Path ({len(path)} points)', alpha=0.7)
        ax1.plot(path_array[:, 0], path_array[:, 1], 'b.', markersize=3)
    
    # Mark start and goal
    ax1.plot(start[0], start[1], 'go', markersize=12, label=f'Start {start}', 
             markeredgecolor='darkgreen', markeredgewidth=2)
    ax1.plot(goal[0], goal[1], 'r*', markersize=15, label=f'Goal {goal}',
             markeredgecolor='darkred', markeredgewidth=2)
    
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Right: Zoomed view around path
    if path:
        # Calculate bounding box around path
        path_array = np.array(path)
        margin = 3.0  # meters
        x_min = min(path_array[:, 0].min(), start[0], goal[0]) - margin
        x_max = max(path_array[:, 0].max(), start[0], goal[0]) + margin
        y_min = min(path_array[:, 1].min(), start[1], goal[1]) - margin
        y_max = max(path_array[:, 1].max(), start[1], goal[1]) + margin
        
        # Ensure within map bounds
        x_min = max(x_min, parser.origin[0])
        x_max = min(x_max, parser.origin[0] + parser.width_meters)
        y_min = max(y_min, parser.origin[1])
        y_max = min(y_max, parser.origin[1] + parser.height_meters)
        
        ax2.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower',
                  extent=[parser.origin[0], parser.origin[0] + parser.width_meters,
                         parser.origin[1], parser.origin[1] + parser.height_meters])
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        
        # Plot path with waypoint numbers
        ax2.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=3, 
                label='RRT* Path', alpha=0.7)
        
        # Show waypoints
        for i, point in enumerate(path):
            if i % max(1, len(path)//10) == 0 or i == 0 or i == len(path)-1:
                ax2.plot(point[0], point[1], 'b.', markersize=8)
                ax2.annotate(f'{i}', (point[0], point[1]), 
                           xytext=(3, 3), textcoords='offset points', 
                           fontsize=8, color='blue')
        
        # Mark start and goal
        ax2.plot(start[0], start[1], 'go', markersize=15, label='Start',
                markeredgecolor='darkgreen', markeredgewidth=2)
        ax2.plot(goal[0], goal[1], 'r*', markersize=18, label='Goal',
                markeredgecolor='darkred', markeredgewidth=2)
        
        # Add path statistics
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                         for i in range(len(path)-1))
        expected = test_config['distance']
        
        ax2.set_title(f'Zoomed Path View\nPath Length: {path_length:.2f}m (Expected: {expected:.2f}m)', 
                     fontsize=12)
    else:
        ax2.set_title('No Path Found', fontsize=12)
    
    ax2.set_xlabel('X (meters)')
    ax2.set_ylabel('Y (meters)')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect('equal')
    
    plt.suptitle(f'{env_name.upper()} Environment - Test {test_id}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    return fig

# List of successful tests from the results
successful_tests = [
    ('office01add', 1),
    ('office01add', 2),
    ('office02', 3),
    ('room02', 1),
    ('room02', 3),
    ('narrow_graph', 2),
]

print("Generating visualizations for successful paths...")
print("="*60)

for env_name, test_id in successful_tests:
    print(f"\nProcessing {env_name} - Test {test_id}...")
    fig = visualize_successful_path(env_name, test_id)
    if fig:
        filename = f'../plots/success_{env_name}_test{test_id}.png'
        fig.savefig(filename, dpi=100, bbox_inches='tight')
        print(f"  Saved to {filename}")
        plt.close(fig)

# Create a combined overview figure
fig, axes = plt.subplots(3, 3, figsize=(18, 18))
axes = axes.flatten()

for idx, (env_name, test_id) in enumerate(successful_tests):
    ax = axes[idx]
    
    # Parse map
    parser = MRPBMapParser(env_name, '../mrpb_dataset')
    
    # Load config
    with open('config_env.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Get test config
    for test in env_config['environments'][env_name]['tests']:
        if test['id'] == test_id:
            test_config = test
            break
    
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])
    
    # Run planner
    planner = RRTStarGrid(
        start=start,
        goal=goal,
        occupancy_grid=parser.occupancy_grid,
        origin=parser.origin,
        resolution=parser.resolution,
        robot_radius=0.17,
        step_size=2.0,
        max_iter=1500,
        goal_threshold=1.0,
        search_radius=5.0,
        seed=42  # For reproducibility
    )
    
    path = planner.plan()
    
    # Plot
    ax.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower',
              extent=[parser.origin[0], parser.origin[0] + parser.width_meters,
                     parser.origin[1], parser.origin[1] + parser.height_meters])
    
    if path:
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, alpha=0.7)
        path_length = sum(np.linalg.norm(np.array(path[i+1]) - np.array(path[i])) 
                         for i in range(len(path)-1))
        title = f'{env_name} T{test_id}\nPath: {path_length:.1f}m'
    else:
        title = f'{env_name} T{test_id}\nFailed'
    
    ax.plot(start[0], start[1], 'go', markersize=8)
    ax.plot(goal[0], goal[1], 'r*', markersize=10)
    
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('X (m)', fontsize=8)
    ax.set_ylabel('Y (m)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_aspect('equal')

# Hide unused subplots
for idx in range(len(successful_tests), len(axes)):
    axes[idx].set_visible(False)

plt.suptitle('All Successful Path Planning Results - Naive RRT* on MRPB Maps', 
             fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/all_successful_paths_overview.png', dpi=150, bbox_inches='tight')
print(f"\nSaved overview to ../plots/all_successful_paths_overview.png")

print("\n" + "="*60)
print("Visualization complete!")
print(f"Generated {len(successful_tests)} individual plots + 1 overview plot")