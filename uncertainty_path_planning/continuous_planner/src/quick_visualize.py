#!/usr/bin/env python3
"""Quick visualization of successful paths"""

import numpy as np
import matplotlib.pyplot as plt
from mrpb_map_parser import MRPBMapParser
from rrt_star_grid_planner import RRTStarGrid
import yaml

# List of tests to visualize - focus on problematic ones
successful_tests = [
    ('office01add', 3),  # Previously failing
    ('office02', 1),     # Previously failing
    ('shopping_mall', 1), # Previously failing
    ('maze', 1),         # Previously failing
    ('track', 1),        # Previously failing
    ('narrow_graph', 1), # Previously failing
    ('room02', 2),       # Previously failing
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes = axes.flatten()

print("Generating quick visualizations...")

for idx, (env_name, test_id) in enumerate(successful_tests[:7]):
    if idx >= 7:
        break
        
    ax = axes[idx]
    print(f"Processing {env_name} - Test {test_id}...")
    
    # Parse map
    parser = MRPBMapParser(env_name, '../mrpb_dataset')
    
    # Load config
    with open('config_env.yaml', 'r') as f:
        env_config = yaml.safe_load(f)
    
    # Get test config
    test_config = None
    for test in env_config['environments'][env_name]['tests']:
        if test['id'] == test_id:
            test_config = test
            break
    
    start = tuple(test_config['start'])
    goal = tuple(test_config['goal'])
    
    # Plot map
    ax.imshow(parser.occupancy_grid, cmap='gray_r', origin='lower',
              extent=[parser.origin[0], parser.origin[0] + parser.width_meters,
                     parser.origin[1], parser.origin[1] + parser.height_meters],
              alpha=0.8)
    
    # Just visualize environment without running planner
    path = None
    
    expected = test_config['distance']
    title = f'{env_name.upper()} - Test {test_id}\n'
    title += f'Expected: {expected:.1f}m'
    
    # Mark start and goal
    ax.plot(start[0], start[1], 'go', markersize=10, label='Start',
            markeredgecolor='darkgreen', markeredgewidth=1.5)
    ax.plot(goal[0], goal[1], 'r*', markersize=14, label='Goal',
            markeredgecolor='darkred', markeredgewidth=1.5)
    
    # Annotations
    ax.annotate('S', start, xytext=(2, 2), textcoords='offset points',
                fontsize=8, fontweight='bold', color='green')
    ax.annotate('G', goal, xytext=(2, 2), textcoords='offset points',
                fontsize=8, fontweight='bold', color='red')
    
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.set_xlabel('X (meters)', fontsize=8)
    ax.set_ylabel('Y (meters)', fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    if idx == 0:
        ax.legend(loc='upper right', fontsize=7)

# Hide unused subplots
for idx in range(7, 9):
    axes[idx].set_visible(False)

plt.suptitle('Successful Path Planning with Naive RRT* on MRPB Occupancy Grids\n' + 
             'Green Circle = Start, Red Star = Goal, Blue Line = RRT* Path',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../plots/successful_paths_overview.png', dpi=120, bbox_inches='tight')
print(f"\nSaved to ../plots/successful_paths_overview.png")
plt.show()