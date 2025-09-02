#!/usr/bin/env python3
"""
STEP 1: BASELINE VERIFICATION
Goal: Confirm we can use PathPlanning's A* and environment correctly
Expected: Should find a path from (5,15) to (45,15) through narrow passages
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path to import PathPlanning modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

# Import A* and environment from PathPlanning
from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import plotting
from Search_based_Planning.Search_2D import env

def test_baseline():
    """
    Test basic A* functionality with existing environment
    """
    print("="*50)
    print("STEP 1: BASELINE VERIFICATION")
    print("="*50)
    
    # Define start and goal
    # Using middle height to force path through narrow passages
    s_start = (5, 15)   # Left side, middle height
    s_goal = (45, 15)   # Right side, middle height
    
    print(f"Start: {s_start}")
    print(f"Goal: {s_goal}")
    
    # Create A* planner with Euclidean heuristic
    astar = Astar.AStar(s_start, s_goal, "euclidean")
    
    # Get environment info
    print(f"\nEnvironment size: {astar.Env.x_range} x {astar.Env.y_range}")
    print(f"Number of obstacles: {len(astar.obs)}")
    
    # Run A* search
    print("\nRunning A* search...")
    path, visited = astar.searching()
    
    # Path analysis
    if path:
        print(f"\n✓ Path found successfully!")
        print(f"  Path length: {len(path)} cells")
        print(f"  Nodes visited: {len(visited)}")
        
        # Calculate actual path distance
        path_distance = 0
        for i in range(len(path)-1):
            dx = path[i+1][0] - path[i][0]
            dy = path[i+1][1] - path[i][1]
            path_distance += np.sqrt(dx**2 + dy**2)
        print(f"  Path distance: {path_distance:.2f} units")
        
        # Check if path goes through expected narrow passages
        x_coords = [p[0] for p in path]
        narrow_passages = [20, 30, 40]  # Expected x-coordinates of walls
        passages_crossed = []
        for passage_x in narrow_passages:
            if passage_x in x_coords or passage_x+1 in x_coords or passage_x-1 in x_coords:
                passages_crossed.append(passage_x)
        
        print(f"\n  Narrow passages crossed: {passages_crossed}")
        
        # Show first and last few points of path
        print(f"\n  Path start: {path[:3]}")
        print(f"  Path end: {path[-3:]}")
        
    else:
        print("\n✗ No path found!")
        return False
    
    # Create results directory if it doesn't exist
    os.makedirs('discrete_planner/results', exist_ok=True)
    
    # Visualize
    print("\nGenerating visualization...")
    
    # Create a cleaner visualization showing just path and obstacles
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot environment
    ax.set_xlim(-1, 51)
    ax.set_ylim(-1, 31)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('STEP 1: Baseline A* Path Through Environment')
    
    # Plot obstacles
    for obs in astar.obs:
        ax.add_patch(plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                                  color='black', alpha=0.8))
    
    # Plot path
    if path:
        path_x = [p[0] for p in path]
        path_y = [p[1] for p in path]
        ax.plot(path_x, path_y, 'b-', linewidth=2, label='A* Path', zorder=5)
        ax.plot(path_x, path_y, 'bo', markersize=3, zorder=6)
    
    # Mark start and goal
    ax.plot(s_start[0], s_start[1], 'go', markersize=10, label='Start', zorder=10)
    ax.plot(s_goal[0], s_goal[1], 'ro', markersize=10, label='Goal', zorder=10)
    
    # Highlight narrow passages
    for x in [20, 30, 40]:
        ax.axvline(x, color='red', alpha=0.2, linestyle='--', label='Narrow passage' if x==20 else '')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    
    # Save the figure
    save_path = 'discrete_planner/results/step1_baseline_path.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    
    # Close the figure to free memory
    plt.close()
    
    print("\n" + "="*50)
    print("BASELINE TEST COMPLETE")
    print("="*50)
    
    # Success criteria check
    print("\nSuccess Criteria Check:")
    success = True
    
    if path:
        print("✓ Path found successfully")
    else:
        print("✗ Path not found")
        success = False
    
    if 40 <= len(path) <= 60:
        print(f"✓ Path length reasonable ({len(path)} cells)")
    else:
        print(f"⚠ Path length unexpected ({len(path)} cells)")
    
    if len(passages_crossed) >= 2:
        print(f"✓ Path crosses narrow passages ({passages_crossed})")
    else:
        print(f"⚠ Path may not cross all passages ({passages_crossed})")
    
    if success:
        print("\n✅ READY TO PROCEED TO STEP 2")
    else:
        print("\n❌ Issues found - debug before proceeding")
    
    return success


if __name__ == "__main__":
    # Run baseline test
    success = test_baseline()
    
    if success:
        print("\nNext step: Implement noise model (step2_noise_model.py)")
    else:
        print("\nDebug issues before moving to Step 2")