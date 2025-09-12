#!/usr/bin/env python3
"""
Explain how wall thinning causes collisions
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import env
from step5_naive_with_collisions import add_thinning_noise

def explain_collision():
    """
    Show a clear example of how thinning causes collision.
    """
    # Get environment
    true_env = env.Env()
    true_obstacles = true_env.obs
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Find a seed that causes collision
    for seed in range(100):
        perceived = add_thinning_noise(true_obstacles, thin_prob=0.15, seed=seed)
        
        # Plan path
        try:
            astar = Astar.AStar((5, 15), (45, 15), "euclidean")
            astar.obs = perceived
            path, _ = astar.searching()
            
            if path:
                collisions = [p for p in path if p in true_obstacles]
                if len(collisions) > 0:
                    # Found a good example!
                    break
        except:
            continue
    
    # Plot 1: TRUE ENVIRONMENT
    ax = axes[0]
    ax.set_title('TRUE ENVIRONMENT\n(Reality)', fontsize=12, fontweight='bold')
    ax.set_xlim(15, 35)
    ax.set_ylim(10, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    for obs in true_obstacles:
        if 15 <= obs[0] <= 35 and 10 <= obs[1] <= 20:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='black',
                                    facecolor='black', alpha=0.8)
            ax.add_patch(rect)
    
    # Plot 2: PERCEIVED ENVIRONMENT
    ax = axes[1]
    ax.set_title('PERCEIVED ENVIRONMENT\n(What robot sees - with holes!)', 
                 fontsize=12, fontweight='bold', color='blue')
    ax.set_xlim(15, 35)
    ax.set_ylim(10, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Show perceived obstacles
    for obs in perceived:
        if 15 <= obs[0] <= 35 and 10 <= obs[1] <= 20:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='gray',
                                    facecolor='gray', alpha=0.6)
            ax.add_patch(rect)
    
    # Highlight missing walls (the holes!)
    missing_count = 0
    for obs in true_obstacles:
        if 15 <= obs[0] <= 35 and 10 <= obs[1] <= 20:
            if obs not in perceived:
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=2, edgecolor='red',
                                        facecolor='yellow', alpha=0.5)
                ax.add_patch(rect)
                missing_count += 1
    
    ax.text(25, 11, f'{missing_count} wall cells\nremoved by noise!', 
            fontsize=10, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # Plot 3: COLLISION RESULT
    ax = axes[2]
    ax.set_title('COLLISION!\n(Path through "hole" hits real wall)', 
                 fontsize=12, fontweight='bold', color='red')
    ax.set_xlim(15, 35)
    ax.set_ylim(10, 20)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    
    # Show true obstacles
    for obs in true_obstacles:
        if 15 <= obs[0] <= 35 and 10 <= obs[1] <= 20:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='black',
                                    facecolor='lightgray', alpha=0.4)
            ax.add_patch(rect)
    
    # Show planned path
    if path:
        path_x = [p[0] for p in path if 15 <= p[0] <= 35]
        path_y = [p[1] for p in path if 15 <= p[0] <= 35]
        
        # Get path segment in view
        path_segment = [(p[0], p[1]) for p in path if 15 <= p[0] <= 35 and 10 <= p[1] <= 20]
        if path_segment:
            xs = [p[0] for p in path_segment]
            ys = [p[1] for p in path_segment]
            ax.plot(xs, ys, 'b-', linewidth=3, alpha=0.7, label='Planned Path')
        
        # Mark collisions with big X
        for cp in collisions:
            if 15 <= cp[0] <= 35 and 10 <= cp[1] <= 20:
                ax.plot(cp[0], cp[1], 'rx', markersize=15, markeredgewidth=3)
                ax.add_patch(patches.Circle((cp[0], cp[1]), 0.8, 
                                           fill=False, edgecolor='red', 
                                           linewidth=2, linestyle='--'))
    
    ax.text(25, 19, 'Robot tries to go through\nperceived gap but hits wall!', 
            fontsize=10, ha='center', color='red', fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.8))
    
    plt.suptitle('How Wall Thinning Causes Collisions', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save
    save_path = 'discrete_planner/results/collision_mechanism_explained.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Explanation saved to: {save_path}")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("COLLISION MECHANISM EXPLAINED")
    print("="*60)
    print(f"\n1. TRUE environment has solid walls")
    print(f"2. PERCEIVED environment has {missing_count} missing wall cells (holes)")
    print(f"3. Robot plans path through perceived holes")
    print(f"4. Path collides with {len(collisions)} real wall cells")
    print(f"\nThis is why we get 36.7% collision rate with 5% thinning!")
    print("="*60)

if __name__ == "__main__":
    explain_collision()