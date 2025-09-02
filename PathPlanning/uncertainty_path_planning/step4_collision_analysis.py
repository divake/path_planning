#!/usr/bin/env python3
"""
STEP 4: COLLISION ANALYSIS - Understanding why no collisions occur
Goal: Analyze the current noise model and design alternatives that cause collisions
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")

from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import env

# Import our noise models
from step2_noise_model import add_perception_noise_thickness


def add_phantom_obstacles(true_obstacles, phantom_prob=0.02, seed=None):
    """
    Add random phantom obstacles that don't exist in reality.
    These cause collisions when the robot tries to avoid them.
    """
    if seed is not None:
        np.random.seed(seed)
    
    perceived_obstacles = set(true_obstacles)
    
    # Add phantom obstacles near passages
    critical_zones = [
        # Near passages where phantom obstacles would force bad paths
        (18, 22, 10, 20),  # Around first passage
        (28, 32, 10, 20),  # Around second passage  
        (38, 42, 10, 20),  # Around third passage
    ]
    
    for x_min, x_max, y_min, y_max in critical_zones:
        for x in range(x_min, x_max):
            for y in range(y_min, y_max):
                if (x, y) not in true_obstacles and np.random.random() < phantom_prob:
                    perceived_obstacles.add((x, y))
    
    return perceived_obstacles


def add_thinning_noise(true_obstacles, thin_prob=0.1, seed=None):
    """
    Make walls appear thinner - remove some obstacle cells.
    This causes robot to think it can pass through walls.
    """
    if seed is not None:
        np.random.seed(seed)
    
    perceived_obstacles = set()
    
    # Keep boundary walls always
    for (x, y) in true_obstacles:
        if x == 0 or x == 50 or y == 0 or y == 30:
            perceived_obstacles.add((x, y))
        elif np.random.random() > thin_prob:
            perceived_obstacles.add((x, y))
    
    return perceived_obstacles


def add_mixed_noise(true_obstacles, thickness_std=0.2, phantom_prob=0.01, thin_prob=0.05, seed=None):
    """
    Combined noise model:
    - Some walls get thicker (original approach)
    - Some phantom obstacles appear
    - Some walls get thinner
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Start with thickness noise
    perceived = add_perception_noise_thickness(true_obstacles, thickness_std, seed)
    
    # Add phantoms
    for x in range(5, 45):
        for y in range(5, 25):
            if (x, y) not in perceived and (x, y) not in true_obstacles:
                if np.random.random() < phantom_prob:
                    perceived.add((x, y))
    
    # Thin some walls (but not boundaries)
    to_remove = set()
    for (x, y) in perceived:
        if x != 0 and x != 50 and y != 0 and y != 30:
            if (x, y) in true_obstacles and np.random.random() < thin_prob:
                to_remove.add((x, y))
    
    perceived = perceived - to_remove
    
    return perceived


def test_noise_models(true_env, num_trials=100):
    """
    Test different noise models to see which causes collisions.
    """
    true_obstacles = true_env.obs
    s_start = (5, 15)
    s_goal = (45, 15)
    
    models = [
        ("Thickness Only (σ=0.3)", lambda seed: add_perception_noise_thickness(true_obstacles, 0.3, seed)),
        ("Thickness Only (σ=0.5)", lambda seed: add_perception_noise_thickness(true_obstacles, 0.5, seed)),
        ("Phantom Obstacles (2%)", lambda seed: add_phantom_obstacles(true_obstacles, 0.02, seed)),
        ("Thinning Walls (10%)", lambda seed: add_thinning_noise(true_obstacles, 0.1, seed)),
        ("Mixed Noise", lambda seed: add_mixed_noise(true_obstacles, 0.2, 0.01, 0.05, seed)),
    ]
    
    results = []
    
    for model_name, noise_func in models:
        collision_counts = []
        path_lengths = []
        failed_paths = 0
        
        for trial in range(num_trials):
            perceived = noise_func(trial)
            
            try:
                astar = Astar.AStar(s_start, s_goal, "euclidean")
                astar.obs = perceived
                path, _ = astar.searching()
                
                if path:
                    # Check collisions with TRUE obstacles
                    collisions = sum(1 for p in path if p in true_obstacles)
                    collision_counts.append(collisions)
                    path_lengths.append(len(path))
                else:
                    failed_paths += 1
            except:
                failed_paths += 1
        
        collision_rate = np.mean([c > 0 for c in collision_counts]) * 100 if collision_counts else 0
        avg_collisions = np.mean(collision_counts) if collision_counts else 0
        avg_path_length = np.mean(path_lengths) if path_lengths else 0
        
        results.append({
            'model': model_name,
            'collision_rate': collision_rate,
            'avg_collisions': avg_collisions,
            'avg_path_length': avg_path_length,
            'failed_paths': failed_paths,
            'success_rate': (num_trials - failed_paths) / num_trials * 100
        })
    
    return results


def visualize_noise_effects(true_env):
    """
    Visualize how different noise models affect perception and paths.
    """
    true_obstacles = true_env.obs
    s_start = (5, 15)
    s_goal = (45, 15)
    
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(3, 3, figure=fig)
    
    noise_configs = [
        ("Ground Truth", lambda: true_obstacles),
        ("Thickness σ=0.3", lambda: add_perception_noise_thickness(true_obstacles, 0.3, 42)),
        ("Thickness σ=0.5", lambda: add_perception_noise_thickness(true_obstacles, 0.5, 42)),
        ("Phantom 2%", lambda: add_phantom_obstacles(true_obstacles, 0.02, 42)),
        ("Phantom 5%", lambda: add_phantom_obstacles(true_obstacles, 0.05, 42)),
        ("Thinning 10%", lambda: add_thinning_noise(true_obstacles, 0.1, 42)),
        ("Thinning 20%", lambda: add_thinning_noise(true_obstacles, 0.2, 42)),
        ("Mixed Noise", lambda: add_mixed_noise(true_obstacles, 0.2, 0.01, 0.05, 42)),
        ("Mixed Heavy", lambda: add_mixed_noise(true_obstacles, 0.3, 0.02, 0.1, 42)),
    ]
    
    for idx, (title, noise_func) in enumerate(noise_configs):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Get perceived obstacles
        perceived = noise_func()
        
        # Plot true obstacles in light gray
        for obs in true_obstacles:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='gray',
                                    facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
        
        # Plot perceived obstacles
        for obs in perceived:
            if obs in true_obstacles:
                # Correctly perceived
                color = 'black'
                alpha = 0.6
            else:
                # Phantom obstacle
                color = 'red'
                alpha = 0.4
            
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor=color,
                                    facecolor=color, alpha=alpha)
            ax.add_patch(rect)
        
        # Plot missing obstacles (thinned walls)
        for obs in true_obstacles:
            if obs not in perceived:
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=1, edgecolor='blue',
                                        facecolor='none', linestyle='--')
                ax.add_patch(rect)
        
        # Try to plan path
        try:
            astar = Astar.AStar(s_start, s_goal, "euclidean")
            astar.obs = perceived
            path, _ = astar.searching()
            
            if path:
                # Check for collisions
                collision_points = [p for p in path if p in true_obstacles]
                
                # Plot path
                path_x = [p[0] for p in path]
                path_y = [p[1] for p in path]
                ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.6)
                
                # Mark collisions
                if collision_points:
                    for cp in collision_points:
                        ax.plot(cp[0], cp[1], 'rx', markersize=10, markeredgewidth=2)
                
                # Add statistics
                stats = f"Path: {len(path)} cells\n"
                stats += f"Collisions: {len(collision_points)}"
                if collision_points:
                    stats += " ⚠️"
                
                ax.text(2, 28, stats, fontsize=9,
                       bbox=dict(boxstyle="round,pad=0.3", 
                                facecolor="yellow" if collision_points else "lightgreen", 
                                alpha=0.7),
                       verticalalignment='top')
        except:
            ax.text(2, 28, "Path Failed!", fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7),
                   verticalalignment='top')
        
        # Mark start and goal
        ax.plot(5, 15, 'go', markersize=8)
        ax.plot(45, 15, 'ro', markersize=8)
    
    plt.suptitle('Noise Model Comparison: Finding Models that Cause Collisions', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    save_path = 'results/step4_collision_analysis.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nCollision analysis saved to: {save_path}")
    plt.close()


def main():
    """
    Main collision analysis.
    """
    print("="*80)
    print("STEP 4: COLLISION ANALYSIS")
    print("="*80)
    print("\nProblem: Current thickness-based noise doesn't cause collisions")
    print("Reason: Thicker walls make robot MORE cautious, not less")
    print("Solution: Test alternative noise models that induce collisions")
    
    # Get environment
    true_env = env.Env()
    
    # Test different noise models
    print("\n" + "="*80)
    print("TESTING DIFFERENT NOISE MODELS (100 trials each)")
    print("="*80)
    
    results = test_noise_models(true_env, num_trials=100)
    
    # Print results table
    print(f"\n{'Model':<25} | {'Collision Rate':>14} | {'Avg Collisions':>14} | {'Avg Path Len':>12} | {'Success Rate':>12}")
    print("-"*80)
    
    for r in results:
        print(f"{r['model']:<25} | {r['collision_rate']:>13.1f}% | {r['avg_collisions']:>14.2f} | {r['avg_path_length']:>12.1f} | {r['success_rate']:>11.1f}%")
    
    print("-"*80)
    
    # Analysis
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    best_model = max(results, key=lambda x: x['collision_rate'])
    
    if best_model['collision_rate'] > 10:
        print(f"\n✅ SUCCESS! Found noise model with >10% collision rate:")
        print(f"   Model: {best_model['model']}")
        print(f"   Collision Rate: {best_model['collision_rate']:.1f}%")
        print(f"\n   This model can be used to demonstrate naive method failure!")
    else:
        print(f"\n⚠️ No model achieved >10% collision rate")
        print(f"   Best: {best_model['model']} with {best_model['collision_rate']:.1f}%")
        print(f"\n   Need more aggressive noise models:")
        print(f"   - Higher phantom obstacle probability")
        print(f"   - More aggressive wall thinning")
        print(f"   - Strategic obstacle placement/removal")
    
    # Generate visualization
    print("\nGenerating noise model comparison visualization...")
    visualize_noise_effects(true_env)
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print("\n1. PHANTOM OBSTACLES: Add fake obstacles that don't exist")
    print("   → Forces robot to take detours into real walls")
    print("\n2. WALL THINNING: Make walls appear thinner than reality")
    print("   → Robot thinks it can pass through solid walls")
    print("\n3. MIXED APPROACH: Combine multiple noise types")
    print("   → More realistic perception errors")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()