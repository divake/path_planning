#!/usr/bin/env python3
"""
STEP 3: VERIFY PROBLEM - ANALYZE PATH BEHAVIOR WITH NOISE
Goal: Show how paths change with noise levels
Analysis: Path length should increase as noise increases (avoiding thicker walls)
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import env

# Import our noise model from step 2
from step2_noise_model import add_perception_noise_thickness


def check_collision(path, true_obstacles):
    """
    Check if a path collides with true obstacles.
    """
    collisions = []
    for i, point in enumerate(path):
        if point in true_obstacles:
            collisions.append((i, point))
    
    return {
        'has_collision': len(collisions) > 0,
        'num_collisions': len(collisions),
        'collision_points': collisions,
        'collision_rate': len(collisions) / len(path) if path else 0
    }


def run_trial_with_noise(true_env, noise_std=0.3, seed=None):
    """
    Run a single trial with given noise level.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Get true obstacles
    true_obstacles = true_env.obs
    
    # Add perception noise
    perceived_obstacles = add_perception_noise_thickness(true_obstacles, noise_std, seed)
    
    # Plan path on PERCEIVED environment
    s_start = (5, 15)
    s_goal = (45, 15)
    
    try:
        astar = Astar.AStar(s_start, s_goal, "euclidean")
        astar.obs = perceived_obstacles  # Use perceived obstacles
        path, visited = astar.searching()
    except:
        return None
    
    if not path:
        return None
    
    # Check for collisions
    collision_info = check_collision(path, true_obstacles)
    
    # Calculate path metrics
    path_length = len(path)
    
    # Calculate actual distance
    total_distance = 0
    for i in range(len(path)-1):
        dx = path[i+1][0] - path[i][0]
        dy = path[i+1][1] - path[i][1]
        total_distance += np.sqrt(dx**2 + dy**2)
    
    return {
        'path': path,
        'path_length': path_length,
        'path_distance': total_distance,
        'perceived_obstacles': perceived_obstacles,
        'collision_info': collision_info
    }


def visualize_paths_comparison(true_env, noise_levels, num_trials=10):
    """
    Visualize paths for different noise levels.
    """
    true_obstacles = true_env.obs
    
    # Create figure with subplots for each noise level
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    # Store results for table
    results_table = []
    
    for idx, noise_std in enumerate(noise_levels):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Title
        if noise_std == 0:
            ax.set_title(f'No Noise (Baseline)', fontsize=12, fontweight='bold')
        else:
            ax.set_title(f'Noise σ = {noise_std}', fontsize=12, fontweight='bold')
        
        # Set up plot
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Collect path statistics
        path_lengths = []
        path_distances = []
        collision_counts = []
        
        # Run multiple trials for this noise level
        for trial in range(num_trials):
            if noise_std == 0:
                # No noise - use true obstacles
                perceived_obstacles = true_obstacles
                seed = None
            else:
                # Add noise
                perceived_obstacles = add_perception_noise_thickness(true_obstacles, noise_std, seed=trial)
            
            # Plan path
            s_start = (5, 15)
            s_goal = (45, 15)
            
            try:
                astar = Astar.AStar(s_start, s_goal, "euclidean")
                astar.obs = perceived_obstacles
                path, _ = astar.searching()
                
                if path:
                    # Calculate metrics
                    path_lengths.append(len(path))
                    
                    distance = 0
                    for i in range(len(path)-1):
                        dx = path[i+1][0] - path[i][0]
                        dy = path[i+1][1] - path[i][1]
                        distance += np.sqrt(dx**2 + dy**2)
                    path_distances.append(distance)
                    
                    # Check collisions
                    collision_info = check_collision(path, true_obstacles)
                    collision_counts.append(collision_info['num_collisions'])
                    
                    # Plot path (only first few for clarity)
                    if trial < 3:
                        path_x = [p[0] for p in path]
                        path_y = [p[1] for p in path]
                        alpha = 0.3 + 0.2 * (2 - trial)  # Different transparency
                        ax.plot(path_x, path_y, 'b-', linewidth=1.5, alpha=alpha)
            except:
                pass
        
        # Plot obstacles (perceived for this noise level)
        if noise_std > 0:
            # Show one sample of perceived obstacles
            perceived_sample = add_perception_noise_thickness(true_obstacles, noise_std, seed=0)
        else:
            perceived_sample = true_obstacles
            
        # Plot true obstacles in light gray
        for obs in true_obstacles:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='gray',
                                    facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
        
        # Plot perceived obstacles in black
        for obs in perceived_sample:
            if obs not in true_obstacles:
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=0.5, edgecolor='black',
                                        facecolor='black', alpha=0.6)
                ax.add_patch(rect)
        
        # Mark start and goal
        ax.plot(5, 15, 'go', markersize=8, label='Start')
        ax.plot(45, 15, 'ro', markersize=8, label='Goal')
        
        # Mark narrow passages
        for x in [20, 30, 40]:
            ax.axvline(x, color='red', alpha=0.15, linestyle='--')
        
        # Add statistics text
        if path_lengths:
            avg_length = np.mean(path_lengths)
            avg_distance = np.mean(path_distances)
            avg_collisions = np.mean(collision_counts)
            
            stats_text = f'Avg Length: {avg_length:.1f}\n'
            stats_text += f'Avg Distance: {avg_distance:.1f}\n'
            stats_text += f'Collisions: {avg_collisions:.1f}'
            
            ax.text(2, 28, stats_text, fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                   verticalalignment='top')
            
            # Store for table
            results_table.append({
                'noise_std': noise_std,
                'avg_path_length': avg_length,
                'std_path_length': np.std(path_lengths),
                'avg_distance': avg_distance,
                'avg_collisions': avg_collisions,
                'num_successful': len(path_lengths)
            })
        
        if idx == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Path Planning with Different Noise Levels', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('discrete_planner/results', exist_ok=True)
    save_path = 'discrete_planner/results/step3_path_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPath comparison saved to: {save_path}")
    plt.close()
    
    return results_table


def print_results_table(results_table):
    """
    Print a formatted table of results.
    """
    print("\n" + "="*80)
    print("PATH LENGTH ANALYSIS TABLE")
    print("="*80)
    
    # Header
    print(f"{'Noise Level':^12} | {'Avg Path Length':^15} | {'Std Dev':^10} | {'Avg Distance':^12} | {'Collisions':^10} | {'Success Rate':^12}")
    print("-"*80)
    
    # Baseline for comparison
    baseline_length = None
    baseline_distance = None
    
    for result in results_table:
        noise = result['noise_std']
        avg_length = result['avg_path_length']
        std_length = result['std_path_length']
        avg_dist = result['avg_distance']
        collisions = result['avg_collisions']
        success = result['num_successful']
        
        # Store baseline
        if noise == 0:
            baseline_length = avg_length
            baseline_distance = avg_dist
        
        # Calculate percentage increase
        if baseline_length and noise > 0:
            length_increase = ((avg_length - baseline_length) / baseline_length) * 100
            dist_increase = ((avg_dist - baseline_distance) / baseline_distance) * 100
            increase_str = f" (+{length_increase:.1f}%)"
            dist_increase_str = f" (+{dist_increase:.1f}%)"
        else:
            increase_str = ""
            dist_increase_str = ""
        
        print(f"σ = {noise:^8.1f} | {avg_length:^15.1f}{increase_str:<8} | {std_length:^10.2f} | {avg_dist:^12.1f}{dist_increase_str:<8} | {collisions:^10.1f} | {success:^12}/10")
    
    print("-"*80)
    
    # Analysis
    print("\nKEY OBSERVATIONS:")
    
    if len(results_table) > 1:
        first_noise = results_table[1]['avg_path_length']
        last_noise = results_table[-1]['avg_path_length']
        
        if baseline_length:
            total_increase = ((last_noise - baseline_length) / baseline_length) * 100
            
            if total_increase > 0:
                print(f"✓ Path length INCREASES with noise level")
                print(f"  - Baseline: {baseline_length:.1f} cells")
                print(f"  - With max noise: {last_noise:.1f} cells")
                print(f"  - Total increase: {total_increase:.1f}%")
                print(f"\n  → This confirms walls are getting thicker!")
                print(f"  → Robot takes longer paths to avoid perceived obstacles")
            else:
                print(f"✗ Path length does not increase consistently")
    
    # Check collision trend
    collision_trend = [r['avg_collisions'] for r in results_table]
    if any(c > 0 for c in collision_trend):
        print(f"\n✓ Some collisions detected at higher noise levels")
    else:
        print(f"\n⚠ No collisions detected yet")
        print(f"  → Thicker walls make robot MORE cautious, not less")
        print(f"  → Need different noise model for collisions")


def analyze_passage_usage(true_env):
    """
    Analyze which passages the path uses at different noise levels.
    """
    print("\n" + "="*80)
    print("PASSAGE USAGE ANALYSIS")
    print("="*80)
    
    true_obstacles = true_env.obs
    noise_levels = [0, 0.2, 0.3, 0.4, 0.5]
    
    for noise_std in noise_levels:
        print(f"\nNoise σ = {noise_std}:")
        
        # Get path with this noise level
        if noise_std == 0:
            perceived = true_obstacles
        else:
            perceived = add_perception_noise_thickness(true_obstacles, noise_std, seed=42)
        
        s_start = (5, 15)
        s_goal = (45, 15)
        
        try:
            astar = Astar.AStar(s_start, s_goal, "euclidean")
            astar.obs = perceived
            path, _ = astar.searching()
            
            if path:
                # Check which passages are used
                passages_used = []
                for point in path:
                    x, y = point
                    if 19 <= x <= 21:
                        if y not in [p[1] for p in passages_used if p[0] == 'First']:
                            passages_used.append(('First', y))
                    elif 29 <= x <= 31:
                        if y not in [p[1] for p in passages_used if p[0] == 'Second']:
                            passages_used.append(('Second', y))
                    elif 39 <= x <= 41:
                        if y not in [p[1] for p in passages_used if p[0] == 'Third']:
                            passages_used.append(('Third', y))
                
                for passage, y_coord in passages_used:
                    print(f"  {passage} passage at y={y_coord}")
                
                # Check clearance at passages
                for x_check in [20, 30, 40]:
                    points_at_x = [(x, y) for (x, y) in path if x == x_check]
                    if points_at_x:
                        for px, py in points_at_x:
                            # Count free cells around
                            free_above = 0
                            free_below = 0
                            for dy in range(1, 5):
                                if (px, py + dy) not in perceived:
                                    free_above += 1
                                else:
                                    break
                            for dy in range(1, 5):
                                if (px, py - dy) not in perceived:
                                    free_below += 1
                                else:
                                    break
                            print(f"    At x={x_check}, y={py}: {free_below} cells free below, {free_above} cells free above")
        except:
            print("  Path planning failed!")


def main():
    """
    Main analysis of path behavior with noise.
    """
    print("="*80)
    print("STEP 3: PATH BEHAVIOR ANALYSIS WITH NOISE")
    print("="*80)
    
    # Get true environment
    true_env = env.Env()
    
    # Test different noise levels
    noise_levels = [0, 0.2, 0.3, 0.4, 0.5, 0.6]
    num_trials = 10
    
    print(f"\nConfiguration:")
    print(f"  Noise levels: {noise_levels}")
    print(f"  Trials per level: {num_trials}")
    print(f"  Start: (5, 15)")
    print(f"  Goal: (45, 15)")
    
    # Generate comparison visualization and get results
    print(f"\nGenerating path comparison visualization...")
    results_table = visualize_paths_comparison(true_env, noise_levels, num_trials)
    
    # Print results table
    print_results_table(results_table)
    
    # Analyze passage usage
    analyze_passage_usage(true_env)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    
    if results_table:
        baseline = results_table[0]['avg_path_length']
        highest_noise = results_table[-1]['avg_path_length']
        
        if highest_noise > baseline:
            print("\n✅ SUCCESS: Path length increases with noise!")
            print(f"   This confirms that perception noise affects path planning")
            print(f"   The robot avoids thicker walls by taking longer paths")
            print(f"\n   Next: We need a different strategy to create collisions")
            print(f"   Idea: Add phantom obstacles or make walls appear thinner")
        else:
            print("\n⚠ Path length not significantly affected")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    main()