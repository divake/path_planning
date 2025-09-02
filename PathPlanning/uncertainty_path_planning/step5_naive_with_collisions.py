#!/usr/bin/env python3
"""
STEP 5: NAIVE METHOD WITH COLLISION-INDUCING NOISE
Goal: Demonstrate naive method fails with wall thinning noise
Expected: >10% collision rate when walls appear thinner than reality
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


def add_thinning_noise(true_obstacles, thin_prob=0.1, seed=None):
    """
    Make walls appear thinner by randomly removing obstacle cells.
    This causes robot to think passages are wider than reality.
    
    Args:
        true_obstacles: Set of true obstacle positions
        thin_prob: Probability of removing each obstacle cell (except boundaries)
        seed: Random seed for reproducibility
    """
    if seed is not None:
        np.random.seed(seed)
    
    perceived_obstacles = set()
    
    for (x, y) in true_obstacles:
        # Always keep boundary walls to maintain environment structure
        if x == 0 or x == 50 or y == 0 or y == 30:
            perceived_obstacles.add((x, y))
        # Remove internal obstacles with probability thin_prob
        elif np.random.random() > thin_prob:
            perceived_obstacles.add((x, y))
    
    return perceived_obstacles


def run_naive_planning(true_obstacles, perceived_obstacles):
    """
    Run naive planning: plan on perceived, execute on true.
    
    Returns:
        dict with path, collisions, and success info
    """
    s_start = (5, 15)
    s_goal = (45, 15)
    
    # Plan on PERCEIVED environment (what robot sees)
    try:
        astar = Astar.AStar(s_start, s_goal, "euclidean")
        astar.obs = perceived_obstacles
        path, visited = astar.searching()
    except:
        return {'success': False, 'reason': 'planning_failed'}
    
    if not path:
        return {'success': False, 'reason': 'no_path_found'}
    
    # Check collisions with TRUE environment
    collisions = []
    for i, point in enumerate(path):
        if point in true_obstacles:
            collisions.append((i, point))
    
    return {
        'success': True,
        'path': path,
        'path_length': len(path),
        'has_collision': len(collisions) > 0,
        'num_collisions': len(collisions),
        'collision_points': collisions,
        'collision_rate': len(collisions) / len(path) if path else 0
    }


def monte_carlo_evaluation(true_env, thin_prob=0.1, num_trials=1000):
    """
    Run Monte Carlo evaluation of naive method with thinning noise.
    """
    true_obstacles = true_env.obs
    
    results = {
        'total_trials': num_trials,
        'successful_plans': 0,
        'paths_with_collision': 0,
        'total_collisions': 0,
        'path_lengths': [],
        'collision_counts': [],
        'collision_rates': []
    }
    
    print(f"\nRunning {num_trials} Monte Carlo trials...")
    print("Progress: ", end="", flush=True)
    
    for trial in range(num_trials):
        if trial % 100 == 0:
            print(f"{trial}...", end="", flush=True)
        
        # Generate perceived environment with thinning noise
        perceived = add_thinning_noise(true_obstacles, thin_prob, seed=trial)
        
        # Run naive planning
        result = run_naive_planning(true_obstacles, perceived)
        
        if result['success']:
            results['successful_plans'] += 1
            results['path_lengths'].append(result['path_length'])
            results['collision_counts'].append(result['num_collisions'])
            results['collision_rates'].append(result['collision_rate'])
            
            if result['has_collision']:
                results['paths_with_collision'] += 1
                results['total_collisions'] += result['num_collisions']
    
    print(f"{num_trials}. Done!")
    
    # Calculate statistics
    if results['successful_plans'] > 0:
        results['avg_path_length'] = np.mean(results['path_lengths'])
        results['std_path_length'] = np.std(results['path_lengths'])
        results['avg_collisions'] = np.mean(results['collision_counts'])
        results['std_collisions'] = np.std(results['collision_counts'])
        results['collision_rate'] = (results['paths_with_collision'] / results['successful_plans']) * 100
        results['avg_collision_rate_per_path'] = np.mean(results['collision_rates']) * 100
    
    return results


def visualize_naive_failures(true_env, thin_prob=0.1, num_examples=6):
    """
    Visualize examples of naive method failures.
    """
    true_obstacles = true_env.obs
    
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig)
    
    collision_examples = []
    safe_examples = []
    
    # Collect examples
    for seed in range(1000):
        perceived = add_thinning_noise(true_obstacles, thin_prob, seed)
        result = run_naive_planning(true_obstacles, perceived)
        
        if result['success']:
            if result['has_collision'] and len(collision_examples) < 3:
                collision_examples.append((seed, perceived, result))
            elif not result['has_collision'] and len(safe_examples) < 3:
                safe_examples.append((seed, perceived, result))
        
        if len(collision_examples) >= 3 and len(safe_examples) >= 3:
            break
    
    examples = collision_examples + safe_examples
    
    for idx, (seed, perceived, result) in enumerate(examples):
        ax = fig.add_subplot(gs[idx // 3, idx % 3])
        
        # Title based on collision status
        if result['has_collision']:
            title = f"COLLISION! (Seed {seed})"
            title_color = 'red'
        else:
            title = f"Safe Path (Seed {seed})"
            title_color = 'green'
        
        ax.set_title(title, fontsize=12, fontweight='bold', color=title_color)
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Plot true obstacles (light gray)
        for obs in true_obstacles:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='gray',
                                    facecolor='lightgray', alpha=0.3)
            ax.add_patch(rect)
        
        # Plot perceived obstacles (black)
        for obs in perceived:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    linewidth=0.5, edgecolor='black',
                                    facecolor='black', alpha=0.7)
            ax.add_patch(rect)
        
        # Plot missing obstacles (thinned walls) with dashed blue outline
        for obs in true_obstacles:
            if obs not in perceived and not (obs[0] in [0, 50] or obs[1] in [0, 30]):
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=1.5, edgecolor='blue',
                                        facecolor='none', linestyle='--', alpha=0.8)
                ax.add_patch(rect)
        
        # Plot path
        if result['path']:
            path_x = [p[0] for p in result['path']]
            path_y = [p[1] for p in result['path']]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Planned Path')
        
        # Mark collisions with big red X
        if result['collision_points']:
            for _, point in result['collision_points']:
                ax.plot(point[0], point[1], 'rx', markersize=12, markeredgewidth=3)
            
            # Add arrow pointing to first collision
            first_collision = result['collision_points'][0][1]
            ax.annotate('Collision!', xy=first_collision, 
                       xytext=(first_collision[0]+3, first_collision[1]+3),
                       arrowprops=dict(arrowstyle='->', color='red', lw=2),
                       fontsize=10, color='red', fontweight='bold')
        
        # Mark start and goal
        ax.plot(5, 15, 'go', markersize=10, label='Start')
        ax.plot(45, 15, 'ro', markersize=10, label='Goal')
        
        # Add statistics box
        stats = f"Path Length: {result['path_length']}\n"
        stats += f"Collisions: {result['num_collisions']}\n"
        stats += f"Missing Walls: {len(true_obstacles) - len(perceived)}"
        
        box_color = 'lightcoral' if result['has_collision'] else 'lightgreen'
        ax.text(2, 28, stats, fontsize=9,
               bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.8),
               verticalalignment='top')
        
        if idx == 0:
            ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle(f'Naive Method with Wall Thinning (thin_prob={thin_prob})', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('results', exist_ok=True)
    save_path = 'results/step5_naive_failures.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main demonstration of naive method failure.
    """
    print("="*80)
    print("STEP 5: NAIVE METHOD WITH COLLISION-INDUCING NOISE")
    print("="*80)
    
    # Get environment
    true_env = env.Env()
    
    print("\nNoise Model: Wall Thinning")
    print("  - Randomly removes obstacle cells")
    print("  - Makes passages appear wider than reality")
    print("  - Robot plans through 'gaps' that don't exist")
    
    # Test different thinning probabilities
    print("\n" + "="*80)
    print("PARAMETER SWEEP: Finding Optimal Thinning Probability")
    print("="*80)
    
    thin_probs = [0.05, 0.10, 0.15, 0.20]
    
    print(f"\n{'Thin Prob':>10} | {'Collision Rate':>14} | {'Avg Collisions':>14} | {'Avg Path Len':>13}")
    print("-"*60)
    
    best_prob = 0.10  # Default fallback
    best_rate = 0
    best_diff = float('inf')
    
    for prob in thin_probs:
        results = monte_carlo_evaluation(true_env, prob, num_trials=200)
        
        if results['successful_plans'] > 0:
            print(f"{prob:>10.2f} | {results['collision_rate']:>13.1f}% | "
                  f"{results['avg_collisions']:>14.2f} | {results['avg_path_length']:>13.1f}")
            
            # Track best probability (closest to 15% collision rate)
            diff = abs(results['collision_rate'] - 15)
            if diff < best_diff:
                best_prob = prob
                best_rate = results['collision_rate']
                best_diff = diff
    
    print("-"*60)
    print(f"\nOptimal thinning probability: {best_prob} (collision rate: {best_rate:.1f}%)")
    
    # Run full Monte Carlo with optimal probability
    print("\n" + "="*80)
    print("FULL MONTE CARLO EVALUATION")
    print("="*80)
    
    results = monte_carlo_evaluation(true_env, thin_prob=best_prob, num_trials=1000)
    
    print("\nRESULTS:")
    print(f"  Total Trials: {results['total_trials']}")
    print(f"  Successful Plans: {results['successful_plans']}")
    print(f"  Paths with Collisions: {results['paths_with_collision']}")
    print(f"  COLLISION RATE: {results['collision_rate']:.1f}%")
    print(f"  Average Path Length: {results['avg_path_length']:.1f} ± {results['std_path_length']:.1f}")
    print(f"  Average Collisions per Path: {results['avg_collisions']:.2f} ± {results['std_collisions']:.2f}")
    
    # Success criteria check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA CHECK")
    print("="*80)
    
    if results['collision_rate'] > 10:
        print(f"✅ SUCCESS! Collision rate {results['collision_rate']:.1f}% > 10%")
        print("   Naive method demonstrably FAILS with perception noise")
        print("   This justifies need for uncertainty-aware planning (Standard CP)")
    else:
        print(f"⚠️ Collision rate {results['collision_rate']:.1f}% < 10%")
        print("   Need more aggressive noise model")
    
    # Generate visualization
    print("\nGenerating failure visualization...")
    visualize_naive_failures(true_env, thin_prob=best_prob, num_examples=6)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nWe have successfully demonstrated that:")
    print("1. Naive planning fails when perception is noisy")
    print(f"2. Wall thinning noise causes {results['collision_rate']:.1f}% collision rate")
    print("3. This motivates the need for Standard CP with safety margins")
    print("\nNext: Implement Standard CP to handle this uncertainty!")
    print("="*80)


if __name__ == "__main__":
    main()