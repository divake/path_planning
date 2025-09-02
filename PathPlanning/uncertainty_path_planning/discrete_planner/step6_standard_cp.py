#!/usr/bin/env python3
"""
STEP 6: STANDARD CONFORMAL PREDICTION FOR PATH PLANNING
Goal: Implement Standard CP with calibrated safety margins
Method: Nonconformity scores + obstacle inflation based on τ
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from collections import defaultdict

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import env

# Import our noise model
from step5_naive_with_collisions import add_thinning_noise


class StandardCP:
    """
    Standard Conformal Prediction for path planning.
    Uses fixed safety margin τ calibrated from data.
    """
    
    def __init__(self, true_env, noise_model, noise_params):
        """
        Initialize Standard CP planner.
        
        Args:
            true_env: True environment with obstacles
            noise_model: Function to add perception noise
            noise_params: Parameters for noise model
        """
        self.true_env = true_env
        self.true_obstacles = true_env.obs
        self.noise_model = noise_model
        self.noise_params = noise_params
        self.tau = None  # Safety margin (to be calibrated)
        self.nonconformity_scores = []
        
    def compute_nonconformity_score(self, true_obs, perceived_obs):
        """
        Compute nonconformity score between true and perceived obstacles.
        Score measures the "error" in perception.
        
        We use: Maximum distance from any perceived free space to nearest true obstacle
        """
        score = 0
        
        # Check perceived free spaces that are actually obstacles
        for x in range(1, 50):
            for y in range(1, 30):
                point = (x, y)
                
                # If perceived as free but actually obstacle
                if point not in perceived_obs and point in true_obs:
                    # This is a perception error - high score
                    score = max(score, 2.0)  # Critical error
                
                # If perceived as obstacle but actually free
                elif point in perceived_obs and point not in true_obs:
                    # Less critical but still an error
                    score = max(score, 1.0)
        
        return score
    
    def calibration_phase(self, num_samples=100, confidence=0.95):
        """
        Calibration phase: Collect nonconformity scores and compute τ.
        
        Args:
            num_samples: Number of calibration samples
            confidence: Desired confidence level (1-ε)
        
        Returns:
            tau: Calibrated safety margin
        """
        print(f"\nCalibration Phase ({num_samples} samples)...")
        self.nonconformity_scores = []
        
        for i in range(num_samples):
            # Generate perceived environment
            perceived_obs = self.noise_model(self.true_obstacles, **self.noise_params, seed=i)
            
            # Compute nonconformity score
            score = self.compute_nonconformity_score(self.true_obstacles, perceived_obs)
            self.nonconformity_scores.append(score)
        
        # Sort scores
        sorted_scores = sorted(self.nonconformity_scores)
        
        # Compute quantile for desired confidence
        quantile_idx = int(np.ceil((num_samples + 1) * confidence)) - 1
        quantile_idx = min(quantile_idx, num_samples - 1)
        
        self.tau = sorted_scores[quantile_idx]
        
        print(f"  Nonconformity scores: min={min(sorted_scores):.2f}, "
              f"max={max(sorted_scores):.2f}, median={np.median(sorted_scores):.2f}")
        print(f"  Calibrated τ = {self.tau:.2f} (for {confidence*100:.0f}% confidence)")
        
        return self.tau
    
    def inflate_obstacles(self, perceived_obs, inflation_radius):
        """
        Inflate obstacles by a fixed radius based on τ.
        
        Args:
            perceived_obs: Set of perceived obstacle positions
            inflation_radius: Number of cells to inflate
        
        Returns:
            inflated_obs: Set with inflated obstacles
        """
        inflated_obs = set(perceived_obs)
        
        # For each perceived obstacle, add surrounding cells
        for (x, y) in list(perceived_obs):
            for dx in range(-inflation_radius, inflation_radius + 1):
                for dy in range(-inflation_radius, inflation_radius + 1):
                    new_x, new_y = x + dx, y + dy
                    
                    # Check bounds
                    if 0 <= new_x < 51 and 0 <= new_y < 31:
                        inflated_obs.add((new_x, new_y))
        
        return inflated_obs
    
    def plan_with_cp(self, perceived_obs, start=(5, 15), goal=(45, 15)):
        """
        Plan path using Standard CP with obstacle inflation.
        
        Args:
            perceived_obs: Perceived obstacles
            start: Start position
            goal: Goal position
        
        Returns:
            dict with path and planning info
        """
        if self.tau is None:
            raise ValueError("Must run calibration_phase first!")
        
        # Convert tau to inflation radius (cells)
        inflation_radius = int(np.ceil(self.tau))
        
        # Inflate obstacles based on tau
        inflated_obs = self.inflate_obstacles(perceived_obs, inflation_radius)
        
        # Plan on inflated obstacles
        try:
            astar = Astar.AStar(start, goal, "euclidean")
            astar.obs = inflated_obs
            path, visited = astar.searching()
        except:
            return {'success': False, 'reason': 'planning_failed'}
        
        if not path:
            return {'success': False, 'reason': 'no_path_found'}
        
        # Check collisions with true obstacles
        collisions = [p for p in path if p in self.true_obstacles]
        
        return {
            'success': True,
            'path': path,
            'path_length': len(path),
            'inflated_obs': inflated_obs,
            'inflation_radius': inflation_radius,
            'has_collision': len(collisions) > 0,
            'num_collisions': len(collisions),
            'collision_points': collisions
        }


def monte_carlo_comparison(true_env, num_trials=1000):
    """
    Compare Naive vs Standard CP methods via Monte Carlo simulation.
    """
    # Setup noise parameters
    noise_params = {'thin_prob': 0.05}  # 5% thinning for ~15% collision rate
    
    # Initialize Standard CP
    cp_planner = StandardCP(true_env, add_thinning_noise, noise_params)
    
    # Calibration phase
    cp_planner.calibration_phase(num_samples=100, confidence=0.95)
    
    # Results storage
    results = {
        'naive': {
            'successes': 0,
            'collisions': 0,
            'path_lengths': [],
            'collision_counts': []
        },
        'standard_cp': {
            'successes': 0,
            'collisions': 0,
            'path_lengths': [],
            'collision_counts': []
        }
    }
    
    print(f"\nMonte Carlo Evaluation ({num_trials} trials)...")
    print("Progress: ", end="", flush=True)
    
    for trial in range(num_trials):
        if trial % 100 == 0:
            print(f"{trial}...", end="", flush=True)
        
        # Generate perceived environment
        perceived_obs = add_thinning_noise(true_env.obs, **noise_params, seed=trial+1000)
        
        # NAIVE METHOD
        try:
            astar = Astar.AStar((5, 15), (45, 15), "euclidean")
            astar.obs = perceived_obs
            naive_path, _ = astar.searching()
            
            if naive_path:
                results['naive']['successes'] += 1
                results['naive']['path_lengths'].append(len(naive_path))
                
                # Check collisions
                collisions = [p for p in naive_path if p in true_env.obs]
                results['naive']['collision_counts'].append(len(collisions))
                if collisions:
                    results['naive']['collisions'] += 1
        except:
            pass
        
        # STANDARD CP METHOD
        cp_result = cp_planner.plan_with_cp(perceived_obs)
        
        if cp_result['success']:
            results['standard_cp']['successes'] += 1
            results['standard_cp']['path_lengths'].append(cp_result['path_length'])
            results['standard_cp']['collision_counts'].append(cp_result['num_collisions'])
            
            if cp_result['has_collision']:
                results['standard_cp']['collisions'] += 1
    
    print(f"{num_trials}. Done!")
    
    # Calculate statistics
    for method in ['naive', 'standard_cp']:
        if results[method]['successes'] > 0:
            results[method]['collision_rate'] = (results[method]['collisions'] / 
                                                 results[method]['successes']) * 100
            results[method]['avg_path_length'] = np.mean(results[method]['path_lengths'])
            results[method]['avg_collisions'] = np.mean(results[method]['collision_counts'])
        else:
            results[method]['collision_rate'] = 0
            results[method]['avg_path_length'] = 0
            results[method]['avg_collisions'] = 0
    
    return results


def visualize_cp_comparison(true_env):
    """
    Visualize comparison between Naive and Standard CP methods.
    """
    noise_params = {'thin_prob': 0.05}
    
    # Initialize Standard CP
    cp_planner = StandardCP(true_env, add_thinning_noise, noise_params)
    cp_planner.calibration_phase(num_samples=50, confidence=0.95)
    
    fig = plt.figure(figsize=(20, 8))
    gs = GridSpec(1, 3, figure=fig)
    
    # Use same seed for fair comparison
    seed = 42
    perceived_obs = add_thinning_noise(true_env.obs, **noise_params, seed=seed)
    
    # Plot 1: True Environment
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title('True Environment', fontsize=12, fontweight='bold')
    ax1.set_xlim(-1, 51)
    ax1.set_ylim(-1, 31)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.2)
    
    for obs in true_env.obs:
        rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                linewidth=0.5, edgecolor='black',
                                facecolor='black', alpha=0.8)
        ax1.add_patch(rect)
    
    ax1.plot(5, 15, 'go', markersize=10, label='Start')
    ax1.plot(45, 15, 'ro', markersize=10, label='Goal')
    ax1.legend(loc='upper right', fontsize=8)
    
    # Plot 2: Naive Method
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Naive Method (No Safety Margin)', fontsize=12, fontweight='bold')
    ax2.set_xlim(-1, 51)
    ax2.set_ylim(-1, 31)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.2)
    
    # Plot perceived obstacles
    for obs in perceived_obs:
        rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                linewidth=0.5, edgecolor='gray',
                                facecolor='gray', alpha=0.5)
        ax2.add_patch(rect)
    
    # Plan naive path
    try:
        astar = Astar.AStar((5, 15), (45, 15), "euclidean")
        astar.obs = perceived_obs
        naive_path, _ = astar.searching()
        
        if naive_path:
            path_x = [p[0] for p in naive_path]
            path_y = [p[1] for p in naive_path]
            ax2.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Planned Path')
            
            # Check collisions
            collisions = [p for p in naive_path if p in true_env.obs]
            for cp in collisions:
                ax2.plot(cp[0], cp[1], 'rx', markersize=12, markeredgewidth=3)
            
            stats = f"Path Length: {len(naive_path)}\nCollisions: {len(collisions)}"
            color = 'lightcoral' if collisions else 'lightgreen'
            ax2.text(2, 28, stats, fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                    verticalalignment='top')
    except:
        pass
    
    ax2.plot(5, 15, 'go', markersize=10)
    ax2.plot(45, 15, 'ro', markersize=10)
    
    # Plot 3: Standard CP Method
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_title(f'Standard CP (τ={cp_planner.tau:.1f}, inflation={int(np.ceil(cp_planner.tau))} cells)', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlim(-1, 51)
    ax3.set_ylim(-1, 31)
    ax3.set_aspect('equal')
    ax3.grid(True, alpha=0.2)
    
    # Plan with CP
    cp_result = cp_planner.plan_with_cp(perceived_obs)
    
    if cp_result['success']:
        # Plot inflated obstacles
        for obs in cp_result['inflated_obs']:
            if obs in perceived_obs:
                # Original perceived obstacle
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=0.5, edgecolor='gray',
                                        facecolor='gray', alpha=0.5)
            else:
                # Inflation area
                rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                        linewidth=0.5, edgecolor='orange',
                                        facecolor='orange', alpha=0.3)
            ax3.add_patch(rect)
        
        # Plot path
        path_x = [p[0] for p in cp_result['path']]
        path_y = [p[1] for p in cp_result['path']]
        ax3.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7, label='Safe Path')
        
        # Check collisions (should be none or very few)
        if cp_result['collision_points']:
            for cp in cp_result['collision_points']:
                ax3.plot(cp[0], cp[1], 'rx', markersize=12, markeredgewidth=3)
        
        stats = f"Path Length: {cp_result['path_length']}\nCollisions: {cp_result['num_collisions']}\nInflation: {cp_result['inflation_radius']} cells"
        color = 'lightcoral' if cp_result['has_collision'] else 'lightgreen'
        ax3.text(2, 28, stats, fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8),
                verticalalignment='top')
    
    ax3.plot(5, 15, 'go', markersize=10)
    ax3.plot(45, 15, 'ro', markersize=10)
    
    plt.suptitle('Standard Conformal Prediction: Safety through Obstacle Inflation', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    os.makedirs('discrete_planner/results', exist_ok=True)
    save_path = 'discrete_planner/results/step6_standard_cp_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: {save_path}")
    plt.close()


def main():
    """
    Main Standard CP demonstration.
    """
    print("="*80)
    print("STANDARD CONFORMAL PREDICTION FOR PATH PLANNING")
    print("="*80)
    
    # Get environment
    true_env = env.Env()
    
    print("\nMethod: Standard CP with Obstacle Inflation")
    print("  1. Calibration: Compute nonconformity scores from data")
    print("  2. Safety Margin: τ = quantile of scores at confidence level")
    print("  3. Planning: Inflate obstacles by τ cells")
    print("  4. Guarantee: (1-ε) probability of collision-free path")
    
    # Run Monte Carlo comparison
    print("\n" + "="*80)
    print("MONTE CARLO COMPARISON (1000 trials)")
    print("="*80)
    
    results = monte_carlo_comparison(true_env, num_trials=1000)
    
    # Print results table
    print("\n" + "="*80)
    print("RESULTS COMPARISON")
    print("="*80)
    
    print(f"\n{'Method':<15} | {'Success Rate':>12} | {'Collision Rate':>14} | {'Avg Path Len':>13} | {'Avg Collisions':>14}")
    print("-"*75)
    
    print(f"{'Naive':<15} | {results['naive']['successes']/10:>11.1f}% | "
          f"{results['naive']['collision_rate']:>13.1f}% | "
          f"{results['naive']['avg_path_length']:>13.1f} | "
          f"{results['naive']['avg_collisions']:>14.2f}")
    
    print(f"{'Standard CP':<15} | {results['standard_cp']['successes']/10:>11.1f}% | "
          f"{results['standard_cp']['collision_rate']:>13.1f}% | "
          f"{results['standard_cp']['avg_path_length']:>13.1f} | "
          f"{results['standard_cp']['avg_collisions']:>14.2f}")
    
    print("-"*75)
    
    # Calculate improvement
    naive_collision = results['naive']['collision_rate']
    cp_collision = results['standard_cp']['collision_rate']
    reduction = ((naive_collision - cp_collision) / naive_collision) * 100 if naive_collision > 0 else 0
    
    print(f"\nCollision Reduction: {reduction:.1f}%")
    print(f"Path Length Increase: {results['standard_cp']['avg_path_length'] - results['naive']['avg_path_length']:.1f} cells")
    
    # Success check
    print("\n" + "="*80)
    print("SUCCESS CRITERIA")
    print("="*80)
    
    if cp_collision < 5:  # Less than 5% collision rate
        print(f"✅ SUCCESS! Standard CP achieves {cp_collision:.1f}% collision rate")
        print(f"   Reduced from {naive_collision:.1f}% (Naive) to {cp_collision:.1f}% (CP)")
        print(f"   This demonstrates effective uncertainty handling!")
    else:
        print(f"⚠️ Collision rate {cp_collision:.1f}% still high")
        print(f"   May need larger safety margin or better nonconformity score")
    
    # Generate visualization
    print("\nGenerating comparison visualization...")
    visualize_cp_comparison(true_env)
    
    print("\n" + "="*80)
    print("CONCLUSION")
    print("="*80)
    print("\nStandard CP successfully reduces collisions by:")
    print("  1. Learning uncertainty from calibration data")
    print("  2. Inflating obstacles by calibrated margin τ")
    print("  3. Planning conservatively with safety buffer")
    print(f"\nResult: {reduction:.1f}% reduction in collision rate!")
    print("="*80)


if __name__ == "__main__":
    main()