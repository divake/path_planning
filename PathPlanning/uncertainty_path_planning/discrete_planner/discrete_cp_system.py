#!/usr/bin/env python3
"""
DISCRETE PLANNER WITH CONFORMAL PREDICTION
Grid-based A* with integer τ values
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")

from Search_based_Planning.Search_2D import Astar
from Search_based_Planning.Search_2D import env


class DiscreteEnvironment:
    """
    Discrete grid environment (51x31)
    """
    def __init__(self):
        self.env = env.Env()
        self.obstacles = self.env.obs
        self.x_range = 51
        self.y_range = 31
        
    def add_discrete_noise(self, thin_prob=0.05, seed=None):
        """
        Add discrete noise - remove wall cells with probability
        """
        if seed is not None:
            np.random.seed(seed)
        
        perceived = set()
        for (x, y) in self.obstacles:
            # Keep boundaries always
            if x == 0 or x == 50 or y == 0 or y == 30:
                perceived.add((x, y))
            elif np.random.random() > thin_prob:
                perceived.add((x, y))
        
        return perceived


class DiscreteNonconformity:
    """
    Discrete nonconformity scoring
    Returns integer scores: 0, 1, 2, 3, etc.
    """
    @staticmethod
    def compute_score(true_obs, perceived_obs):
        """
        Discrete score based on missing obstacles
        Score = number of missing obstacle cells in critical zones
        """
        # Define critical zones (near passages)
        critical_zones = [
            (15, 25, 10, 20),  # First passage area
            (25, 35, 10, 20),  # Second passage area
            (35, 45, 10, 20),  # Third passage area
        ]
        
        max_missing = 0
        
        for x_min, x_max, y_min, y_max in critical_zones:
            missing_count = 0
            for x in range(x_min, x_max):
                for y in range(y_min, y_max):
                    point = (x, y)
                    # Count missing obstacles
                    if point in true_obs and point not in perceived_obs:
                        missing_count += 1
            
            # Take maximum missing count across zones
            max_missing = max(max_missing, missing_count)
        
        # Return discrete score (0, 1, 2, 3, ...)
        if max_missing == 0:
            return 0
        elif max_missing <= 2:
            return 1
        elif max_missing <= 5:
            return 2
        else:
            return 3


class DiscreteCP:
    """
    Conformal Prediction for discrete planner
    """
    def __init__(self, environment):
        self.env = environment
        self.tau = None
        self.scores = []
        
    def calibrate(self, num_samples=200, confidence=0.95, noise_prob=0.05):
        """
        Calibration with discrete scores
        """
        print(f"\nDISCRETE CALIBRATION ({num_samples} samples)")
        print("-" * 50)
        
        self.scores = []
        
        for i in range(num_samples):
            # Generate noisy perception
            perceived = self.env.add_discrete_noise(noise_prob, seed=i)
            
            # Compute discrete score
            score = DiscreteNonconformity.compute_score(
                self.env.obstacles, perceived
            )
            self.scores.append(score)
        
        # Compute discrete τ
        sorted_scores = sorted(self.scores)
        quantile_idx = int(np.ceil((num_samples + 1) * confidence)) - 1
        quantile_idx = min(quantile_idx, num_samples - 1)
        
        self.tau = sorted_scores[quantile_idx]
        
        # Print statistics
        unique_scores = sorted(set(self.scores))
        print(f"Unique scores: {unique_scores}")
        print(f"Score distribution:")
        for s in unique_scores:
            count = self.scores.count(s)
            print(f"  Score {s}: {count} samples ({count/num_samples*100:.1f}%)")
        
        print(f"\nDISCRETE τ = {self.tau} (integer cells)")
        print(f"This means: inflate obstacles by {self.tau} cells")
        
        return self.tau
    
    def inflate_obstacles(self, perceived_obs):
        """
        Inflate by integer number of cells
        """
        if self.tau is None:
            raise ValueError("Must calibrate first!")
        
        inflated = set(perceived_obs)
        
        # Inflate by τ cells (integer)
        inflation_radius = int(self.tau)
        
        for (x, y) in list(perceived_obs):
            for dx in range(-inflation_radius, inflation_radius + 1):
                for dy in range(-inflation_radius, inflation_radius + 1):
                    new_x, new_y = x + dx, y + dy
                    if 0 <= new_x < 51 and 0 <= new_y < 31:
                        inflated.add((new_x, new_y))
        
        return inflated
    
    def plan_safe_path(self, perceived_obs, start=(5, 15), goal=(45, 15)):
        """
        Plan with discrete obstacle inflation
        """
        # Inflate obstacles
        inflated_obs = self.inflate_obstacles(perceived_obs)
        
        # Plan with A*
        try:
            astar = Astar.AStar(start, goal, "euclidean")
            astar.obs = inflated_obs
            path, _ = astar.searching()
            
            if path:
                # Check for collisions with true obstacles
                collisions = [p for p in path if p in self.env.obstacles]
                
                return {
                    'success': True,
                    'path': path,
                    'path_length': len(path),
                    'collisions': collisions,
                    'inflated_obs': inflated_obs
                }
        except:
            pass
        
        return {'success': False}


def run_discrete_monte_carlo(num_trials=1000):
    """
    Monte Carlo evaluation for discrete system
    """
    print("\n" + "="*60)
    print("DISCRETE PLANNER MONTE CARLO EVALUATION")
    print("="*60)
    
    # Setup
    env = DiscreteEnvironment()
    cp = DiscreteCP(env)
    
    # Calibrate
    tau = cp.calibrate(num_samples=200, confidence=0.95, noise_prob=0.05)
    
    # Run trials
    results = {
        'naive_collisions': 0,
        'cp_collisions': 0,
        'naive_lengths': [],
        'cp_lengths': []
    }
    
    print(f"\nRunning {num_trials} trials...")
    
    for trial in range(num_trials):
        if trial % 100 == 0:
            print(f"  Trial {trial}...")
        
        # Generate noisy perception
        perceived = env.add_discrete_noise(0.05, seed=trial+1000)
        
        # Naive planning
        try:
            astar = Astar.AStar((5, 15), (45, 15), "euclidean")
            astar.obs = perceived
            naive_path, _ = astar.searching()
            
            if naive_path:
                results['naive_lengths'].append(len(naive_path))
                collisions = [p for p in naive_path if p in env.obstacles]
                if collisions:
                    results['naive_collisions'] += 1
        except:
            pass
        
        # CP planning
        cp_result = cp.plan_safe_path(perceived)
        if cp_result['success']:
            results['cp_lengths'].append(cp_result['path_length'])
            if cp_result['collisions']:
                results['cp_collisions'] += 1
    
    # Print results
    print("\n" + "="*60)
    print("DISCRETE RESULTS SUMMARY")
    print("="*60)
    
    naive_rate = (results['naive_collisions'] / num_trials) * 100
    cp_rate = (results['cp_collisions'] / num_trials) * 100
    
    print(f"\nCollision Rates:")
    print(f"  Naive:       {naive_rate:.1f}%")
    print(f"  Discrete CP: {cp_rate:.1f}%")
    
    if results['naive_lengths'] and results['cp_lengths']:
        print(f"\nAverage Path Lengths:")
        print(f"  Naive:       {np.mean(results['naive_lengths']):.1f} cells")
        print(f"  Discrete CP: {np.mean(results['cp_lengths']):.1f} cells")
    
    print(f"\nKey Observation:")
    print(f"  τ = {tau} (integer cells)")
    print(f"  Can only inflate by whole cells")
    print(f"  Coverage jumps: ~85% (τ=1) or ~99% (τ=2)")
    
    return results


def visualize_discrete_system():
    """
    Visualize discrete CP system
    """
    env = DiscreteEnvironment()
    cp = DiscreteCP(env)
    cp.calibrate(num_samples=100, confidence=0.95)
    
    # Get example perception
    perceived = env.add_discrete_noise(0.05, seed=42)
    
    # Plan with CP
    result = cp.plan_safe_path(perceived)
    
    if result['success']:
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot 1: Perceived environment
        ax = axes[0]
        ax.set_title(f'Discrete Perception', fontsize=12)
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        for obs in perceived:
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    facecolor='gray', edgecolor='black', alpha=0.5)
            ax.add_patch(rect)
        
        ax.plot(5, 15, 'go', markersize=8)
        ax.plot(45, 15, 'ro', markersize=8)
        
        # Plot 2: With inflation
        ax = axes[1]
        ax.set_title(f'Discrete CP (τ={cp.tau} cells)', fontsize=12)
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2)
        
        # Show inflated obstacles
        for obs in result['inflated_obs']:
            if obs in perceived:
                color = 'gray'
                alpha = 0.5
            else:
                color = 'orange'
                alpha = 0.3
            rect = patches.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1,
                                    facecolor=color, edgecolor='black', alpha=alpha)
            ax.add_patch(rect)
        
        # Plot path
        if result['path']:
            path_x = [p[0] for p in result['path']]
            path_y = [p[1] for p in result['path']]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        
        ax.plot(5, 15, 'go', markersize=8)
        ax.plot(45, 15, 'ro', markersize=8)
        
        plt.suptitle('DISCRETE PLANNER (Grid-based A* with Integer τ)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = 'discrete_planner/results/discrete_system.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()


if __name__ == "__main__":
    # Run discrete system
    visualize_discrete_system()
    results = run_discrete_monte_carlo(num_trials=500)
    print("\n" + "="*60)
    print("DISCRETE PLANNER COMPLETE")
    print("="*60)