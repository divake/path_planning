#!/usr/bin/env python3
"""
CONTINUOUS PLANNER WITH CONFORMAL PREDICTION
RRT* with continuous τ values
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import random
import math

# Add parent directories to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../../")


class ContinuousEnvironment:
    """
    Continuous space environment (50.0 x 30.0 units)
    Obstacles represented as continuous rectangles
    """
    def __init__(self):
        self.x_range = 50.0
        self.y_range = 30.0
        
        # Define obstacles as continuous rectangles (x, y, width, height)
        self.obstacles = [
            # Boundaries
            (0, 0, 50, 1),      # Bottom
            (0, 29, 50, 1),     # Top
            (0, 0, 1, 30),      # Left
            (49, 0, 1, 30),     # Right
            
            # Internal walls (same layout as discrete)
            (10, 14.5, 10, 1),  # Horizontal wall at y=15
            (20, 0, 1, 15),     # Vertical wall from bottom
            (30, 15, 1, 15),    # Vertical wall from middle to top
            (40, 0, 1, 16),     # Vertical wall from bottom
        ]
        
    def point_in_obstacle(self, x, y):
        """
        Check if point (x,y) is inside any obstacle
        """
        for (ox, oy, width, height) in self.obstacles:
            if ox <= x <= ox + width and oy <= y <= oy + height:
                return True
        return False
    
    def add_continuous_noise(self, noise_std=0.3, seed=None):
        """
        Add continuous noise - obstacles shrink/expand by continuous amounts
        Returns modified obstacle list
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        
        perceived = []
        
        for (x, y, w, h) in self.obstacles:
            # Add continuous noise to dimensions
            # Negative noise = thinning (dangerous!)
            # Positive noise = thickening (safe but inefficient)
            
            # Keep boundaries unchanged
            if x == 0 or y == 0 or x == 49 or y == 29:
                perceived.append((x, y, w, h))
            else:
                # Apply thinning noise (shrink obstacles)
                shrink = np.random.uniform(0, noise_std)
                
                # Shrink from all sides
                new_x = x + shrink/2
                new_y = y + shrink/2
                new_w = max(0.1, w - shrink)  # Keep minimum width
                new_h = max(0.1, h - shrink)  # Keep minimum height
                
                perceived.append((new_x, new_y, new_w, new_h))
        
        return perceived


class ContinuousNonconformity:
    """
    Continuous nonconformity scoring
    Returns float values: 0.0, 0.3, 1.2, 1.7, etc.
    """
    @staticmethod
    def compute_score(true_obs, perceived_obs):
        """
        Continuous score based on maximum penetration depth
        """
        max_penetration = 0.0
        
        # Sample points in space
        for _ in range(500):
            x = np.random.uniform(1, 49)
            y = np.random.uniform(1, 29)
            
            # Check if perceived as free but actually occupied
            perceived_free = not any(
                ox <= x <= ox+w and oy <= y <= oy+h 
                for (ox, oy, w, h) in perceived_obs
            )
            
            true_occupied = any(
                ox <= x <= ox+w and oy <= y <= oy+h 
                for (ox, oy, w, h) in true_obs
            )
            
            if perceived_free and true_occupied:
                # Measure penetration depth (distance into obstacle)
                # Find nearest edge of true obstacle
                min_dist = float('inf')
                
                for (ox, oy, w, h) in true_obs:
                    if ox <= x <= ox+w and oy <= y <= oy+h:
                        # Point is inside this obstacle
                        # Find distance to nearest edge
                        dist_to_left = x - ox
                        dist_to_right = (ox + w) - x
                        dist_to_bottom = y - oy
                        dist_to_top = (oy + h) - y
                        
                        edge_dist = min(dist_to_left, dist_to_right, 
                                      dist_to_bottom, dist_to_top)
                        min_dist = min(min_dist, edge_dist)
                
                max_penetration = max(max_penetration, min_dist)
        
        return round(max_penetration, 2)  # Return continuous score


class RRTStar:
    """
    RRT* planner for continuous space
    """
    def __init__(self, start, goal, obstacles, max_iter=2000):
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.max_iter = max_iter
        self.step_size = 1.0
        self.goal_sample_rate = 0.1
        self.search_radius = 2.0
        
    def plan(self):
        """
        Simple RRT* planning
        """
        nodes = [self.start]
        parents = {0: None}
        costs = {0: 0}
        
        for i in range(self.max_iter):
            # Sample random point
            if random.random() < self.goal_sample_rate:
                sample = self.goal
            else:
                sample = (random.uniform(1, 49), random.uniform(1, 29))
            
            # Find nearest node
            nearest_idx = self.nearest_node(nodes, sample)
            nearest = nodes[nearest_idx]
            
            # Steer towards sample
            new_node = self.steer(nearest, sample)
            
            # Check collision
            if not self.collision_check(nearest, new_node):
                # Find nearby nodes for rewiring
                near_indices = self.near_nodes(nodes, new_node)
                
                # Choose best parent
                min_cost = costs[nearest_idx] + self.distance(nearest, new_node)
                min_idx = nearest_idx
                
                for idx in near_indices:
                    if not self.collision_check(nodes[idx], new_node):
                        cost = costs[idx] + self.distance(nodes[idx], new_node)
                        if cost < min_cost:
                            min_cost = cost
                            min_idx = idx
                
                # Add node
                nodes.append(new_node)
                new_idx = len(nodes) - 1
                parents[new_idx] = min_idx
                costs[new_idx] = min_cost
                
                # Rewire tree
                for idx in near_indices:
                    if idx != min_idx:
                        cost = costs[new_idx] + self.distance(new_node, nodes[idx])
                        if cost < costs[idx] and not self.collision_check(new_node, nodes[idx]):
                            parents[idx] = new_idx
                            costs[idx] = cost
                
                # Check if reached goal
                if self.distance(new_node, self.goal) < 1.0:
                    # Extract path
                    path = []
                    current = new_idx
                    while current is not None:
                        path.append(nodes[current])
                        current = parents.get(current)
                    return path[::-1]
        
        return None
    
    def nearest_node(self, nodes, point):
        """Find nearest node to point"""
        distances = [self.distance(node, point) for node in nodes]
        return np.argmin(distances)
    
    def near_nodes(self, nodes, point):
        """Find nodes within search radius"""
        indices = []
        for i, node in enumerate(nodes):
            if self.distance(node, point) < self.search_radius:
                indices.append(i)
        return indices
    
    def steer(self, from_node, to_point):
        """Steer from node towards point"""
        dist = self.distance(from_node, to_point)
        if dist < self.step_size:
            return to_point
        
        ratio = self.step_size / dist
        x = from_node[0] + ratio * (to_point[0] - from_node[0])
        y = from_node[1] + ratio * (to_point[1] - from_node[1])
        return (x, y)
    
    def collision_check(self, from_node, to_node):
        """Check if path from_node to to_node collides"""
        # Sample along the line
        steps = int(self.distance(from_node, to_node) / 0.1) + 1
        for i in range(steps):
            t = i / float(steps)
            x = from_node[0] + t * (to_node[0] - from_node[0])
            y = from_node[1] + t * (to_node[1] - from_node[1])
            
            for (ox, oy, w, h) in self.obstacles:
                if ox <= x <= ox+w and oy <= y <= oy+h:
                    return True
        return False
    
    def distance(self, p1, p2):
        """Euclidean distance"""
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)


class ContinuousCP:
    """
    Conformal Prediction for continuous planner
    """
    def __init__(self, environment):
        self.env = environment
        self.tau = None
        self.scores = []
        
    def calibrate(self, num_samples=200, confidence=0.95, noise_std=0.3):
        """
        Calibration with continuous scores
        """
        print(f"\nCONTINUOUS CALIBRATION ({num_samples} samples)")
        print("-" * 50)
        
        self.scores = []
        
        for i in range(num_samples):
            # Generate noisy perception
            perceived = self.env.add_continuous_noise(noise_std, seed=i)
            
            # Compute continuous score
            score = ContinuousNonconformity.compute_score(
                self.env.obstacles, perceived
            )
            self.scores.append(score)
        
        # Compute continuous τ
        sorted_scores = sorted(self.scores)
        quantile_idx = int(np.ceil((num_samples + 1) * confidence)) - 1
        quantile_idx = min(quantile_idx, num_samples - 1)
        
        self.tau = sorted_scores[quantile_idx]
        
        # Print statistics
        print(f"Score range: [{min(self.scores):.2f}, {max(self.scores):.2f}]")
        print(f"Score distribution:")
        print(f"  Min:    {min(self.scores):.2f}")
        print(f"  25th:   {np.percentile(self.scores, 25):.2f}")
        print(f"  Median: {np.median(self.scores):.2f}")
        print(f"  75th:   {np.percentile(self.scores, 75):.2f}")
        print(f"  Max:    {max(self.scores):.2f}")
        
        print(f"\nCONTINUOUS τ = {self.tau:.2f} (continuous units)")
        print(f"This means: inflate obstacles by {self.tau:.2f} units")
        
        # Show how τ varies with confidence
        print(f"\nτ for different confidence levels:")
        for conf in [0.80, 0.85, 0.90, 0.95, 0.99]:
            idx = int(np.ceil((num_samples + 1) * conf)) - 1
            idx = min(idx, num_samples - 1)
            tau_conf = sorted_scores[idx]
            print(f"  {conf*100:.0f}% confidence → τ = {tau_conf:.2f}")
        
        return self.tau
    
    def inflate_obstacles(self, perceived_obs):
        """
        Inflate by continuous amount
        """
        if self.tau is None:
            raise ValueError("Must calibrate first!")
        
        inflated = []
        
        for (x, y, w, h) in perceived_obs:
            # Inflate by τ units (continuous)
            new_x = max(0, x - self.tau)
            new_y = max(0, y - self.tau)
            new_w = min(50 - new_x, w + 2*self.tau)
            new_h = min(30 - new_y, h + 2*self.tau)
            
            inflated.append((new_x, new_y, new_w, new_h))
        
        return inflated
    
    def plan_safe_path(self, perceived_obs, start=(5, 15), goal=(45, 15)):
        """
        Plan with continuous obstacle inflation
        """
        # Inflate obstacles
        inflated_obs = self.inflate_obstacles(perceived_obs)
        
        # Plan with RRT*
        planner = RRTStar(start, goal, inflated_obs)
        path = planner.plan()
        
        if path:
            # Check for collisions with true obstacles
            collisions = []
            for point in path:
                if self.env.point_in_obstacle(point[0], point[1]):
                    collisions.append(point)
            
            # Calculate path length
            path_length = 0
            for i in range(len(path)-1):
                path_length += math.sqrt(
                    (path[i+1][0] - path[i][0])**2 + 
                    (path[i+1][1] - path[i][1])**2
                )
            
            return {
                'success': True,
                'path': path,
                'path_length': path_length,
                'collisions': collisions,
                'inflated_obs': inflated_obs
            }
        
        return {'success': False}


def run_continuous_monte_carlo(num_trials=500):
    """
    Monte Carlo evaluation for continuous system
    """
    print("\n" + "="*60)
    print("CONTINUOUS PLANNER MONTE CARLO EVALUATION")
    print("="*60)
    
    # Setup
    env = ContinuousEnvironment()
    cp = ContinuousCP(env)
    
    # Calibrate
    tau = cp.calibrate(num_samples=200, confidence=0.95, noise_std=0.3)
    
    # Run trials
    results = {
        'naive_collisions': 0,
        'cp_collisions': 0,
        'naive_lengths': [],
        'cp_lengths': []
    }
    
    print(f"\nRunning {num_trials} trials...")
    
    for trial in range(num_trials):
        if trial % 50 == 0:
            print(f"  Trial {trial}...")
        
        # Generate noisy perception
        perceived = env.add_continuous_noise(0.3, seed=trial+1000)
        
        # Naive planning
        planner = RRTStar((5, 15), (45, 15), perceived, max_iter=500)
        naive_path = planner.plan()
        
        if naive_path:
            # Calculate path length
            length = 0
            for i in range(len(naive_path)-1):
                length += math.sqrt(
                    (naive_path[i+1][0] - naive_path[i][0])**2 + 
                    (naive_path[i+1][1] - naive_path[i][1])**2
                )
            results['naive_lengths'].append(length)
            
            # Check collisions
            has_collision = False
            for point in naive_path:
                if env.point_in_obstacle(point[0], point[1]):
                    has_collision = True
                    break
            if has_collision:
                results['naive_collisions'] += 1
        
        # CP planning
        cp_result = cp.plan_safe_path(perceived)
        if cp_result['success']:
            results['cp_lengths'].append(cp_result['path_length'])
            if cp_result['collisions']:
                results['cp_collisions'] += 1
    
    # Print results
    print("\n" + "="*60)
    print("CONTINUOUS RESULTS SUMMARY")
    print("="*60)
    
    naive_rate = (results['naive_collisions'] / num_trials) * 100
    cp_rate = (results['cp_collisions'] / num_trials) * 100
    
    print(f"\nCollision Rates:")
    print(f"  Naive:         {naive_rate:.1f}%")
    print(f"  Continuous CP: {cp_rate:.1f}%")
    
    if results['naive_lengths'] and results['cp_lengths']:
        print(f"\nAverage Path Lengths:")
        print(f"  Naive:         {np.mean(results['naive_lengths']):.1f} units")
        print(f"  Continuous CP: {np.mean(results['cp_lengths']):.1f} units")
    
    print(f"\nKey Observation:")
    print(f"  τ = {tau:.2f} (continuous units)")
    print(f"  Can inflate by ANY amount (1.23, 1.57, etc.)")
    print(f"  Precise coverage control possible!")
    
    return results


def visualize_continuous_system():
    """
    Visualize continuous CP system
    """
    env = ContinuousEnvironment()
    cp = ContinuousCP(env)
    cp.calibrate(num_samples=100, confidence=0.95)
    
    # Get example perception
    perceived = env.add_continuous_noise(0.3, seed=42)
    
    # Plan with CP
    result = cp.plan_safe_path(perceived)
    
    if result['success']:
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Plot 1: Perceived environment
        ax = axes[0]
        ax.set_title(f'Continuous Perception', fontsize=12)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect('equal')
        
        # Plot perceived obstacles
        for (x, y, w, h) in perceived:
            rect = patches.Rectangle((x, y), w, h,
                                    facecolor='gray', edgecolor='black', alpha=0.5)
            ax.add_patch(rect)
        
        ax.plot(5, 15, 'go', markersize=8)
        ax.plot(45, 15, 'ro', markersize=8)
        
        # Plot 2: With inflation
        ax = axes[1]
        ax.set_title(f'Continuous CP (τ={cp.tau:.2f} units)', fontsize=12)
        ax.set_xlim(0, 50)
        ax.set_ylim(0, 30)
        ax.set_aspect('equal')
        
        # Show original obstacles
        for (x, y, w, h) in perceived:
            rect = patches.Rectangle((x, y), w, h,
                                    facecolor='gray', edgecolor='black', alpha=0.5)
            ax.add_patch(rect)
        
        # Show inflated obstacles
        for (x, y, w, h) in result['inflated_obs']:
            rect = patches.Rectangle((x, y), w, h,
                                    facecolor='orange', edgecolor='red', 
                                    alpha=0.3, linestyle='--', linewidth=2)
            ax.add_patch(rect)
        
        # Plot path
        if result['path']:
            path_x = [p[0] for p in result['path']]
            path_y = [p[1] for p in result['path']]
            ax.plot(path_x, path_y, 'b-', linewidth=2, alpha=0.7)
        
        ax.plot(5, 15, 'go', markersize=8)
        ax.plot(45, 15, 'ro', markersize=8)
        
        plt.suptitle('CONTINUOUS PLANNER (RRT* with Continuous τ)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        save_path = 'continuous_planner/results/continuous_system.png'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {save_path}")
        plt.close()


if __name__ == "__main__":
    # Run continuous system
    visualize_continuous_system()
    results = run_continuous_monte_carlo(num_trials=200)
    print("\n" + "="*60)
    print("CONTINUOUS PLANNER COMPLETE")
    print("="*60)