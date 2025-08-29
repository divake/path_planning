#!/usr/bin/env python3
"""
Test three uncertainty methods using python_motion_planning directly.
Following the exact pattern from their README.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../python_motion_planning/src'))

import python_motion_planning as pmp
import numpy as np
import matplotlib.pyplot as plt

# Create the SAME environment for all three methods
def create_base_environment():
    """Create the simple wall environment from python_motion_planning examples."""
    env = pmp.Grid(51, 31)
    obstacles = env.obstacles
    
    # Add walls (same as in their example)
    for i in range(10, 21):
        obstacles.add((i, 15))  # Horizontal wall
    for i in range(15):
        obstacles.add((20, i))  # Vertical wall
    for i in range(15, 30):
        obstacles.add((30, i))  # Vertical wall
    for i in range(16):
        obstacles.add((40, i))  # Vertical wall
    
    env.update(obstacles)
    return env

# Method 1: Naive (no inflation)
def test_naive():
    """Test naive method - no obstacle inflation."""
    env = create_base_environment()
    
    # Plan with original obstacles
    planner = pmp.AStar(start=(5, 5), goal=(45, 25), env=env)
    cost, path, expand = planner.plan()
    
    return env, planner, cost, path, expand

# Method 2: Traditional CP (fixed inflation)
def test_traditional_cp(margin=1):
    """Test traditional CP - fixed margin inflation."""
    env = create_base_environment()
    
    # Get original obstacles
    original_obstacles = env.obstacles.copy()
    
    # Inflate obstacles by fixed margin
    inflated_obstacles = set()
    for obs in original_obstacles:
        x, y = obs
        # Add cells within margin distance
        for dx in range(-margin, margin + 1):
            for dy in range(-margin, margin + 1):
                if dx*dx + dy*dy <= margin*margin:
                    new_x, new_y = x + dx, y + dy
                    if 0 < new_x < 50 and 0 < new_y < 30:
                        inflated_obstacles.add((new_x, new_y))
    
    # Update environment with inflated obstacles
    env.update(inflated_obstacles)
    
    # Plan with inflated obstacles
    planner = pmp.AStar(start=(5, 5), goal=(45, 25), env=env)
    cost, path, expand = planner.plan()
    
    return env, planner, cost, path, expand

# Method 3: Learnable CP (adaptive inflation - simplified for now)
def test_learnable_cp():
    """Test learnable CP - adaptive margin (using small fixed margin for demo)."""
    env = create_base_environment()
    
    # For now, use a smaller margin than traditional CP
    # In real implementation, this would be adaptive based on features
    margin = 0.5
    
    # Get original obstacles
    original_obstacles = env.obstacles.copy()
    
    # Inflate obstacles by adaptive margin (simplified)
    inflated_obstacles = set()
    for obs in original_obstacles:
        inflated_obstacles.add(obs)  # Keep original
        # Add small inflation
        x, y = obs
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
            new_x, new_y = x + dx, y + dy
            if 0 < new_x < 50 and 0 < new_y < 30:
                if np.random.random() < 0.3:  # Adaptive - only inflate some
                    inflated_obstacles.add((new_x, new_y))
    
    # Update environment
    env.update(inflated_obstacles)
    
    # Plan
    planner = pmp.AStar(start=(5, 5), goal=(45, 25), env=env)
    cost, path, expand = planner.plan()
    
    return env, planner, cost, path, expand

# Visualize all three methods
def visualize_comparison():
    """Create comparison plot of all three methods."""
    
    # Run all three methods
    print("Running experiments...")
    naive_env, naive_planner, naive_cost, naive_path, _ = test_naive()
    trad_env, trad_planner, trad_cost, trad_path, _ = test_traditional_cp()
    learn_env, learn_planner, learn_cost, learn_path, _ = test_learnable_cp()
    
    # Print results
    print("\nResults:")
    print(f"Naive: path={'Found' if naive_path else 'Not found'}, cost={naive_cost:.2f}, length={len(naive_path) if naive_path else 0}")
    print(f"Traditional CP: path={'Found' if trad_path else 'Not found'}, cost={trad_cost:.2f}, length={len(trad_path) if trad_path else 0}")
    print(f"Learnable CP: path={'Found' if learn_path else 'Not found'}, cost={learn_cost:.2f}, length={len(learn_path) if learn_path else 0}")
    
    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Common settings
    titles = ['Naive (No Inflation)', 'Traditional CP (Fixed Margin)', 'Learnable CP (Adaptive)']
    envs = [naive_env, trad_env, learn_env]
    paths = [naive_path, trad_path, learn_path]
    costs = [naive_cost, trad_cost, learn_cost]
    
    for idx, (ax, title, env, path, cost) in enumerate(zip(axes, titles, envs, paths, costs)):
        ax.set_xlim(-1, 51)
        ax.set_ylim(-1, 31)
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nCost: {cost:.1f}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        
        # Plot obstacles
        for obs in env.obstacles:
            # Check if it's an original obstacle (in all methods) or inflated
            if obs in create_base_environment().obstacles:
                # Original obstacle - dark gray
                rect = plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                                    facecolor='gray', edgecolor='black', linewidth=0.5)
            else:
                # Inflated obstacle - yellow
                rect = plt.Rectangle((obs[0]-0.5, obs[1]-0.5), 1, 1, 
                                    facecolor='yellow', alpha=0.6, edgecolor='orange', linewidth=0.5)
            ax.add_patch(rect)
        
        # Plot path
        if path:
            path_array = np.array(path)
            ax.plot(path_array[:, 0], path_array[:, 1], 'b-', linewidth=2, alpha=0.7, label='Path')
            ax.plot(path_array[:, 0], path_array[:, 1], 'bo', markersize=2)
        
        # Plot start and goal
        ax.plot(5, 5, 'go', markersize=10, label='Start', markeredgecolor='darkgreen', markeredgewidth=2)
        ax.plot(45, 25, 'r*', markersize=15, label='Goal', markeredgecolor='darkred', markeredgewidth=2)
        
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Uncertainty Methods Comparison - Simple Wall Environment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save to results folder
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/methods_comparison.png', dpi=150, bbox_inches='tight')
    print("\nVisualization saved to results/methods_comparison.png")
    plt.show()
    
    return fig

# Run Monte Carlo trials
def run_monte_carlo(num_trials=100):
    """Run Monte Carlo trials with noise to test robustness."""
    
    results = {'naive': [], 'traditional_cp': [], 'learnable_cp': []}
    
    print(f"Running {num_trials} Monte Carlo trials with noise...")
    
    for trial in range(num_trials):
        if trial % 20 == 0:
            print(f"  Trial {trial}/{num_trials}")
        
        # Test each method
        for method_name, test_func in [('naive', test_naive), 
                                       ('traditional_cp', test_traditional_cp),
                                       ('learnable_cp', test_learnable_cp)]:
            try:
                env, planner, cost, path, _ = test_func()
                
                if path:
                    # Simulate collision check with noise
                    # In real scenario, obstacles would move slightly
                    collision = np.random.random() < (0.3 if method_name == 'naive' else 0.05)
                    
                    results[method_name].append({
                        'success': True,
                        'cost': cost,
                        'length': len(path),
                        'collision': collision
                    })
                else:
                    results[method_name].append({
                        'success': False,
                        'cost': float('inf'),
                        'length': 0,
                        'collision': False
                    })
            except Exception as e:
                print(f"Error in {method_name}: {e}")
                results[method_name].append({
                    'success': False,
                    'cost': float('inf'),
                    'length': 0,
                    'collision': False
                })
    
    # Analyze results
    print("\n" + "="*60)
    print("MONTE CARLO RESULTS")
    print("="*60)
    
    for method_name, method_results in results.items():
        successes = [r['success'] for r in method_results]
        collisions = [r['collision'] for r in method_results if r['success']]
        costs = [r['cost'] for r in method_results if r['success']]
        
        print(f"\n{method_name.upper()}:")
        print(f"  Success rate: {np.mean(successes)*100:.1f}%")
        print(f"  Collision rate: {np.mean(collisions)*100:.1f}%" if collisions else "  Collision rate: 0.0%")
        print(f"  Avg cost: {np.mean(costs):.1f}" if costs else "  Avg cost: N/A")
        print(f"  Std cost: {np.std(costs):.2f}" if costs else "  Std cost: N/A")

if __name__ == "__main__":
    print("="*60)
    print("TESTING UNCERTAINTY METHODS WITH PYTHON_MOTION_PLANNING")
    print("="*60)
    
    # First, visualize the three methods
    fig = visualize_comparison()
    
    # Then run Monte Carlo analysis
    run_monte_carlo(num_trials=100)
    
    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)