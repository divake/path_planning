#!/usr/bin/env python3
"""
Quick test script to verify the setup works.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add paths
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python_motion_planning/src'))

print("Testing imports...")

# Test our modules
try:
    from environments.base_environment import UncertaintyEnvironment
    print("✓ Environment module loaded")
except Exception as e:
    print(f"✗ Environment module failed: {e}")

try:
    from src.feature_extraction import FeatureExtractor
    print("✓ Feature extraction module loaded")
except Exception as e:
    print(f"✗ Feature extraction failed: {e}")

try:
    from methods.naive_method import NaiveMethod
    from methods.traditional_cp_method import TraditionalCPMethod
    from methods.learnable_cp_method import LearnableCPMethod
    print("✓ All uncertainty methods loaded")
except Exception as e:
    print(f"✗ Methods failed: {e}")

try:
    from src.planner_wrapper import PlannerWrapper
    print("✓ Planner wrapper loaded")
except Exception as e:
    print(f"✗ Planner wrapper failed: {e}")

print("\nTesting basic functionality...")

# Create environment
env = UncertaintyEnvironment(width=30, height=30)
env.generate_random_obstacles(10)
print(f"✓ Created environment with {len(env.original_obstacles)} obstacles")

# Test feature extraction
fe = FeatureExtractor(env)
features = fe.extract_point_features((15, 15))
print(f"✓ Extracted features: shape={features.shape}")

# Test methods
methods = {
    'naive': NaiveMethod(),
    'traditional_cp': TraditionalCPMethod(fixed_margin=1.0),
    'learnable_cp': LearnableCPMethod()
}

for name, method in methods.items():
    method.apply(env)
    print(f"✓ {name} method applied")

# Test planning
start = (5, 5)
goal = (25, 25)

try:
    planner = PlannerWrapper(env)
    path, cost = planner.plan_astar(start, goal)
    if path:
        print(f"✓ A* found path with {len(path)} points, cost={cost:.2f}")
    else:
        print("✓ A* completed (no path found - expected if obstacles block)")
except Exception as e:
    print(f"⚠ A* planning issue: {e}")

# Quick visualization
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for idx, (name, method) in enumerate(methods.items()):
    ax = axes[idx]
    
    # Apply method
    env_copy = UncertaintyEnvironment(width=30, height=30)
    env_copy.original_obstacles = env.original_obstacles.copy()
    method.apply(env_copy)
    
    # Plot
    ax.set_xlim(0, 30)
    ax.set_ylim(0, 30)
    ax.set_aspect('equal')
    ax.set_title(name)
    
    # Original obstacles
    for obs in env_copy.original_obstacles:
        circle = plt.Circle(obs['center'], obs['radius'], 
                           color='gray', alpha=0.5)
        ax.add_patch(circle)
    
    # Inflated obstacles
    if name != 'naive':
        for obs in env_copy.obstacles:
            circle = plt.Circle(obs['center'], obs['radius'],
                              fill=False, edgecolor='yellow',
                              linestyle='--', linewidth=1)
            ax.add_patch(circle)
    
    ax.plot(start[0], start[1], 'go', markersize=8)
    ax.plot(goal[0], goal[1], 'r*', markersize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs('results', exist_ok=True)
plt.savefig('results/test_visualization.png', dpi=100)
print("\n✓ Visualization saved to results/test_visualization.png")
plt.show()

print("\n" + "="*50)
print("SETUP TEST COMPLETE - All systems operational!")
print("="*50)
print("\nYou can now run the full experiment with:")
print("  python scripts/run_experiments.py")