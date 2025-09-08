#!/usr/bin/env python3
"""
Quick test to verify experiment setup works
"""

import sys
sys.path.append('/mnt/ssd1/divake/path_planning/PathPlanning/uncertainty_path_planning/continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar
from cross_validation_environments import get_training_environments

print("Testing experiment setup...")

# Get one environment
envs = get_training_environments()
env_name = 'passages'
obstacles = envs[env_name]

print(f"\nTesting with {env_name} environment")
print(f"Number of obstacles: {len(obstacles)}")

# Create environment
env = ContinuousEnvironment()
env.obstacles = obstacles

# Test planning
start = (2.0, 2.0)
goal = (48.0, 28.0)

print(f"Start: {start}, Goal: {goal}")

# Test naive planning
noisy_obstacles = ContinuousNoiseModel.add_thinning_noise(
    env.obstacles, 
    thin_factor=0.15,
    seed=42
)

planner = RRTStar(start, goal, noisy_obstacles, bounds=(0, 50, 0, 30), seed=42)
path = planner.plan()

if path:
    print(f"Path found with {len(path)} points")
    
    # Check collision
    collision = False
    for point in path:
        if env.point_in_obstacle(point[0], point[1]):
            collision = True
            break
    
    print(f"Collision with true obstacles: {collision}")
else:
    print("No path found")

# Test Standard CP
print("\nTesting Standard CP...")
cp = ContinuousStandardCP(env.obstacles, nonconformity_type='penetration')
tau = cp.calibrate(
    ContinuousNoiseModel.add_thinning_noise,
    {'thin_factor': 0.15},
    num_samples=100,
    confidence=0.90,
    seed=42
)
print(f"Calibrated tau: {tau:.3f}")

print("\n✓ Setup test complete!")