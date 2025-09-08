#!/usr/bin/env python3
"""
Debug Learnable CP to understand why it's performing poorly
"""

import numpy as np
import sys
sys.path.append('/mnt/ssd1/divake/path_planning/PathPlanning/uncertainty_path_planning/continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from learnable_cp_final import FinalLearnableCP
from rrt_star_planner import RRTStar
from cross_validation_environments import get_training_environments

print("DEBUGGING LEARNABLE CP PERFORMANCE")
print("="*60)

# Initialize model
model = FinalLearnableCP(coverage=0.90, max_tau=1.0)
model.baseline_taus = {
    'passages': 0.4,
    'open': 0.3,
    'narrow': 0.5,
}

# Train the model
print("\n1. Training model...")
model.train(num_epochs=50)

# Test on a simple case
print("\n2. Testing on passages environment...")
envs = get_training_environments()
env = ContinuousEnvironment()
env.obstacles = envs['passages']

start = (2.0, 2.0)
goal = (48.0, 28.0)

# Add noise
noisy_obstacles = ContinuousNoiseModel.add_thinning_noise(
    env.obstacles, 
    thin_factor=0.15,
    seed=42
)

print(f"\nOriginal obstacles: {len(env.obstacles)}")
print(f"Noisy obstacles: {len(noisy_obstacles)}")

# Test tau predictions
print("\n3. Testing tau predictions for different obstacles:")
for i, (x, y, w, h) in enumerate(noisy_obstacles[:5]):
    cx, cy = x + w/2, y + h/2
    tau = model.predict_tau(cx, cy, noisy_obstacles, goal)
    print(f"   Obstacle {i}: center=({cx:.1f}, {cy:.1f}), tau={tau:.3f}")

# Compare planning results
print("\n4. Comparing planning results:")

# Naive
planner_naive = RRTStar(start, goal, noisy_obstacles, bounds=(0, 50, 0, 30), seed=42)
path_naive = planner_naive.plan()
if path_naive:
    print(f"   Naive: path found with {len(path_naive)} points")
    collision_naive = any(env.point_in_obstacle(p[0], p[1]) for p in path_naive)
    print(f"          collision with true obstacles: {collision_naive}")

# Learnable CP with adaptive inflation
inflated_obstacles = []
total_tau = 0
for x, y, w, h in noisy_obstacles:
    cx, cy = x + w/2, y + h/2
    tau = model.predict_tau(cx, cy, noisy_obstacles, goal)
    total_tau += tau
    inflated_obstacles.append((
        max(0, x - tau),
        max(0, y - tau),
        w + 2*tau,
        h + 2*tau
    ))

avg_tau = total_tau / len(noisy_obstacles)
print(f"\n   Learnable CP: average tau = {avg_tau:.3f}")

planner_learn = RRTStar(start, goal, inflated_obstacles, bounds=(0, 50, 0, 30), seed=42)
path_learn = planner_learn.plan()
if path_learn:
    print(f"                 path found with {len(path_learn)} points")
    collision_learn = any(env.point_in_obstacle(p[0], p[1]) for p in path_learn)
    print(f"                 collision with true obstacles: {collision_learn}")
else:
    print(f"                 NO PATH FOUND!")

# Check if obstacles are over-inflated
print("\n5. Checking obstacle inflation:")
for i in range(min(3, len(noisy_obstacles))):
    orig = noisy_obstacles[i]
    infl = inflated_obstacles[i]
    print(f"   Obstacle {i}:")
    print(f"      Original: x={orig[0]:.1f}, y={orig[1]:.1f}, w={orig[2]:.1f}, h={orig[3]:.1f}")
    print(f"      Inflated: x={infl[0]:.1f}, y={infl[1]:.1f}, w={infl[2]:.1f}, h={infl[3]:.1f}")
    print(f"      Growth: w+{infl[2]-orig[2]:.1f}, h+{infl[3]-orig[3]:.1f}")

print("\nDIAGNOSIS:")
if avg_tau > 0.5:
    print("❌ TAU TOO HIGH - obstacles are over-inflated, blocking paths")
elif avg_tau < 0.1:
    print("❌ TAU TOO LOW - not enough safety margin")
else:
    print("✓ TAU seems reasonable")

if not path_learn and path_naive:
    print("❌ CRITICAL: Learnable CP blocks paths that naive can find!")
    print("   This explains poor performance - inflated obstacles are too conservative")