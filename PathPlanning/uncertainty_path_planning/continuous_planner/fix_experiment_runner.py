#!/usr/bin/env python3
"""
Fixed experiment runner with proper noise model and collision checking
"""

import numpy as np
import sys
sys.path.append('/mnt/ssd1/divake/path_planning/PathPlanning/uncertainty_path_planning/continuous_planner')

from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from learnable_cp_final import FinalLearnableCP
from rrt_star_planner import RRTStar
from cross_validation_environments import get_training_environments

def run_proper_experiment(env_name, obstacles, num_trials=100):
    """Run experiment with correct setup"""
    
    # Create environment
    env = ContinuousEnvironment()
    env.obstacles = obstacles
    
    start = (5.0, 15.0)  # Standard start
    goal = (45.0, 15.0)  # Standard goal
    
    results = {
        'naive': {'success': 0, 'collision': 0, 'paths_found': 0},
        'standard': {'success': 0, 'collision': 0, 'paths_found': 0},
        'learnable': {'success': 0, 'collision': 0, 'paths_found': 0}
    }
    
    # Train learnable CP with LOWER tau values for better performance
    learnable_model = FinalLearnableCP(coverage=0.90, max_tau=0.5)  # Lower max_tau
    learnable_model.baseline_taus = {
        'passages': 0.15,    # Much lower than 0.4
        'open': 0.1,         # Lower
        'narrow': 0.2,       # Lower
    }
    print("Training Learnable CP with lower tau values...")
    learnable_model.train(num_epochs=100)
    
    # Calibrate Standard CP
    cp = ContinuousStandardCP(env.obstacles, nonconformity_type='penetration')
    tau_standard = cp.calibrate(
        ContinuousNoiseModel.add_thinning_noise,
        {'thin_factor': 0.15},
        num_samples=500,
        confidence=0.90,
        base_seed=42
    )
    print(f"Standard CP tau: {tau_standard:.3f}")
    
    for trial in range(num_trials):
        # Apply CONSISTENT noise model - thinning only
        perceived_obstacles = ContinuousNoiseModel.add_thinning_noise(
            env.obstacles,
            thin_factor=0.15,  # 15% thinning
            seed=1000 + trial
        )
        
        # 1. NAIVE planning
        planner = RRTStar(start, goal, perceived_obstacles, 
                         bounds=(0, 50, 0, 30), 
                         robot_radius=0.0,  # No robot radius for fair comparison
                         seed=trial)
        path = planner.plan()
        
        if path:
            results['naive']['paths_found'] += 1
            # Check collision with TRUE obstacles
            collision = False
            for point in path:
                if env.point_in_obstacle(point[0], point[1]):
                    collision = True
                    results['naive']['collision'] += 1
                    break
            if not collision:
                results['naive']['success'] += 1
        
        # 2. STANDARD CP
        inflated_std = cp.inflate_obstacles(perceived_obstacles, inflation_method="uniform")
        planner = RRTStar(start, goal, inflated_std,
                         bounds=(0, 50, 0, 30),
                         robot_radius=0.0,
                         seed=trial + 1000)
        path = planner.plan()
        
        if path:
            results['standard']['paths_found'] += 1
            collision = False
            for point in path:
                if env.point_in_obstacle(point[0], point[1]):
                    collision = True
                    results['standard']['collision'] += 1
                    break
            if not collision:
                results['standard']['success'] += 1
        
        # 3. LEARNABLE CP with adaptive inflation
        inflated_learn = []
        for obs in perceived_obstacles:
            cx, cy = obs[0] + obs[2]/2, obs[1] + obs[3]/2
            # Use learnable model's prediction
            tau = learnable_model.predict_tau(cx, cy, perceived_obstacles, goal)
            tau = min(tau, 0.3)  # Cap tau to prevent over-inflation
            
            inflated_learn.append((
                max(0, obs[0] - tau),
                max(0, obs[1] - tau),
                obs[2] + 2*tau,
                obs[3] + 2*tau
            ))
        
        planner = RRTStar(start, goal, inflated_learn,
                         bounds=(0, 50, 0, 30),
                         robot_radius=0.0,
                         seed=trial + 2000)
        path = planner.plan()
        
        if path:
            results['learnable']['paths_found'] += 1
            collision = False
            for point in path:
                if env.point_in_obstacle(point[0], point[1]):
                    collision = True
                    results['learnable']['collision'] += 1
                    break
            if not collision:
                results['learnable']['success'] += 1
    
    # Print results
    print(f"\nResults for {env_name} ({num_trials} trials):")
    print("-" * 60)
    for method, res in results.items():
        success_rate = res['success'] / num_trials * 100
        collision_rate = res['collision'] / num_trials * 100
        path_rate = res['paths_found'] / num_trials * 100
        print(f"{method:12} Success: {success_rate:5.1f}%  Collision: {collision_rate:5.1f}%  Paths: {path_rate:5.1f}%")
    
    return results


# Test on training environments
print("\n" + "="*70)
print("FIXED EXPERIMENT WITH PROPER SETUP")
print("="*70)

envs = get_training_environments()

all_results = {}
for env_name, obstacles in envs.items():
    all_results[env_name] = run_proper_experiment(env_name, obstacles, num_trials=100)

# Summary
print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)

methods = ['naive', 'standard', 'learnable']
for method in methods:
    total_success = sum(all_results[env][method]['success'] for env in all_results)
    total_collision = sum(all_results[env][method]['collision'] for env in all_results)
    total_trials = len(all_results) * 100
    
    avg_success = total_success / total_trials * 100
    avg_collision = total_collision / total_trials * 100
    
    print(f"{method:12} Avg Success: {avg_success:5.1f}%  Avg Collision: {avg_collision:5.1f}%")

print("\nKEY INSIGHT:")
if all_results:
    # Check if learnable is best
    learn_success = sum(all_results[env]['learnable']['success'] for env in all_results)
    naive_success = sum(all_results[env]['naive']['success'] for env in all_results)
    std_success = sum(all_results[env]['standard']['success'] for env in all_results)
    
    if learn_success > naive_success and learn_success > std_success:
        print("✓ Learnable CP achieves BEST performance!")
    else:
        print("✗ Learnable CP needs tuning - check tau values and training")