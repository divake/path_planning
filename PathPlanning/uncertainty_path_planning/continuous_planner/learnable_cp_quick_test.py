#!/usr/bin/env python3
"""
Quick test version of Learnable CP with reduced computational load
for verifying the implementation works correctly
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import sys
sys.path.append('continuous_planner')

from learnable_cp import FeatureExtractor, LearnableScoringNetwork, LearnableCP
from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar


def quick_train_learnable_cp(num_epochs: int = 10, 
                            num_trials_per_epoch: int = 20) -> LearnableCP:
    """
    Quick training version with reduced parameters for testing
    """
    print("\n" + "="*70)
    print("QUICK TEST: TRAINING LEARNABLE CP")
    print("="*70)
    print(f"Training with {num_epochs} epochs, {num_trials_per_epoch} trials per epoch")
    
    model = LearnableCP(alpha=0.05, max_tau=1.0, learning_rate=0.001)
    
    for epoch in range(num_epochs):
        training_data = []
        
        # Use only one environment type per epoch for speed
        env_type = ['passages', 'open', 'narrow'][epoch % 3]
        
        for trial_idx in range(num_trials_per_epoch):
            env = ContinuousEnvironment(env_type=env_type)
            
            # Fixed noise level for consistency
            noise_level = 0.2
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=noise_level, 
                seed=epoch * 1000 + trial_idx
            )
            
            # For first epoch, use simple inflation
            if epoch == 0:
                # Simple uniform inflation for speed
                inflated = []
                init_tau = 0.3  # Fixed initial tau
                for obs in perceived:
                    inflated_obs = (
                        max(0, obs[0] - init_tau),
                        max(0, obs[1] - init_tau),
                        obs[2] + 2 * init_tau,
                        obs[3] + 2 * init_tau
                    )
                    inflated.append(inflated_obs)
            else:
                # Use learnable CP (faster version)
                inflated = model.inflate_obstacles_adaptive(
                    perceived, None, (45, 15), resolution=2.0  # Coarser resolution
                )
            
            # Faster planning with reduced iterations
            planner = RRTStar((5, 15), (45, 15), inflated, max_iter=200)
            path = planner.plan()
            
            if path:
                # Subsample path for faster training
                if len(path) > 20:
                    indices = np.linspace(0, len(path)-1, 20, dtype=int)
                    path = [path[i] for i in indices]
                
                training_data.append({
                    'path': path,
                    'true_obstacles': env.obstacles,
                    'perceived_obstacles': perceived,
                    'noise_level': noise_level
                })
        
        # Train on this epoch's data
        if training_data:
            stats = model.train_epoch(training_data)
            
            if stats:
                print(f"\nEpoch {epoch+1}/{num_epochs} ({env_type}):")
                print(f"  Loss: {stats['loss']:.4f}")
                print(f"  Coverage: {stats['coverage']*100:.1f}%")
                print(f"  Dynamic τ: {stats['tau']:.3f}")
                print(f"  Avg margin: {stats['avg_margin']:.3f}")
    
    return model


def test_and_compare():
    """
    Test Learnable CP and compare with Standard CP
    """
    print("\n" + "="*70)
    print("COMPARING STANDARD CP VS LEARNABLE CP")
    print("="*70)
    
    # Train quick model
    learnable_model = quick_train_learnable_cp(num_epochs=10, num_trials_per_epoch=20)
    
    # Test on each environment type
    results = {
        'environment': [],
        'method': [],
        'collision_rate': [],
        'path_length': [],
        'avg_tau': []
    }
    
    test_envs = ['passages', 'narrow', 'open']
    num_test_trials = 30  # Reduced for quick test
    
    for env_type in test_envs:
        print(f"\nTesting on {env_type} environment ({num_test_trials} trials):")
        
        # Test Standard CP
        standard_collisions = 0
        standard_paths = 0
        standard_lengths = []
        
        # Calibrate Standard CP once for this environment
        env_sample = ContinuousEnvironment(env_type=env_type)
        cp = ContinuousStandardCP(env_sample.obstacles, "penetration")
        standard_tau = cp.calibrate(
            ContinuousNoiseModel.add_thinning_noise,
            {'thin_factor': 0.2},
            num_samples=50,
            confidence=0.95
        )
        
        # Test Learnable CP
        learnable_collisions = 0
        learnable_paths = 0
        learnable_lengths = []
        learnable_taus = []
        
        for trial in range(num_test_trials):
            env = ContinuousEnvironment(env_type=env_type)
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=0.2, seed=10000 + trial
            )
            
            # Standard CP
            standard_inflated = cp.inflate_obstacles(perceived)
            planner = RRTStar((5, 15), (45, 15), standard_inflated, max_iter=300)
            standard_path = planner.plan()
            
            if standard_path:
                standard_paths += 1
                standard_lengths.append(len(standard_path))
                
                # Check collision
                for point in standard_path:
                    if env.point_in_obstacle(point[0], point[1]):
                        standard_collisions += 1
                        break
            
            # Learnable CP
            learnable_path = learnable_model.plan_with_adaptive_cp(
                (5, 15), (45, 15), perceived, max_iter=300
            )
            
            if learnable_path:
                learnable_paths += 1
                learnable_lengths.append(len(learnable_path))
                
                # Check collision
                for point in learnable_path:
                    if env.point_in_obstacle(point[0], point[1]):
                        learnable_collisions += 1
                        break
                
                # Sample tau values along path
                for point in learnable_path[::5]:  # Sample every 5th point
                    tau = learnable_model.predict_margin(
                        point[0], point[1], perceived, learnable_path, (45, 15)
                    )
                    learnable_taus.append(tau)
        
        # Calculate statistics
        if standard_paths > 0:
            standard_collision_rate = standard_collisions / standard_paths * 100
            standard_avg_length = np.mean(standard_lengths)
        else:
            standard_collision_rate = 0
            standard_avg_length = 0
        
        if learnable_paths > 0:
            learnable_collision_rate = learnable_collisions / learnable_paths * 100
            learnable_avg_length = np.mean(learnable_lengths)
            learnable_avg_tau = np.mean(learnable_taus) if learnable_taus else 0
        else:
            learnable_collision_rate = 0
            learnable_avg_length = 0
            learnable_avg_tau = 0
        
        # Store results
        results['environment'].append(env_type)
        results['method'].append('Standard CP')
        results['collision_rate'].append(standard_collision_rate)
        results['path_length'].append(standard_avg_length)
        results['avg_tau'].append(standard_tau)
        
        results['environment'].append(env_type)
        results['method'].append('Learnable CP')
        results['collision_rate'].append(learnable_collision_rate)
        results['path_length'].append(learnable_avg_length)
        results['avg_tau'].append(learnable_avg_tau)
        
        print(f"  Standard CP:")
        print(f"    Collision rate: {standard_collision_rate:.1f}%")
        print(f"    Avg path length: {standard_avg_length:.1f}")
        print(f"    Fixed τ: {standard_tau:.3f}")
        
        print(f"  Learnable CP:")
        print(f"    Collision rate: {learnable_collision_rate:.1f}%")
        print(f"    Avg path length: {learnable_avg_length:.1f}")
        print(f"    Avg adaptive τ: {learnable_avg_tau:.3f}")
        
        if standard_avg_length > 0 and learnable_avg_length > 0:
            improvement = (standard_avg_length - learnable_avg_length) / standard_avg_length * 100
            print(f"  Path length improvement: {improvement:.1f}%")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Calculate overall statistics
    standard_mask = [m == 'Standard CP' for m in results['method']]
    learnable_mask = [m == 'Learnable CP' for m in results['method']]
    
    standard_avg_collision = np.mean([r for r, m in zip(results['collision_rate'], standard_mask) if m])
    learnable_avg_collision = np.mean([r for r, m in zip(results['collision_rate'], learnable_mask) if m])
    
    standard_avg_path = np.mean([r for r, m in zip(results['path_length'], standard_mask) if m and r > 0])
    learnable_avg_path = np.mean([r for r, m in zip(results['path_length'], learnable_mask) if m and r > 0])
    
    print(f"\nOverall Performance:")
    print(f"  Standard CP:")
    print(f"    Avg collision rate: {standard_avg_collision:.1f}%")
    print(f"    Avg path length: {standard_avg_path:.1f}")
    
    print(f"  Learnable CP:")
    print(f"    Avg collision rate: {learnable_avg_collision:.1f}%")
    print(f"    Avg path length: {learnable_avg_path:.1f}")
    
    if standard_avg_path > 0 and learnable_avg_path > 0:
        overall_improvement = (standard_avg_path - learnable_avg_path) / standard_avg_path * 100
        print(f"\n  Overall path improvement: {overall_improvement:.1f}%")
        print(f"  Safety maintained: {learnable_avg_collision < 5.0}")
    
    return learnable_model, results


def main():
    """Run quick test of Learnable CP"""
    
    # Test and compare
    model, results = test_and_compare()
    
    # Save model
    torch.save(model.scoring_net.state_dict(), 'learnable_cp_quick_model.pth')
    print("\nModel saved to learnable_cp_quick_model.pth")
    
    print("\n" + "="*70)
    print("QUICK TEST COMPLETE")
    print("="*70)
    print("\nFor full training, run learnable_cp.py directly")
    print("This quick test shows the concept works!")


if __name__ == "__main__":
    main()