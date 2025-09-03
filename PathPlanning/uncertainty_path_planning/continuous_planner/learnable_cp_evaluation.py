#!/usr/bin/env python3
"""
Comprehensive evaluation of Learnable CP
Addresses all critical questions for ICRA paper
"""

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
import json
import time
from scipy import stats
import sys
sys.path.append('continuous_planner')

from learnable_cp import FeatureExtractor, LearnableScoringNetwork, LearnableCP
from continuous_environment import ContinuousEnvironment, ContinuousNoiseModel
from continuous_standard_cp import ContinuousStandardCP
from rrt_star_planner import RRTStar


class LearnableCPEvaluator:
    """Complete evaluation framework for Learnable CP"""
    
    def __init__(self, num_trials: int = 100):
        self.num_trials = num_trials
        self.results = {}
    
    def train_full_model(self, num_epochs: int = 30, 
                        trials_per_epoch: int = 50) -> Tuple[LearnableCP, Dict]:
        """
        Train full Learnable CP model and track convergence
        """
        print("\n" + "="*70)
        print("TRAINING FULL LEARNABLE CP MODEL")
        print("="*70)
        
        model = LearnableCP(alpha=0.05, max_tau=0.5, learning_rate=0.001)
        
        convergence_data = {
            'epoch': [],
            'loss': [],
            'coverage': [],
            'tau': [],
            'avg_margin': []
        }
        
        # Use all three environments for training
        env_types = ['passages', 'open', 'narrow']
        
        for epoch in range(num_epochs):
            epoch_data = []
            
            for trial_idx in range(trials_per_epoch):
                # Rotate through environments
                env_type = env_types[trial_idx % len(env_types)]
                env = ContinuousEnvironment(env_type=env_type)
                
                # Varying noise levels
                noise_level = np.random.uniform(0.1, 0.3)
                perceived = ContinuousNoiseModel.add_thinning_noise(
                    env.obstacles, thin_factor=noise_level,
                    seed=epoch * 1000 + trial_idx
                )
                
                # For initial epochs, use Standard CP
                if epoch < 3:
                    cp = ContinuousStandardCP(env.obstacles, "penetration")
                    tau = cp.calibrate(
                        ContinuousNoiseModel.add_thinning_noise,
                        {'thin_factor': noise_level},
                        num_samples=30,  # Faster calibration
                        confidence=0.95
                    )
                    inflated = cp.inflate_obstacles(perceived)
                else:
                    # Use learnable CP
                    inflated = model.inflate_obstacles_adaptive(
                        perceived, None, (45, 15), resolution=2.0
                    )
                
                # Plan path
                planner = RRTStar((5, 15), (45, 15), inflated, max_iter=300)
                path = planner.plan()
                
                if path:
                    # Subsample for efficiency
                    if len(path) > 30:
                        indices = np.linspace(0, len(path)-1, 30, dtype=int)
                        path = [path[i] for i in indices]
                    
                    epoch_data.append({
                        'path': path,
                        'true_obstacles': env.obstacles,
                        'perceived_obstacles': perceived,
                        'noise_level': noise_level,
                        'env_type': env_type
                    })
            
            # Train on epoch data
            if epoch_data:
                stats = model.train_epoch(epoch_data)
                
                if stats:
                    convergence_data['epoch'].append(epoch)
                    convergence_data['loss'].append(stats['loss'])
                    convergence_data['coverage'].append(stats['coverage'])
                    convergence_data['tau'].append(stats['tau'])
                    convergence_data['avg_margin'].append(stats['avg_margin'])
                    
                    if epoch % 5 == 0:
                        print(f"\nEpoch {epoch+1}/{num_epochs}:")
                        print(f"  Loss: {stats['loss']:.4f}")
                        print(f"  Coverage: {stats['coverage']*100:.1f}%")
                        print(f"  Dynamic τ: {stats['tau']:.3f}")
                        print(f"  Avg margin: {stats['avg_margin']:.3f}")
        
        return model, convergence_data
    
    def evaluate_method(self, method: str, model: Optional[LearnableCP] = None) -> Dict:
        """
        Evaluate a single method (Naive, Standard CP, or Learnable CP)
        """
        print(f"\nEvaluating {method}...")
        
        results = {
            'collisions': 0,
            'paths_found': 0,
            'path_lengths': [],
            'computation_times': [],
            'tau_values': [],
            'env_results': {
                'passages': {'collisions': 0, 'paths': 0, 'lengths': []},
                'open': {'collisions': 0, 'paths': 0, 'lengths': []},
                'narrow': {'collisions': 0, 'paths': 0, 'lengths': []}
            }
        }
        
        for trial in range(self.num_trials):
            # Rotate through environments
            env_type = ['passages', 'open', 'narrow'][trial % 3]
            env = ContinuousEnvironment(env_type=env_type)
            
            # Fixed noise level for fair comparison
            noise_level = 0.2
            perceived = ContinuousNoiseModel.add_thinning_noise(
                env.obstacles, thin_factor=noise_level,
                seed=50000 + trial  # Different seed for testing
            )
            
            start_time = time.time()
            
            if method == 'Naive':
                # No inflation
                inflated = perceived
                tau = 0.0
                
            elif method == 'Standard CP':
                # Fixed tau calibration
                cp = ContinuousStandardCP(env.obstacles, "penetration")
                tau = cp.calibrate(
                    ContinuousNoiseModel.add_thinning_noise,
                    {'thin_factor': noise_level},
                    num_samples=100,
                    confidence=0.95
                )
                inflated = cp.inflate_obstacles(perceived)
                results['tau_values'].append(tau)
                
            elif method == 'Learnable CP' and model:
                # Adaptive planning
                path = model.plan_with_adaptive_cp(
                    (5, 15), (45, 15), perceived, max_iter=500
                )
                
                # Track adaptive tau values
                if path:
                    for point in path[::5]:
                        tau = model.predict_margin(
                            point[0], point[1], perceived, path, (45, 15)
                        )
                        results['tau_values'].append(tau)
                
                # For timing, we already have the path
                comp_time = time.time() - start_time
                
                if path:
                    results['paths_found'] += 1
                    results['path_lengths'].append(len(path))
                    results['computation_times'].append(comp_time)
                    
                    # Check collision
                    collision = False
                    for point in path:
                        if env.point_in_obstacle(point[0], point[1]):
                            collision = True
                            results['collisions'] += 1
                            break
                    
                    # Environment-specific results
                    results['env_results'][env_type]['paths'] += 1
                    results['env_results'][env_type]['lengths'].append(len(path))
                    if collision:
                        results['env_results'][env_type]['collisions'] += 1
                
                continue  # Skip regular planning for Learnable CP
            
            # Plan path for Naive and Standard CP
            planner = RRTStar((5, 15), (45, 15), inflated, max_iter=500)
            path = planner.plan()
            
            comp_time = time.time() - start_time
            
            if path:
                results['paths_found'] += 1
                results['path_lengths'].append(len(path))
                results['computation_times'].append(comp_time)
                
                # Check collision with true obstacles
                collision = False
                for point in path:
                    if env.point_in_obstacle(point[0], point[1]):
                        collision = True
                        results['collisions'] += 1
                        break
                
                # Environment-specific results
                results['env_results'][env_type]['paths'] += 1
                results['env_results'][env_type]['lengths'].append(len(path))
                if collision:
                    results['env_results'][env_type]['collisions'] += 1
        
        # Calculate statistics
        if results['paths_found'] > 0:
            results['collision_rate'] = results['collisions'] / results['paths_found'] * 100
            results['avg_path_length'] = np.mean(results['path_lengths'])
            results['std_path_length'] = np.std(results['path_lengths'])
            results['avg_comp_time'] = np.mean(results['computation_times'])
            
            if results['tau_values']:
                results['avg_tau'] = np.mean(results['tau_values'])
                results['std_tau'] = np.std(results['tau_values'])
                results['min_tau'] = np.min(results['tau_values'])
                results['max_tau'] = np.max(results['tau_values'])
            else:
                results['avg_tau'] = tau if method != 'Learnable CP' else 0
        
        return results
    
    def generate_comparison_table(self, naive_results: Dict, 
                                 standard_results: Dict,
                                 learnable_results: Dict) -> None:
        """
        Generate comprehensive comparison table for ICRA paper
        """
        print("\n" + "="*70)
        print("COMPREHENSIVE COMPARISON TABLE")
        print("="*70)
        
        # Main comparison table
        print("\n### Overall Performance (100 trials per method)")
        print("-" * 80)
        print(f"{'Method':<15} {'Collision Rate':<15} {'Avg Path Length':<18} "
              f"{'Avg τ':<12} {'Comp Time (ms)':<15}")
        print("-" * 80)
        
        # Naive
        print(f"{'Naive':<15} "
              f"{naive_results.get('collision_rate', 0):<15.1f}% "
              f"{naive_results.get('avg_path_length', 0):<18.1f} "
              f"{'0.000':<12} "
              f"{naive_results.get('avg_comp_time', 0)*1000:<15.1f}")
        
        # Standard CP
        print(f"{'Standard CP':<15} "
              f"{standard_results.get('collision_rate', 0):<15.1f}% "
              f"{standard_results.get('avg_path_length', 0):<18.1f} "
              f"{standard_results.get('avg_tau', 0):<12.3f} "
              f"{standard_results.get('avg_comp_time', 0)*1000:<15.1f}")
        
        # Learnable CP
        print(f"{'Learnable CP':<15} "
              f"{learnable_results.get('collision_rate', 0):<15.1f}% "
              f"{learnable_results.get('avg_path_length', 0):<18.1f} "
              f"{learnable_results.get('avg_tau', 0):<12.3f} "
              f"{learnable_results.get('avg_comp_time', 0)*1000:<15.1f}")
        
        print("-" * 80)
        
        # Calculate improvements
        if standard_results.get('avg_path_length', 0) > 0 and learnable_results.get('avg_path_length', 0) > 0:
            path_improvement = (standard_results['avg_path_length'] - learnable_results['avg_path_length']) / standard_results['avg_path_length'] * 100
            print(f"\nPath Length Improvement (Learnable vs Standard): {path_improvement:.1f}%")
        
        # Per-environment breakdown
        print("\n### Per-Environment Collision Rates")
        print("-" * 60)
        print(f"{'Environment':<15} {'Naive':<15} {'Standard CP':<15} {'Learnable CP':<15}")
        print("-" * 60)
        
        for env_type in ['passages', 'open', 'narrow']:
            naive_env = naive_results['env_results'][env_type]
            standard_env = standard_results['env_results'][env_type]
            learnable_env = learnable_results['env_results'][env_type]
            
            naive_rate = (naive_env['collisions'] / naive_env['paths'] * 100) if naive_env['paths'] > 0 else 0
            standard_rate = (standard_env['collisions'] / standard_env['paths'] * 100) if standard_env['paths'] > 0 else 0
            learnable_rate = (learnable_env['collisions'] / learnable_env['paths'] * 100) if learnable_env['paths'] > 0 else 0
            
            print(f"{env_type.capitalize():<15} "
                  f"{naive_rate:<15.1f}% "
                  f"{standard_rate:<15.1f}% "
                  f"{learnable_rate:<15.1f}%")
        
        # Statistical significance test
        print("\n### Statistical Significance (Welch's t-test)")
        print("-" * 60)
        
        if len(standard_results['path_lengths']) > 1 and len(learnable_results['path_lengths']) > 1:
            t_stat, p_value = stats.ttest_ind(
                standard_results['path_lengths'],
                learnable_results['path_lengths'],
                equal_var=False  # Welch's t-test
            )
            print(f"Path length difference p-value: {p_value:.4f}")
            print(f"Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
        
        # Coverage guarantee check
        print("\n### Coverage Guarantee Verification")
        print("-" * 60)
        target_coverage = 95.0
        
        for method, results in [('Standard CP', standard_results), ('Learnable CP', learnable_results)]:
            collision_rate = results.get('collision_rate', 0)
            coverage = 100 - collision_rate
            meets_guarantee = collision_rate <= 5.0
            
            print(f"{method}: {coverage:.1f}% coverage ({'✓ MEETS' if meets_guarantee else '✗ VIOLATES'} 95% guarantee)")
        
        # Adaptive behavior statistics for Learnable CP
        if learnable_results.get('tau_values'):
            print("\n### Learnable CP Adaptive τ Statistics")
            print("-" * 60)
            print(f"Average τ: {learnable_results['avg_tau']:.3f}")
            print(f"Std Dev τ: {learnable_results.get('std_tau', 0):.3f}")
            print(f"Min τ: {learnable_results.get('min_tau', 0):.3f}")
            print(f"Max τ: {learnable_results.get('max_tau', 0):.3f}")
            print(f"Range: {learnable_results.get('max_tau', 0) - learnable_results.get('min_tau', 0):.3f}")
    
    def plot_convergence(self, convergence_data: Dict) -> None:
        """
        Plot training convergence behavior
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = convergence_data['epoch']
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, convergence_data['loss'], 'b-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss Convergence')
        ax.grid(True, alpha=0.3)
        
        # Coverage
        ax = axes[0, 1]
        ax.plot(epochs, np.array(convergence_data['coverage']) * 100, 'g-', linewidth=2)
        ax.axhline(y=95, color='r', linestyle='--', label='Target 95%')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Coverage Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Dynamic tau
        ax = axes[1, 0]
        ax.plot(epochs, convergence_data['tau'], 'r-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Dynamic τ')
        ax.set_title('τ Adaptation During Training')
        ax.grid(True, alpha=0.3)
        
        # Average margin
        ax = axes[1, 1]
        ax.plot(epochs, convergence_data['avg_margin'], 'm-', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Average Margin')
        ax.set_title('Predicted Margin Evolution')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Learnable CP Training Convergence', fontsize=14, weight='bold')
        plt.tight_layout()
        plt.savefig('results/learnable_cp_convergence.png', dpi=150, bbox_inches='tight')
        print("\nConvergence plot saved to results/learnable_cp_convergence.png")
    
    def test_generalization(self, model: LearnableCP) -> None:
        """
        Test model generalization to unseen obstacle configurations
        """
        print("\n" + "="*70)
        print("TESTING GENERALIZATION")
        print("="*70)
        
        # Create custom test environments not seen during training
        test_configs = [
            {
                'name': 'Dense Random',
                'obstacles': [(np.random.uniform(5, 40), np.random.uniform(5, 25), 3, 3) 
                             for _ in range(15)]
            },
            {
                'name': 'Diagonal Passage',
                'obstacles': [(10 + i*2, 10 + i, 3, 2) for i in range(10)]
            },
            {
                'name': 'Circular Pattern',
                'obstacles': [(25 + 10*np.cos(i*np.pi/4), 15 + 10*np.sin(i*np.pi/4), 2, 2) 
                             for i in range(8)]
            }
        ]
        
        print("\nTesting on unseen obstacle configurations:")
        print("-" * 50)
        
        for config in test_configs:
            # Add boundary walls
            obstacles = [
                (0, 0, 50, 1),
                (0, 0, 1, 30),
                (0, 29, 50, 1),
                (49, 0, 1, 30)
            ] + config['obstacles']
            
            perceived = obstacles  # No noise for this test
            
            # Get adaptive tau values
            tau_values = []
            for x in np.linspace(5, 45, 20):
                for y in np.linspace(5, 25, 10):
                    tau = model.predict_margin(x, y, obstacles)
                    tau_values.append(tau)
            
            avg_tau = np.mean(tau_values)
            std_tau = np.std(tau_values)
            
            print(f"\n{config['name']}:")
            print(f"  Average τ: {avg_tau:.3f}")
            print(f"  Std Dev τ: {std_tau:.3f}")
            print(f"  Adaptation range: [{np.min(tau_values):.3f}, {np.max(tau_values):.3f}]")


def main():
    """
    Run complete evaluation of Learnable CP
    """
    evaluator = LearnableCPEvaluator(num_trials=100)
    
    # Train model
    print("\nTraining Learnable CP model...")
    model, convergence_data = evaluator.train_full_model(
        num_epochs=20,  # Reduced for reasonable runtime
        trials_per_epoch=30
    )
    
    # Evaluate all methods
    naive_results = evaluator.evaluate_method('Naive')
    standard_results = evaluator.evaluate_method('Standard CP')
    learnable_results = evaluator.evaluate_method('Learnable CP', model)
    
    # Generate comparison table
    evaluator.generate_comparison_table(naive_results, standard_results, learnable_results)
    
    # Plot convergence
    evaluator.plot_convergence(convergence_data)
    
    # Test generalization
    evaluator.test_generalization(model)
    
    # Save results to JSON
    all_results = {
        'naive': naive_results,
        'standard_cp': standard_results,
        'learnable_cp': learnable_results,
        'convergence': convergence_data
    }
    
    # Remove non-serializable items
    for method in ['naive', 'standard_cp', 'learnable_cp']:
        if 'computation_times' in all_results[method]:
            all_results[method]['computation_times'] = list(all_results[method]['computation_times'])
        if 'path_lengths' in all_results[method]:
            all_results[method]['path_lengths'] = list(all_results[method]['path_lengths'])
        if 'tau_values' in all_results[method]:
            all_results[method]['tau_values'] = list(all_results[method]['tau_values'])
    
    with open('results/learnable_cp_evaluation.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print("\nResults saved to results/learnable_cp_evaluation.json")
    
    # Final summary for paper
    print("\n### KEY FINDINGS FOR ICRA PAPER ###")
    print("-" * 50)
    
    safety_maintained = learnable_results.get('collision_rate', 100) <= 5.0
    if standard_results.get('avg_path_length', 0) > 0 and learnable_results.get('avg_path_length', 0) > 0:
        efficiency_gain = (standard_results['avg_path_length'] - learnable_results['avg_path_length']) / standard_results['avg_path_length'] * 100
    else:
        efficiency_gain = 0
    
    print(f"1. Safety Guarantee: {'✓ MAINTAINED' if safety_maintained else '✗ VIOLATED'}")
    print(f"2. Efficiency Gain: {efficiency_gain:.1f}% shorter paths than Standard CP")
    print(f"3. Adaptive Range: τ ∈ [{learnable_results.get('min_tau', 0):.3f}, {learnable_results.get('max_tau', 0):.3f}]")
    print(f"4. Computation Overhead: {(learnable_results.get('avg_comp_time', 0) - standard_results.get('avg_comp_time', 0))*1000:.1f}ms")
    
    return model, all_results


if __name__ == "__main__":
    main()